"""Stream candidate scores for a selected public noisy population."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from qldpc_identification.inference import _pm_count_aware as pm_core
from qldpc_identification.inference._dimension_adapter import Resources, engine
from qldpc_identification.inference._pm_count_aware import sum_upper
from qldpc_identification.inference._t_mpjp import log_nonnegative
from qldpc_identification.inference.scalable_bv import BVScalableScorer, load_public_records
from qldpc_identification.inference.scalable_mixed import (
    build_model, conditioned_basis, load_conditioned_model, load_public_case, score_record,
)
from qldpc_identification.inference.scalable_pauli import load_prepared_model as load_pm
from qldpc_identification.inference.scalable_t_injection import load_prepared_model as load_t


def _emit(candidate, index, point, lower=None, upper=None, kind="retained_likelihood"):
    return (index, candidate, kind, point, lower if lower is not None else "",
            upper if upper is not None else "")


def _pm_rows(data_dir, start, stop, selected):
    data, factors, terms = load_pm(data_dir / "pm_low_model.npz")
    pm_core.CANDIDATES = tuple(data["candidate_labels"])
    if selected and selected not in pm_core.CANDIDATES:
        raise ValueError("unknown Pauli measurement candidate")
    pm_core.C = (1,) * 10
    pm_core.CPLUS = (2,) * 10
    pm_core.num.TIME_LIMIT = 10**9
    for ci, candidate in enumerate(pm_core.CANDIDATES):
        if selected and candidate != selected:
            continue
        resources = pm_core.Resources(time.perf_counter())
        model = pm_core.build_candidate(ci, data, factors, terms[ci], resources)
        for index in range(start, stop):
            result = pm_core.score(model, data["public_boundary"][index], resources)
            yield _emit(candidate, index,
                        result["qCplus_retained"] * result["likelihood_factor"],
                        result["L_FULL_lower"], result["L_FULL_upper"])


def _t_rows(data_dir, start, stop, selected):
    data, factors, terms = load_t(data_dir / "t_low_model.npz")
    axes = tuple(data["candidate_labels"])
    if selected and selected not in axes:
        raise ValueError("unknown T-injection axis")
    caps = {"C": (1,) * 10 + (0,), "CPLUS": (2,) * 10 + (1,)}
    for ci, candidate in enumerate(axes):
        if selected and candidate != selected:
            continue
        namespace = engine(41, 7, caps, 11)
        namespace["CANDIDATES"] = axes
        resources = Resources(seconds=10**9, gib=12)
        model = namespace["build_candidate"](ci, data, factors, terms[ci], resources)
        model["mpjp_log"] = log_nonnegative(model["projected_upper"])
        last = model["projected_upper"][:, -1, :].reshape((-1, 2, 128))
        model["mpjp_last"] = log_nonnegative(sum_upper(last, axis=-1))
        for index in range(start, stop):
            result = namespace["score"](model, data["public_joint_raw"][index], resources)
            yield _emit(candidate, index,
                        result["qCplus_retained"] * result["likelihood_factor"],
                        result["L_FULL_lower"], result["L_FULL_upper"])


def _mixed_rows(data_dir, start, stop, selected):
    config, raw, channels = load_public_case(data_dir)
    if selected and selected not in config["candidates"]:
        raise ValueError("unknown mixed-circuit candidate")
    for candidate in config["candidates"]:
        if selected and candidate != selected:
            continue
        groups = {}
        for index in range(start, stop):
            basis = conditioned_basis(candidate, raw[index], config, channels)
            groups.setdefault(basis, []).append(index)
        for basis, indices in groups.items():
            doc = load_conditioned_model(data_dir, candidate, basis)
            resources = Resources(seconds=10**9, gib=14)
            model, scorer = build_model(doc, config, resources)
            for index in indices:
                result = score_record(doc, model, scorer, raw[index], resources)
                yield _emit(candidate, index, result["point_retained"],
                            result["full_lower"], result["full_upper"])


def _bv_rows(data_dir, start, stop, selected):
    if selected:
        raise ValueError("BV evaluates all four circuits together; omit --candidate")
    records = load_public_records(data_dir)
    scorer = BVScalableScorer(data_dir)
    for index in range(start, stop):
        result = scorer.score(records[index])["boundaries"][-1]["normalized_scores"]
        for candidate, value in result.items():
            yield _emit(candidate, index, value, kind="normalized_approximate_score")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("pm", "t", "mixed", "bv"), required=True)
    parser.add_argument("--candidate", help="Optional candidate family for PM, T, or mixed")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.start < args.stop <= 1024:
        parser.error("record range must be within [0, 1024)")
    data = Path(__file__).resolve().parents[1] / "data"
    cases = {"pm": (_pm_rows, data / "code_18_4_2"),
             "t": (_t_rows, data / "code_18_4_2"),
             "mixed": (_mixed_rows, data / "code_18_4_2"),
             "bv": (_bv_rows, data / "code_72_12_6" / "bv")}
    runner, data_dir = cases[args.case]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("record_index", "candidate", "score_kind", "point",
                         "full_lower", "full_upper"))
        for row in runner(data_dir, args.start, args.stop, args.candidate):
            writer.writerow(row)
            stream.flush()


if __name__ == "__main__":
    main()
