"""Scalable likelihood for the full [[18,4,2]] mixed circuit."""

from __future__ import annotations

import gzip
import json
import math
import time
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np

from ._dimension_adapter import Resources, engine
from ._pm_count_aware import down, up, sum_upper
from ._numeric import local_float
from ._t_mpjp import log_nonnegative


def load_public_case(data_dir: Path):
    config = json.loads((data_dir / "mixed_config.json").read_text())
    with np.load(data_dir / "mixed_public_records.npz", allow_pickle=False) as z:
        raw = z["public_raw"].copy()
        channels = tuple(z["channel_ids"].tolist())
    return config, raw, channels


def conditioned_basis(candidate: str, record, config: dict, channels: tuple[str, ...]):
    ci = config["candidates"].index(candidate)
    marker_indices = [i for i, name in enumerate(channels) if name.startswith("T_INJECTION/G1_")]
    m = sum(int(record[i]) for i in marker_indices) & 1
    return config["basis_if_m1"][ci] if m else candidate.split("_")[-1]


def load_conditioned_model(data_dir: Path, candidate: str, basis: str):
    path = data_dir / "mixed_models" / f"{candidate}__{basis}.json.gz"
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def build_model(doc: dict, config: dict, resources: Resources):
    caps = doc["caps"]
    ns = engine(doc["dimension"], doc["ideal_dimension"], caps, len(config["stages"]))
    ns["CANDIDATES"] = (doc["candidate"],)
    words = ns["NWORDS"]
    exact, factors, offsets = [], [], [0]
    for stage in range(len(config["stages"])):
        for group in doc["groups"]:
            if group["stage"] != stage:
                continue
            rate = group.get("rate", config["low_fault_rates"].get(group["class"]))
            p = Fraction(str(rate))
            total = sum(count for _, count in group["pattern"])
            distribution = Counter({0: 1 - p})
            for signature, count in group["pattern"]:
                distribution[int(signature, 16)] += p * Fraction(count, total)
            if sum(distribution.values()) != 1:
                raise ValueError("local fault distribution is not normalized")
            key = tuple(
                (tuple((signature >> (64 * i)) & ((1 << 64) - 1) for i in range(words)),
                 prob.numerator, prob.denominator)
                for signature, prob in sorted(distribution.items()) if prob
            )
            factors.extend([len(exact)] * group["multiplicity"])
            exact.append(key)
        offsets.append(len(factors))
    data = {"stage_location_offsets": np.asarray([offsets]),
            "location_factor_id": np.asarray(factors),
            "low_tail_CPLUS": [caps["tail_CPLUS"]]}
    terms = [(i,) for i in range(doc["dimension"])]
    model = ns["build_candidate"](0, data, exact, terms, resources)
    model["mpjp_log"] = log_nonnegative(model["projected_upper"])
    dimension = doc["dimension"]
    if dimension % 8:
        last = model["projected_upper"][:, -1, :].reshape(
            (-1, 1 << (dimension % 8), 1 << (8 - dimension % 8))
        )
        model["mpjp_last"] = log_nonnegative(sum_upper(last, axis=-1))
    return model, ns["score"]


def _project(value: int, checks: list[int]) -> int:
    return sum(((value & check).bit_count() & 1) << i for i, check in enumerate(checks))


def _magic_weight(sign: int):
    with localcontext() as context:
        context.prec = 110
        value = Decimal(1) + (Decimal(-1) if sign else Decimal(1)) / Decimal(2).sqrt()
        lower = local_float(Fraction(value - Decimal("1e-100")))[1]
        upper = local_float(Fraction(value + Decimal("1e-100")))[2]
    return float(value), lower, upper


def score_record(doc: dict, model: dict, scorer, raw, resources: Resources):
    raw_integer = sum(int(bit) << i for i, bit in enumerate(raw))
    checks = [int(value, 16) for value in doc["checks"]]
    target = _project(raw_integer ^ int(doc["offset"], 16), checks)
    magic = doc["magic"]
    sectors = []
    for sector in range(2 if magic else 1):
        quotient = target | (sector << len(checks))
        projected = [(quotient >> i) & 1 for i in range(doc["dimension"])]
        result = scorer(model, projected, resources)
        if magic:
            sign = ((raw_integer & int(magic["raw_mask"], 16)).bit_count() & 1) ^ magic["constant"] ^ sector
            weight, weight_lower, weight_upper = _magic_weight(sign)
        else:
            weight = weight_lower = weight_upper = 1.0
        sectors.append((result, weight, weight_lower, weight_upper))
    point = sum(weight * result["qCplus_retained"] * result["likelihood_factor"]
                for result, weight, _, _ in sectors)
    lower = float(down(sum(float(down(wl * result["L_FULL_lower"]))
                           for result, _, wl, _ in sectors)))
    upper = float(up(sum(float(up(wu * result["L_FULL_upper"]))
                         for result, _, _, wu in sectors)))
    if not (0 <= lower <= point <= upper and math.isfinite(upper)):
        raise ArithmeticError("mixed likelihood is outside its interval")
    return {"candidate": doc["candidate"], "basis": doc["basis"],
            "point_retained": point, "full_lower": lower, "full_upper": upper,
            "sectors": len(sectors), "retained_states": max(
                result["max_retained_states"] for result, *_ in sectors)}


def score_candidate(data_dir: Path, config: dict, raw, channels, candidate: str,
                    seconds: float = 3600, memory_gib: float = 14):
    basis = conditioned_basis(candidate, raw, config, channels)
    doc = load_conditioned_model(data_dir, candidate, basis)
    resources = Resources(seconds=seconds, gib=memory_gib)
    model, scorer = build_model(doc, config, resources)
    result = score_record(doc, model, scorer, raw, resources)
    result["elapsed_seconds"] = time.perf_counter() - resources.started
    return result
