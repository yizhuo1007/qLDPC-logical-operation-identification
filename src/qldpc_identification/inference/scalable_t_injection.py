"""Prepared [[18,4,2]] LOW-noise T-injection axis scorer."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from fractions import Fraction

from ._dimension_adapter import Resources, engine
from ._t_mpjp import log_nonnegative
from ._pm_count_aware import sum_upper

AXES = ("XI", "YI", "ZI", "IX", "IY", "IZ")


def load_prepared_model(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    if tuple(data["candidate_labels"]) != AXES:
        raise ValueError("T-injection axis order differs")
    if data["public_joint_raw"].shape[1] != 112:
        raise ValueError("unexpected T-injection record width")
    offsets = data["factor_sparse_offsets"]
    exact = tuple(
        tuple(((int(data["sparse_signature"][i]),),
               int(data["probability_numerator"][i]),
               int(data["probability_denominator"][i]))
              for i in range(int(a), int(b)))
        for a, b in zip(offsets[:-1], offsets[1:])
    )
    data["low_tail_CPLUS"] = np.array(
        [str(1 - Fraction(str(value))) for value in data["retained_mass_CPLUS"]],
        dtype="U256",
    )
    terms = json.loads(str(data["term_indices_json"]))
    return data, exact, terms


def score_candidate(data, exact, terms, record_index: int, candidate: str) -> dict:
    """Evaluate a candidate's retained mass and full-likelihood interval."""
    if candidate not in AXES:
        raise ValueError("unknown T-injection axis")
    public = data["public_joint_raw"]
    if not 0 <= record_index < len(public):
        raise IndexError(record_index)
    ci = AXES.index(candidate)
    caps = {"C": (1,) * 10 + (0,), "CPLUS": (2,) * 10 + (1,)}
    namespace = engine(41, 7, caps, 11)
    namespace["CANDIDATES"] = AXES
    resources = Resources(seconds=3600, gib=12)
    model = namespace["build_candidate"](ci, data, exact, terms[ci], resources)
    model["mpjp_log"] = log_nonnegative(model["projected_upper"])
    last = model["projected_upper"][:, -1, :].reshape((-1, 1 << 1, 1 << 7))
    model["mpjp_last"] = log_nonnegative(sum_upper(last, axis=-1))
    result = namespace["score"](model, public[record_index], resources)
    point = result["qCplus_retained"] * result["likelihood_factor"]
    lower, upper = result["L_FULL_lower"], result["L_FULL_upper"]
    if not 0 <= lower <= point <= upper:
        raise ArithmeticError("candidate likelihood is outside its full interval")
    return {"candidate": candidate, "point_retained": point,
            "full_lower": lower, "full_upper": upper,
            "precompute_seconds": model["precompute_seconds"],
            "query_seconds": result["query_seconds"],
            "retained_states": result["max_retained_states"],
            "count_caps": list(caps["C"]), "state_budget": namespace["B"]}
