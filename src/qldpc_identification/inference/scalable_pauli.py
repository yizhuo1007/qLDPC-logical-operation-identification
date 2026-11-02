"""Prepared [[18,4,2]] LOW-noise Pauli measurement scorer."""

from __future__ import annotations

import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import _pm_count_aware as core


def load_prepared_model(path: Path):
    """Load prepared fault-response factors and observation records."""
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    if tuple(data["candidate_labels"]) != core.CANDIDATES:
        raise ValueError("candidate order differs from the selected setting")
    if data["public_boundary"].shape[1] != 48:
        raise ValueError("unexpected boundary record width")
    offsets = data["factor_sparse_offsets"]
    exact = tuple(
        tuple((int(data["sparse_signature"][i]),
               int(data["low_probability_numerator"][i]),
               int(data["low_probability_denominator"][i]))
              for i in range(int(a), int(b)))
        for a, b in zip(offsets[:-1], offsets[1:])
    )
    retained = np.full((15, 2, 5), "1", dtype="U120")
    retained[:, 1, 4] = data["low_retained_mass_CPLUS"]
    data["candidate_quantile_retained_mass"] = retained
    terms = json.loads(str(data["term_indices_json"]))
    return data, exact, terms


def score_candidate(data, exact, terms, record_index: int, candidate: str,
                    seconds: float = 3600, memory_gib: float = 12) -> dict:
    """Evaluate retained likelihood and rigorous full-likelihood interval."""
    core.CANDIDATES = tuple("XI YI ZI IX IY IZ XX XY XZ YX YY YZ ZX ZY ZZ".split())
    core.C = (1,) * 10
    core.CPLUS = (2,) * 10
    if candidate not in core.CANDIDATES:
        raise ValueError("unknown Pauli measurement candidate")
    public = data["public_boundary"]
    if not 0 <= record_index < len(public):
        raise IndexError(record_index)
    resources = core.Resources(time.perf_counter())
    core.num.TIME_LIMIT = seconds
    core.num.MEMORY_LIMIT = int(memory_gib * 2**30)
    ci = core.CANDIDATES.index(candidate)
    model = core.build_candidate(ci, data, exact, terms[ci], resources)
    result = core.score(model, public[record_index], resources)
    point = result["qCplus_retained"] * result["likelihood_factor"]
    lower, upper = result["L_FULL_lower"], result["L_FULL_upper"]
    if not 0 <= lower <= point <= upper:
        raise ArithmeticError("candidate likelihood is outside its full interval")
    return {
        "candidate": candidate,
        "point_retained": point,
        "full_lower": lower,
        "full_upper": upper,
        "precompute_seconds": model["precompute_seconds"],
        "query_seconds": result["query_seconds"],
        "retained_states": result["max_retained_states"],
        "count_caps": list(core.C),
        "state_budget": core.B,
    }
