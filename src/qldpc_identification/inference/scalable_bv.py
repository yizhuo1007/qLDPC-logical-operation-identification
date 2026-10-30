"""Four-candidate Bernstein–Vazirani circuit inference on [[72,12,6]]."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ._bv_order2 import SyndromeCodec, WideOrder2IntervalFactor, normalize_scores, parse_stim_intervals

CANDIDATES = ("V00", "V01", "V10", "V11")
INTERVALS = ("PREPARATION+SEC0", "OP1+SEC1", "OP2+SEC2")


class BVScalableScorer:
    """Retain the 256 most likely local paths and 1024 boundary states."""

    def __init__(self, data_dir: Path):
        self.codec = SyndromeCodec.from_json(data_dir / "syndrome_codec.json")
        with np.load(data_dir / "clean_marker_histories.npz", allow_pickle=False) as archive:
            if tuple(archive["candidates"]) != CANDIDATES:
                raise ValueError("unexpected BV candidate order")
            self.clean = archive["histories"].copy()
        self.factors = {}
        cache = {}
        for candidate in CANDIDATES:
            intervals = parse_stim_intervals(data_dir / f"{candidate}.stim")
            for boundary, name in enumerate(INTERVALS):
                steps = tuple(intervals[name])
                if steps not in cache:
                    cache[steps] = WideOrder2IntervalFactor(
                        steps, self.codec, profile="LOW", local_cap=256, state_cap=1024
                    )
                self.factors[candidate, boundary] = cache[steps]

    def score(self, raw_history) -> dict:
        raw = tuple(int(bit) for bit in raw_history)
        if len(raw) != 216 or any(bit not in (0, 1) for bit in raw):
            raise ValueError("BV raw history must contain 216 binary outcomes")
        cumulative = {candidate: 0.0 for candidate in CANDIDATES}
        frontiers = {candidate: {0: 1.0} for candidate in CANDIDATES}
        boundaries = []
        for boundary in range(3):
            observed = sum(raw[72 * boundary + i] << i for i in range(72))
            scores = {}
            for ci, candidate in enumerate(CANDIDATES):
                baseline = sum(int(self.clean[ci, 72 * boundary + i]) << i for i in range(72))
                nxt, _ = self.factors[candidate, boundary].update(
                    frontiers[candidate], observed, baseline
                )
                if not nxt:
                    cumulative[candidate] = float("-inf")
                    scores[candidate] = float("-inf")
                    continue
                scale = sum(nxt.values())
                cumulative[candidate] += math.log(scale)
                frontiers[candidate] = {state: mass / scale for state, mass in nxt.items()}
                scores[candidate] = cumulative[candidate] + math.log(0.25)
            normalized, status = normalize_scores(scores, CANDIDATES)
            finite = [value for value in normalized.values() if value is not None]
            peak = max(finite) if finite else None
            winners = [] if peak is None else [
                candidate for candidate, value in normalized.items()
                if value is not None and peak - value <= 1e-12
            ]
            boundaries.append({"normalized_scores": normalized, "map_winners": winners,
                               "status": status})
        return {"case": "bv-low-marker-scalable", "code": "[[72,12,6]]",
                "candidate_circuits": list(CANDIDATES), "boundaries": boundaries}


def load_public_records(data_dir: Path):
    with np.load(data_dir / "public_marker_records.npz", allow_pickle=False) as archive:
        return archive["public_raw"].copy()
