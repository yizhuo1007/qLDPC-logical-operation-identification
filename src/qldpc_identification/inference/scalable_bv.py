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

    def __init__(self, data_dir: Path, *, candidates: tuple[str, ...] = CANDIDATES,
                 intervals: tuple[str, ...] = INTERVALS,
                 bits_per_interval: int = 72,
                 case_id: str = "bv-low-marker-scalable"):
        self.candidates = tuple(candidates)
        self.intervals = tuple(intervals)
        self.bits_per_interval = int(bits_per_interval)
        self.case_id = case_id
        if (not self.candidates or len(set(self.candidates)) != len(self.candidates)
                or not self.intervals or len(set(self.intervals)) != len(self.intervals)
                or self.bits_per_interval < 1):
            raise ValueError("invalid candidates or circuit intervals")
        self.codec = SyndromeCodec.from_json(data_dir / "syndrome_codec.json")
        with np.load(data_dir / "clean_marker_histories.npz", allow_pickle=False) as archive:
            available = tuple(str(value) for value in archive["candidates"])
            if any(candidate not in available for candidate in self.candidates):
                raise ValueError("candidate has no clean reference history")
            self.clean = archive["histories"][[available.index(c) for c in self.candidates]].copy()
            if self.clean.shape[1] < len(self.intervals) * self.bits_per_interval:
                raise ValueError("clean history is shorter than the declared circuit")
        self.factors = {}
        cache = {}
        for candidate in self.candidates:
            intervals = parse_stim_intervals(data_dir / f"{candidate}.stim")
            if tuple(intervals)[:len(self.intervals)] != self.intervals:
                raise ValueError("declared intervals must match the compiled circuit prefix")
            for boundary, name in enumerate(self.intervals):
                steps = tuple(intervals[name])
                measured = sum(step.measurement_pos is not None for step in steps)
                if measured != self.bits_per_interval:
                    raise ValueError("compiled interval and observation layout disagree")
                if steps not in cache:
                    cache[steps] = WideOrder2IntervalFactor(
                        steps, self.codec, profile="LOW", local_cap=256, state_cap=1024
                    )
                self.factors[candidate, boundary] = cache[steps]

    def score(self, raw_history) -> dict:
        raw = tuple(int(bit) for bit in raw_history)
        expected = self.bits_per_interval * len(self.intervals)
        if len(raw) != expected or any(bit not in (0, 1) for bit in raw):
            raise ValueError(f"raw history must contain {expected} binary outcomes")
        cumulative = {candidate: 0.0 for candidate in self.candidates}
        frontiers = {candidate: {0: 1.0} for candidate in self.candidates}
        boundaries = []
        for boundary in range(len(self.intervals)):
            observed = sum(raw[self.bits_per_interval * boundary + i] << i
                           for i in range(self.bits_per_interval))
            scores = {}
            for ci, candidate in enumerate(self.candidates):
                baseline = sum(int(self.clean[ci, self.bits_per_interval * boundary + i]) << i
                               for i in range(self.bits_per_interval))
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
                scores[candidate] = cumulative[candidate] + math.log(1 / len(self.candidates))
            normalized, status = normalize_scores(scores, self.candidates)
            finite = [value for value in normalized.values() if value is not None]
            peak = max(finite) if finite else None
            winners = [] if peak is None else [
                candidate for candidate, value in normalized.items()
                if value is not None and peak - value <= 1e-12
            ]
            boundaries.append({"interval": self.intervals[boundary],
                               "log_likelihoods": {
                                   candidate: (value if math.isfinite(value) else None)
                                   for candidate, value in cumulative.items()},
                               "normalized_scores": normalized,
                               "map_winners": winners, "status": status})
        return {"case": self.case_id, "code": "[[72,12,6]]",
                "candidate_circuits": list(self.candidates), "boundaries": boundaries}


def load_public_records(data_dir: Path):
    with np.load(data_dir / "public_marker_records.npz", allow_pickle=False) as archive:
        return archive["public_raw"].copy()
