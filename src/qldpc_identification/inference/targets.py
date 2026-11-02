"""Aggregate candidate evidence by the requested recovery target."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

Number = float | Fraction


@dataclass(frozen=True)
class TargetResult:
    candidate_priors: dict[str, Number]
    target_scores: dict[str, Number]
    normalized_target_scores: dict[str, Number] | None
    map_winners: tuple[str, ...]
    target_bounds: dict[str, tuple[Number, Number]] | None
    certified_winner: str | None


def _validate_number(value: Number, label: str) -> None:
    if not isinstance(value, (int, float, Fraction)) or value < 0:
        raise ValueError(f"{label} must be nonnegative")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} must be finite")


def _candidate_priors(target_of: Mapping[str, str],
                      prior: str | Mapping[str, Number]) -> dict[str, Number]:
    candidates = tuple(target_of)
    if isinstance(prior, str):
        if prior == "uniform_candidates":
            weights = {candidate: Fraction(1, len(candidates)) for candidate in candidates}
        elif prior == "uniform_targets":
            counts = {target: sum(value == target for value in target_of.values())
                      for target in set(target_of.values())}
            weights = {candidate: Fraction(1, len(counts) * counts[target_of[candidate]])
                       for candidate in candidates}
        else:
            raise ValueError("unknown prior rule")
    else:
        if set(prior) != set(candidates):
            raise ValueError("custom prior must cover exactly the candidate set")
        weights = dict(prior)
    for candidate, weight in weights.items():
        _validate_number(weight, f"prior for {candidate}")
    total = sum(weights.values())
    if not math.isclose(float(total), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("candidate priors must sum to one")
    return weights


def _outward_float(value: Fraction, *, upper: bool) -> float:
    rounded = float(value)
    represented = Fraction.from_float(rounded)
    if upper and represented < value:
        return math.nextafter(rounded, math.inf)
    if not upper and represented > value:
        return math.nextafter(rounded, -math.inf)
    return rounded


def aggregate_targets(
    candidate_scores: Mapping[str, Number],
    target_of: Mapping[str, str],
    *,
    prior: str | Mapping[str, Number] = "uniform_candidates",
    candidate_bounds: Mapping[str, tuple[Number, Number]] | None = None,
) -> TargetResult:
    """Sum prior-weighted scores; certify only from complete valid bounds."""
    if not candidate_scores or set(candidate_scores) != set(target_of):
        raise ValueError("scores and target mapping must cover the same nonempty candidate set")
    if any(not isinstance(target, str) or not target for target in target_of.values()):
        raise ValueError("recovery targets must be nonempty strings")
    for candidate, value in candidate_scores.items():
        _validate_number(value, f"score for {candidate}")
    weights = _candidate_priors(target_of, prior)
    scores: dict[str, Number] = {}
    for candidate, value in candidate_scores.items():
        target = target_of[candidate]
        scores[target] = scores.get(target, 0) + weights[candidate] * value
    total = sum(scores.values())
    normalized = ({target: value / total for target, value in scores.items()}
                  if total > 0 else None)
    peak = max(scores.values())
    winners = (tuple(target for target, value in scores.items() if value == peak)
               if peak > 0 else ())

    bounds = None
    certified = None
    if candidate_bounds is not None:
        if set(candidate_bounds) != set(candidate_scores):
            raise ValueError("bounds must cover the complete candidate set")
        exact_bounds = {target: [Fraction(0), Fraction(0)] for target in scores}
        float_bounds = any(isinstance(value, float) for pair in candidate_bounds.values()
                           for value in pair) or any(isinstance(value, float)
                                                     for value in weights.values())
        for candidate, (lower, upper) in candidate_bounds.items():
            _validate_number(lower, f"lower bound for {candidate}")
            _validate_number(upper, f"upper bound for {candidate}")
            if lower > candidate_scores[candidate] or candidate_scores[candidate] > upper:
                raise ValueError(f"score for {candidate} lies outside its bounds")
            target = target_of[candidate]
            exact_bounds[target][0] += Fraction(weights[candidate]) * Fraction(lower)
            exact_bounds[target][1] += Fraction(weights[candidate]) * Fraction(upper)
        bounds = {
            target: (_outward_float(pair[0], upper=False),
                     _outward_float(pair[1], upper=True))
            if float_bounds else (pair[0], pair[1])
            for target, pair in exact_bounds.items()
        }
        if len(winners) == 1:
            winner = winners[0]
            if all(bounds[winner][0] > upper for target, (_, upper) in bounds.items()
                   if target != winner):
                certified = winner
    return TargetResult(weights, scores, normalized, winners, bounds, certified)


def aggregate_log_scores(
    candidate_log_scores: Mapping[str, float],
    target_of: Mapping[str, str],
    *,
    prior: str | Mapping[str, Number] = "uniform_candidates",
) -> TargetResult:
    """Aggregate log scores after a common scale shift to avoid underflow."""
    if not candidate_log_scores:
        raise ValueError("at least one candidate is required")
    if any(math.isnan(value) or value == math.inf for value in candidate_log_scores.values()):
        raise ValueError("log scores must be finite or negative infinity")
    finite = [value for value in candidate_log_scores.values() if math.isfinite(value)]
    peak = max(finite) if finite else 0.0
    scaled = {candidate: math.exp(value - peak) if math.isfinite(value) else 0.0
              for candidate, value in candidate_log_scores.items()}
    return aggregate_targets(scaled, target_of, prior=prior)
