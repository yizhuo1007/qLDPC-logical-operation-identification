"""Exact ideal inference from finite, uniformly weighted observation laws."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from typing import Iterable

from qldpc_identification.models.pauli_measurement import IdealMeasurementModel

Record = tuple[int, int]
Distributions = dict[str, dict[Record, Fraction]]


def ideal_distributions(model: IdealMeasurementModel,
                        candidates: Iterable[str]) -> Distributions:
    """Sum all ideal measurement branches without sampling or approximation."""
    result: Distributions = {}
    total = 1 << model.hidden_dimension
    for candidate in candidates:
        counts = Counter(model.record(candidate, hidden) for hidden in range(total))
        result[candidate] = {record: Fraction(count, total)
                             for record, count in counts.items()}
        if sum(result[candidate].values()) != 1:
            raise ArithmeticError(f"{candidate} ideal law is not normalized")
    if not result:
        raise ValueError("at least one candidate is required")
    return result


def score_record(distributions: Distributions, record: Record) -> tuple[dict[str, Fraction], tuple[str, ...]]:
    """Return all candidate likelihoods and uniform-prior MAP winners."""
    scores = {candidate: law.get(record, Fraction(0))
              for candidate, law in distributions.items()}
    best = max(scores.values())
    if best == 0:
        raise ValueError("record is outside every candidate support")
    winners = tuple(candidate for candidate, value in scores.items() if value == best)
    return scores, winners


def uniform_prior_accuracy(distributions: Distributions) -> tuple[Fraction, Fraction]:
    """Return tie-aware Bayes and strict unique-MAP accuracy exactly."""
    keys = set().union(*(set(law) for law in distributions.values()))
    prior_denominator = len(distributions)
    bayes = Fraction(0)
    strict = Fraction(0)
    for record in keys:
        _, winners = score_record(distributions, record)
        highest = max(law.get(record, Fraction(0)) for law in distributions.values())
        contribution = highest / prior_denominator
        bayes += contribution
        if len(winners) == 1:
            strict += contribution
    return bayes, strict
