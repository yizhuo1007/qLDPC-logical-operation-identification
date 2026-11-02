"""Evaluate supported full-circuit models through one public entry point."""

from __future__ import annotations

from typing import Mapping

from qldpc_identification.inference.scalable_bv import BVScalableScorer
from qldpc_identification.inference.scalable_mixed import load_public_case, score_candidate
from qldpc_identification.inference.targets import Number, aggregate_log_scores, aggregate_targets

from .case import CircuitCase, CircuitObservation


def identify_circuit(case: CircuitCase, observation: CircuitObservation, *,
                     hypothesis: str | None = None, seconds: float = 3600,
                     memory_gib: float = 14,
                     prior: str | Mapping[str, Number] = "uniform_candidates") -> dict:
    """Score a joint circuit model; a one-hypothesis query has no MAP decision."""
    if hypothesis is not None and hypothesis not in case.candidates:
        raise ValueError("hypothesis is outside the case candidate set")
    result = {"case": case.name, "code": case.code,
              "candidate_circuits": list(case.candidates),
              "recovery_targets": dict(case.target_of)}
    if case.backend == "clifford_order2":
        if observation.channels:
            raise ValueError("Clifford observation must contain only syndrome bits")
        selected = (hypothesis,) if hypothesis is not None else case.candidates
        scorer = BVScalableScorer(case.data_dir, candidates=selected,
                                  intervals=case.intervals,
                                  bits_per_interval=case.bits_per_interval,
                                  case_id=case.name)
        detail = scorer.score(observation.bits)
        result["boundaries"] = detail["boundaries"]
        result["score_kind"] = "relative_approximate_uncalibrated"
        if hypothesis is not None:
            result["decision_status"] = "INCOMPLETE_CANDIDATE_COVERAGE"
            return result
        final = detail["boundaries"][-1]
        logs = {candidate: (value if value is not None else float("-inf"))
                for candidate, value in final["log_likelihoods"].items()}
        targets = aggregate_log_scores(logs, case.target_of, prior=prior)
        for boundary in result["boundaries"]:
            boundary_logs = {candidate: (value if value is not None else float("-inf"))
                             for candidate, value in boundary["log_likelihoods"].items()}
            candidate_scores = aggregate_log_scores(
                boundary_logs, {candidate: candidate for candidate in case.candidates},
                prior=targets.candidate_priors,
            )
            normalized = candidate_scores.normalized_target_scores
            boundary["normalized_scores"] = normalized
            boundary["map_winners"] = list(candidate_scores.map_winners)
        result["candidate_priors"] = {candidate: float(weight)
                                      for candidate, weight in targets.candidate_priors.items()}
        result["relative_target_scores"] = targets.target_scores
        result["normalized_approximate_scores"] = targets.normalized_target_scores
        result["map_winners"] = list(targets.map_winners)
        result["decision_status"] = (final["status"] if targets.map_winners
                                     else "ABSTAIN_ZERO_TARGET_SCORE")
        result["exact_map_certified"] = False
        return result

    if case.backend == "precompiled_mixed":
        config, _, channels = load_public_case(case.data_dir)
        if observation.channels != channels or len(observation.bits) != len(channels):
            raise ValueError("observation channels differ from the compiled mixed model")
        selected = (hypothesis,) if hypothesis is not None else case.candidates
        evidence = [score_candidate(case.data_dir, config, observation.bits, channels,
                                    candidate, seconds=seconds, memory_gib=memory_gib)
                    for candidate in selected]
        result["candidate_likelihoods"] = evidence
        result["score_kind"] = "retained_likelihood_with_full_bounds"
        if hypothesis is not None:
            result["decision_status"] = "INCOMPLETE_CANDIDATE_COVERAGE"
            return result
        targets = aggregate_targets(
            {item["candidate"]: item["point_retained"] for item in evidence},
            case.target_of,
            prior=prior,
            candidate_bounds={item["candidate"]: (item["full_lower"], item["full_upper"])
                              for item in evidence},
        )
        result["candidate_priors"] = {candidate: float(weight)
                                      for candidate, weight in targets.candidate_priors.items()}
        result["target_scores"] = targets.target_scores
        result["normalized_approximate_scores"] = targets.normalized_target_scores
        result["target_bounds"] = {target: {"lower": lower, "upper": upper}
                                   for target, (lower, upper) in targets.target_bounds.items()}
        result["map_winners"] = list(targets.map_winners)
        result["exact_map_certified"] = targets.certified_winner is not None
        result["certified_winner"] = targets.certified_winner
        result["decision_status"] = ("OK" if targets.map_winners
                                     else "ABSTAIN_ZERO_TARGET_SCORE")
        return result
    raise ValueError("unsupported circuit backend")
