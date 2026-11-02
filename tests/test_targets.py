"""Recovery-target decisions for exact and bounded candidate evidence."""

from __future__ import annotations

import math
import unittest
from fractions import Fraction

from qldpc_identification.inference.targets import aggregate_log_scores, aggregate_targets


class RecoveryTargetTests(unittest.TestCase):
    def test_many_to_one_target_changes_map_decision(self):
        likelihoods = {"h1": Fraction(30, 100), "h2": Fraction(28, 100),
                       "h3": Fraction(45, 100)}
        mapping = {"h1": "A", "h2": "A", "h3": "B"}
        result = aggregate_targets(likelihoods, mapping)
        self.assertEqual(max(likelihoods, key=likelihoods.get), "h3")
        self.assertEqual(result.map_winners, ("A",))
        self.assertEqual(result.target_scores["A"], Fraction(58, 300))

    def test_prior_rule_changes_target_decision(self):
        likelihoods = {"h1": Fraction(1, 4), "h2": Fraction(1, 4),
                       "h3": Fraction(2, 5)}
        mapping = {"h1": "A", "h2": "A", "h3": "B"}
        self.assertEqual(aggregate_targets(likelihoods, mapping).map_winners, ("A",))
        result = aggregate_targets(likelihoods, mapping, prior="uniform_targets")
        self.assertEqual(result.map_winners, ("B",))
        self.assertEqual(result.candidate_priors["h1"], Fraction(1, 4))
        custom = aggregate_targets(likelihoods, mapping,
                                   prior={"h1": Fraction(1, 10),
                                          "h2": Fraction(1, 10),
                                          "h3": Fraction(4, 5)})
        self.assertEqual(custom.map_winners, ("B",))

    def test_aggregated_bounds_certify_only_with_strict_gap(self):
        mapping = {"h1": "A", "h2": "A", "h3": "B"}
        scores = {"h1": 0.30, "h2": 0.28, "h3": 0.45}
        bounds = {"h1": (0.29, 0.31), "h2": (0.27, 0.29),
                  "h3": (0.44, 0.46)}
        result = aggregate_targets(scores, mapping, candidate_bounds=bounds)
        self.assertEqual(result.certified_winner, "A")
        self.assertGreater(result.target_bounds["A"][0], result.target_bounds["B"][1])
        bounds["h3"] = (0.44, 0.60)
        result = aggregate_targets(scores, mapping, candidate_bounds=bounds)
        self.assertIsNone(result.certified_winner)

    def test_missing_bounds_and_candidates_are_rejected(self):
        scores = {"a": 0.3, "b": 0.4}
        mapping = {"a": "A", "b": "B"}
        with self.assertRaisesRegex(ValueError, "complete candidate set"):
            aggregate_targets(scores, mapping, candidate_bounds={"a": (0.2, 0.4)})
        with self.assertRaisesRegex(ValueError, "same nonempty candidate set"):
            aggregate_targets({"a": 0.3}, mapping)

    def test_float_bound_aggregation_rounds_outward(self):
        mapping = {"a": "A", "b": "A", "c": "B"}
        points = {"a": 0.2, "b": 0.3, "c": 0.4}
        bounds = {"a": (0.1, 0.25), "b": (0.2, 0.35), "c": (0.3, 0.45)}
        result = aggregate_targets(points, mapping, candidate_bounds=bounds)
        exact_lower = (Fraction.from_float(0.1) + Fraction.from_float(0.2)) / 3
        exact_upper = (Fraction.from_float(0.25) + Fraction.from_float(0.35)) / 3
        lower, upper = result.target_bounds["A"]
        self.assertLessEqual(Fraction.from_float(lower), exact_lower)
        self.assertGreaterEqual(Fraction.from_float(upper), exact_upper)

    def test_log_scores_use_common_scale_without_a_certificate(self):
        logs = {"h1": -1000.0, "h2": -1000.0, "h3": -999.5}
        mapping = {"h1": "A", "h2": "A", "h3": "B"}
        result = aggregate_log_scores(logs, mapping)
        self.assertEqual(result.map_winners, ("A",))
        self.assertIsNone(result.target_bounds)
        self.assertIsNone(result.certified_winner)
        self.assertTrue(math.isclose(sum(result.normalized_target_scores.values()), 1.0))


if __name__ == "__main__":
    unittest.main()
