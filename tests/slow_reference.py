"""Optional full-circuit numerical checks; run directly after the quick suite."""

from __future__ import annotations

import unittest
from pathlib import Path

from qldpc_identification.circuits import identify_circuit, load_circuit_case, load_observation

DATA = Path(__file__).resolve().parents[1] / "data"
ROOT = DATA.parent


class FullCircuitReferenceTests(unittest.TestCase):
    def test_bv_prefix_and_full_share_first_two_joint_boundaries(self):
        full = load_circuit_case(ROOT / "examples/configs/bv72.json")
        prefix = load_circuit_case(ROOT / "examples/configs/bv72-prefix.json")
        full_result = identify_circuit(full, load_observation(full))
        prefix_result = identify_circuit(prefix, load_observation(prefix))
        self.assertEqual(full_result["boundaries"][:2], prefix_result["boundaries"])
        self.assertEqual(full_result["map_winners"], ["V00"])
        self.assertAlmostEqual(full_result["boundaries"][-1]["normalized_scores"]["V00"],
                               0.9999603710502278, delta=1e-12)
        self.assertAlmostEqual(full_result["normalized_approximate_scores"]["V00"],
                               0.9999603710502278, delta=1e-12)
        self.assertFalse(full_result["exact_map_certified"])

    def test_mixed_one_and_two_magic_sectors(self):
        case = load_circuit_case(ROOT / "examples/configs/mixed18.json")
        observation = load_observation(case)
        expected = {
            "I_ZI_IX": (1.2086005903682783e-26, 1.2086005903630602e-26,
                        3.656536275811111e-17, 1),
            "g_ZI_IX": (1.0172491596660422e-10, 1.0172491596622423e-10,
                        1.0174738452041785e-10, 2),
        }
        for candidate, (point, lower, upper, sectors) in expected.items():
            partial = identify_circuit(case, observation, hypothesis=candidate)
            self.assertEqual(partial["decision_status"], "INCOMPLETE_CANDIDATE_COVERAGE")
            self.assertNotIn("map_winners", partial)
            result = partial["candidate_likelihoods"][0]
            self.assertAlmostEqual(result["point_retained"], point, delta=abs(point) * 1e-10)
            self.assertAlmostEqual(result["full_lower"], lower, delta=abs(lower) * 1e-10)
            self.assertAlmostEqual(result["full_upper"], upper, delta=abs(upper) * 1e-10)
            self.assertEqual(result["sectors"], sectors)


if __name__ == "__main__":
    unittest.main()
