"""Optional full-circuit numerical checks; run directly after the quick suite."""

from __future__ import annotations

import unittest
from pathlib import Path

from qldpc_identification.inference.scalable_bv import BVScalableScorer, load_public_records
from qldpc_identification.inference.scalable_mixed import load_public_case, score_candidate

DATA = Path(__file__).resolve().parents[1] / "data"


class FullCircuitReferenceTests(unittest.TestCase):
    def test_bv_first_public_record(self):
        folder = DATA / "code_72_12_6" / "bv"
        scores = BVScalableScorer(folder).score(load_public_records(folder)[0])
        self.assertEqual(scores["boundaries"][-1]["map_winners"], ["V00"])
        self.assertAlmostEqual(scores["boundaries"][-1]["normalized_scores"]["V00"],
                               0.9999603710502278, delta=1e-12)

    def test_mixed_one_and_two_magic_sectors(self):
        folder = DATA / "code_18_4_2"
        config, raw, channels = load_public_case(folder)
        expected = {
            "I_ZI_IX": (1.2086005903682783e-26, 1.2086005903630602e-26,
                        3.656536275811111e-17, 1),
            "g_ZI_IX": (1.0172491596660422e-10, 1.0172491596622423e-10,
                        1.0174738452041785e-10, 2),
        }
        for candidate, (point, lower, upper, sectors) in expected.items():
            result = score_candidate(folder, config, raw[0], channels, candidate)
            self.assertAlmostEqual(result["point_retained"], point, delta=abs(point) * 1e-10)
            self.assertAlmostEqual(result["full_lower"], lower, delta=abs(lower) * 1e-10)
            self.assertAlmostEqual(result["full_upper"], upper, delta=abs(upper) * 1e-10)
            self.assertEqual(result["sectors"], sectors)


if __name__ == "__main__":
    unittest.main()
