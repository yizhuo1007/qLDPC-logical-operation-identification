"""Independent checks of the selected exact ideal [[18,4,2]] calculations."""

from __future__ import annotations

import unittest
from fractions import Fraction
from pathlib import Path

from qldpc_identification.inference.exact import (
    ideal_distributions,
    score_record,
    uniform_prior_accuracy,
)
from qldpc_identification.models.automorphism import ideal_history_catalog, load_automorphism_code_18_4_2
from qldpc_identification.inference.noisy_automorphism import exact_syndrome_noise, exact_word_likelihoods
from qldpc_identification.models.pauli_measurement import load_ideal_code_18_4_2
from qldpc_identification.models.t_injection import ideal_axis_distributions
from qldpc_identification.inference.scalable_pauli import load_prepared_model, score_candidate
from qldpc_identification.inference.scalable_t_injection import load_prepared_model as load_t_model, score_candidate as score_t_candidate

DATA = Path(__file__).resolve().parents[1] / "data" / "code_18_4_2"


class IdealCode18_4_2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_ideal_code_18_4_2(DATA)

    def test_15_way_pauli_measurement(self) -> None:
        laws = ideal_distributions(self.model, self.model.candidates)
        self.assertEqual(len(laws), 15)
        self.assertTrue(all(sum(law.values()) == 1 for law in laws.values()))
        self.assertEqual(uniform_prior_accuracy(laws),
                         (Fraction(67, 120), Fraction(497, 1920)))

    def test_automorphism_marker_resolves_all_625_words(self) -> None:
        model = load_automorphism_code_18_4_2(DATA / "code_and_ports.json")
        catalog = ideal_history_catalog(model)
        self.assertEqual(len(catalog), 625)
        self.assertTrue(all(len(words) == 1 for words in catalog.values()))

    def test_noisy_exact_auto_sums_all_fault_patterns(self) -> None:
        model = load_automorphism_code_18_4_2(DATA / "code_and_ports.json")
        noise = exact_syndrome_noise(model, 0.001, 0.001)
        self.assertEqual(len(noise), 1 << 14)
        self.assertAlmostEqual(sum(noise.values()), 1)
        word = ("x", "y", "I", "x^-1")
        scores = exact_word_likelihoods(model, model.history(word), 0.0, 0.0)
        self.assertEqual(scores[word], 1)
        self.assertEqual(sum(value > 0 for value in scores.values()), 1)

    def test_six_way_t_injection(self) -> None:
        laws = ideal_axis_distributions(self.model)
        self.assertEqual(len(laws), 6)
        self.assertEqual(uniform_prior_accuracy(laws),
                         (Fraction(593, 768), Fraction(107, 192)))

    def test_low_noisy_pauli_likelihood_matches_frozen_query(self) -> None:
        data, factors, terms = load_prepared_model(DATA / "pm_low_model.npz")
        result = score_candidate(data, factors, terms, 0, "XI")
        self.assertAlmostEqual(result["point_retained"], 6.1464855405451465e-15, delta=1e-25)
        self.assertAlmostEqual(result["full_lower"], 6.146485540538372e-15, delta=1e-25)
        self.assertAlmostEqual(result["full_upper"], 1.4113650664517071e-10, delta=1e-20)

    def test_low_noisy_t_likelihood_matches_frozen_query(self) -> None:
        data, factors, terms = load_t_model(DATA / "t_low_model.npz")
        result = score_t_candidate(data, factors, terms, 0, "XI")
        self.assertAlmostEqual(result["point_retained"], 9.40875705504442e-10, delta=1e-20)
        self.assertAlmostEqual(result["full_lower"], 9.408757055034512e-10, delta=1e-20)
        self.assertAlmostEqual(result["full_upper"], 1.7947287390605636e-09, delta=1e-19)

    def test_likelihood_uses_observation_only(self) -> None:
        laws = ideal_distributions(self.model, self.model.candidates)
        record = self.model.record("XI", 3)
        scores, winners = score_record(laws, record)
        self.assertEqual(scores["XI"], Fraction(1, 128))
        self.assertIn("XI", winners)
        self.assertEqual(sum(laws["XI"].values()), 1)


if __name__ == "__main__":
    unittest.main()
