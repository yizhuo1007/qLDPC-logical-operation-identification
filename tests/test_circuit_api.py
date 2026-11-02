"""Input validation and logical-action mapping for circuit identification."""

from __future__ import annotations

import itertools
import unittest
from pathlib import Path

from qldpc_identification.circuits import load_circuit_case, load_observation
from qldpc_identification.models.automorphism import SYMBOLS, load_automorphism_code_18_4_2

ROOT = Path(__file__).resolve().parents[1]


class CircuitInputTests(unittest.TestCase):
    def test_bv_full_and_prefix_inputs_have_declared_boundary_lengths(self):
        full = load_circuit_case(ROOT / "examples/configs/bv72.json")
        prefix = load_circuit_case(ROOT / "examples/configs/bv72-prefix.json")
        self.assertEqual(len(load_observation(full).bits), 216)
        self.assertEqual(len(load_observation(prefix).bits), 144)
        self.assertEqual(load_observation(prefix).bits, load_observation(full).bits[:144])

    def test_mixed_case_uses_compiled_candidate_catalogue(self):
        case = load_circuit_case(ROOT / "examples/configs/mixed18.json")
        observation = load_observation(case)
        self.assertEqual(len(case.candidates), 8)
        self.assertEqual(len(observation.channels), len(observation.bits))

    def test_frozen_basis_actions_group_four_step_sequences(self):
        model = load_automorphism_code_18_4_2(
            ROOT / "data/code_18_4_2/code_and_ports.json")
        identity = model.logical_actions["I"]
        self.assertEqual(model.net_logical_action(("x", "x^-1")), identity)
        self.assertEqual(model.net_logical_action(("y^-1", "y")), identity)
        actions = {model.logical_action_id(word)
                   for word in itertools.product(SYMBOLS, repeat=4)}
        self.assertEqual(len(actions), 3)


if __name__ == "__main__":
    unittest.main()
