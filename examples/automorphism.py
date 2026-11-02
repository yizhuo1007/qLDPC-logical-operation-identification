"""Identify a four-step [[18,4,2]] automorphism word in the ideal Y0-marker setting."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from qldpc_identification.inference.noisy_automorphism import exact_word_likelihoods, simulate_history
from qldpc_identification.inference.targets import aggregate_targets
from qldpc_identification.models.automorphism import (
    SYMBOLS,
    ideal_history_catalog,
    load_automorphism_code_18_4_2,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("q4-marker-ideal", "q4-marker-noisy-exact"), default="q4-marker-ideal")
    parser.add_argument("--target", choices=("physical-sequence", "net-logical-action"),
                        default="physical-sequence")
    parser.add_argument("--prior", choices=("uniform_candidates", "uniform_targets"),
                        default="uniform_candidates")
    parser.add_argument("--word", default="x,y,I,x^-1", help="Four comma-separated symbols")
    parser.add_argument("--p-x", type=float, default=0.001, help="Per-qubit X fault probability")
    parser.add_argument("--p-z", type=float, default=0.001, help="Per-qubit Z fault probability")
    parser.add_argument("--seed", type=int, default=2026, help="Observation-generation seed")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    word = tuple(args.word.split(","))
    if len(word) != 4 or any(symbol not in SYMBOLS for symbol in word):
        parser.error("word must contain four symbols from I,x,x^-1,y,y^-1")

    code_path = Path(__file__).resolve().parents[1] / "data" / "code_18_4_2" / "code_and_ports.json"
    model = load_automorphism_code_18_4_2(code_path)
    if args.case == "q4-marker-ideal":
        history = model.history(word)
        catalogue = ideal_history_catalog(model)
        winners = catalogue[history]
        result = {
            "case": args.case,
            "candidate_count": len(SYMBOLS) ** 4,
            "distinct_histories": len(catalogue),
            "observation": {"bb_syndromes_at_sec_boundaries": list(history)},
            "nonzero_candidate_likelihoods": {",".join(w): "1" for w in winners},
            "map_winners": [list(w) for w in winners],
            "example_generating_word": list(word),
            "ideal_unique_identification_accuracy_exact": "1" if len(catalogue) == len(SYMBOLS) ** 4 else "not_unique",
        }
        if args.target == "net-logical-action":
            candidate_scores = {",".join(candidate): float(candidate in winners)
                                for candidate in itertools.product(SYMBOLS, repeat=4)}
    else:
        history = simulate_history(model, word, args.p_x, args.p_z, args.seed)
        scores = exact_word_likelihoods(model, history, args.p_x, args.p_z)
        best = max(scores.values())
        winners = [candidate for candidate, score in scores.items() if abs(score - best) <= 1e-15 * best]
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:10]
        result = {
            "case": args.case,
            "candidate_count": len(scores),
            "noise_model": {"independent_x": args.p_x, "independent_z": args.p_z},
            "seed": args.seed,
            "observation": {"bb_syndromes_at_sec_boundaries": list(history)},
            "top_candidate_likelihoods": {",".join(w): score for w, score in ranked},
            "map_winners": [list(w) for w in winners],
            "example_generating_word": list(word),
            "likelihood_kind": "exact fault-pattern marginalization, perfect SEC boundaries",
        }
        if args.target == "net-logical-action":
            candidate_scores = {",".join(candidate): score for candidate, score in scores.items()}
    if args.target == "net-logical-action":
        target_of = {candidate: model.logical_action_id(tuple(candidate.split(",")))
                     for candidate in candidate_scores}
        targets = aggregate_targets(candidate_scores, target_of, prior=args.prior)
        result["recovery_target"] = "net_logical_action"
        result["distinct_logical_actions"] = len(targets.target_scores)
        result["target_scores"] = targets.target_scores
        result["map_winners"] = list(targets.map_winners)
        result["candidate_map_winners"] = [list(w) for w in winners]
        result["prior"] = args.prior
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
