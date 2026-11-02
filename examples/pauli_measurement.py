"""Run exact ideal identification of 15 [[18,4,2]] logical Pauli measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qldpc_identification.inference.exact import (
    ideal_distributions,
    score_record,
    uniform_prior_accuracy,
)
from qldpc_identification.models.pauli_measurement import load_ideal_code_18_4_2
from qldpc_identification.inference.scalable_pauli import load_prepared_model, score_candidate
from qldpc_identification.inference.targets import aggregate_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("ideal-exact", "low-noisy-scalable"), default="ideal-exact")
    parser.add_argument("--candidate", default="XI", help="Candidate used only to generate the demo observation")
    parser.add_argument("--branch", type=int, default=3, help="Seven-bit ideal measurement branch")
    parser.add_argument("--record", type=int, default=0, help="Public noisy record index")
    parser.add_argument("--hypothesis", help="Score one noisy hypothesis; omit to score all 15")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    data = Path(__file__).resolve().parents[1] / "data" / "code_18_4_2"
    if args.case == "ideal-exact":
        model = load_ideal_code_18_4_2(data)
        if args.candidate not in model.candidates:
            parser.error("candidate must be one of the 15 [[18,4,2]] two-port Paulis")
        record = model.record(args.candidate, args.branch)
        laws = ideal_distributions(model, model.candidates)
        scores, winners = score_record(laws, record)
        bayes, strict = uniform_prior_accuracy(laws)
        result = {
            "case": args.case,
            "candidates": list(model.candidates),
            "observation": {"edge_z_bits": record[0], "post_bb_syndrome_bits": record[1]},
            "candidate_likelihoods": {label: str(value) for label, value in scores.items()},
            "map_winners": list(winners),
            "example_generating_candidate": args.candidate,
            "bayes_accuracy_exact": str(bayes),
            "strict_unique_accuracy_exact": str(strict),
        }
    else:
        data_model, exact, terms = load_prepared_model(data / "pm_low_model.npz")
        hypotheses = ([args.hypothesis] if args.hypothesis else list(data_model["candidate_labels"]))
        scores = [score_candidate(data_model, exact, terms, args.record, hypothesis)
                  for hypothesis in hypotheses]
        result = {"case": args.case, "code": "[[18,4,2]]", "record_index": args.record,
                  "observation_channels": 48, "candidate_likelihoods": scores}
        if len(scores) == 15:
            labels = {item["candidate"]: item["candidate"] for item in scores}
            targets = aggregate_targets(
                {item["candidate"]: item["point_retained"] for item in scores}, labels,
                candidate_bounds={item["candidate"]: (item["full_lower"], item["full_upper"])
                                  for item in scores},
            )
            result["map_winners"] = list(targets.map_winners)
            result["full_likelihood_certified"] = targets.certified_winner is not None
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
