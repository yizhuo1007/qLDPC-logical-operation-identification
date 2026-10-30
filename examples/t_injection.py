"""Run exact ideal identification of six [[18,4,2]] T-injection axes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qldpc_identification.inference.exact import score_record, uniform_prior_accuracy
from qldpc_identification.models.pauli_measurement import load_ideal_code_18_4_2
from qldpc_identification.models.t_injection import AXES, ideal_axis_distributions
from qldpc_identification.inference.scalable_t_injection import load_prepared_model, score_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("ideal-exact", "low-noisy-scalable"), default="ideal-exact")
    parser.add_argument("--candidate", choices=AXES, default="XI", help="Used only to generate the demo observation")
    parser.add_argument("--branch", type=int, default=3, help="Seven-bit ideal measurement branch")
    parser.add_argument("--record", type=int, default=0, help="Public noisy record index")
    parser.add_argument("--hypothesis", choices=AXES, help="Score one noisy axis; omit to score all six")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    data = Path(__file__).resolve().parents[1] / "data" / "code_18_4_2"
    if args.case == "ideal-exact":
        model = load_ideal_code_18_4_2(data)
        record = model.record(args.candidate, args.branch)
        laws = ideal_axis_distributions(model)
        scores, winners = score_record(laws, record)
        bayes, strict = uniform_prior_accuracy(laws)
        result = {
            "case": args.case,
            "candidate_axes": list(AXES),
            "observation": {"edge_z_bits": record[0], "post_bb_syndrome_bits": record[1]},
            "candidate_likelihoods": {label: str(value) for label, value in scores.items()},
            "map_winners": list(winners),
            "example_generating_axis": args.candidate,
            "bayes_accuracy_exact": str(bayes),
            "strict_unique_accuracy_exact": str(strict),
        }
    else:
        prepared, factors, terms = load_prepared_model(data / "t_low_model.npz")
        axes = [args.hypothesis] if args.hypothesis else AXES
        scores = [score_candidate(prepared, factors, terms, args.record, axis) for axis in axes]
        result = {"case": args.case, "code": "[[18,4,2]]", "record_index": args.record,
                  "observation_channels": 112, "candidate_likelihoods": scores}
        if len(scores) == len(AXES):
            best = max(item["point_retained"] for item in scores)
            winners = [item["candidate"] for item in scores if item["point_retained"] == best]
            result["map_winners"] = winners
            if len(winners) == 1:
                winner = next(item for item in scores if item["candidate"] == winners[0])
                result["full_likelihood_certified"] = winner["full_lower"] > max(
                    item["full_upper"] for item in scores if item["candidate"] != winners[0]
                )
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
