"""Score a public full-circuit observation with scalable inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qldpc_identification.inference.scalable_bv import BVScalableScorer, load_public_records
from qldpc_identification.inference.scalable_mixed import load_public_case, score_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("bv-low-marker-scalable", "mixed-low-marker-scalable"),
                        default="bv-low-marker-scalable")
    parser.add_argument("--record", type=int, default=0)
    parser.add_argument("--hypothesis", help="For mixed: score one hypothesis; omit for all eight")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "data"
    if args.case == "bv-low-marker-scalable":
        data_dir = root / "code_72_12_6" / "bv"
        records = load_public_records(data_dir)
        if not 0 <= args.record < len(records):
            parser.error("record index outside public BV population")
        scorer = BVScalableScorer(data_dir)
        result = scorer.score(records[args.record])
        result["record_index"] = args.record
    else:
        data_dir = root / "code_18_4_2"
        config, raw, channels = load_public_case(data_dir)
        if not 0 <= args.record < len(raw):
            parser.error("record index outside public mixed population")
        hypotheses = [args.hypothesis] if args.hypothesis else config["candidates"]
        if any(candidate not in config["candidates"] for candidate in hypotheses):
            parser.error("unknown mixed-circuit hypothesis")
        scores = [score_candidate(data_dir, config, raw[args.record], channels, candidate)
                  for candidate in hypotheses]
        result = {"case": args.case, "code": "[[18,4,2]]", "record_index": args.record,
                  "candidate_likelihoods": scores}
        if len(scores) == 8:
            peak = max(item["point_retained"] for item in scores)
            winners = [item["candidate"] for item in scores if item["point_retained"] == peak]
            result["map_winners"] = winners
            if len(winners) == 1:
                result["full_likelihood_certified"] = next(
                    item["full_lower"] for item in scores if item["candidate"] == winners[0]
                ) > max(item["full_upper"] for item in scores if item["candidate"] != winners[0])
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
