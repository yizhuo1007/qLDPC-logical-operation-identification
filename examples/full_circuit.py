"""Identify a supported compiled logical circuit from one public observation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qldpc_identification.circuits import identify_circuit, load_circuit_case, load_observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="bv-low-marker-scalable",
                        help="Built-in case name or path to a supported circuit-case JSON")
    parser.add_argument("--record", type=int, default=0)
    parser.add_argument("--hypothesis", help="Score one candidate without a MAP decision")
    parser.add_argument("--prior", choices=("uniform_candidates", "uniform_targets"),
                        default="uniform_candidates")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_dir = Path(__file__).resolve().parent / "configs"
    built_in = {
        "bv-low-marker-scalable": config_dir / "bv72.json",
        "bv-prefix-low-marker-scalable": config_dir / "bv72-prefix.json",
        "mixed-low-marker-scalable": config_dir / "mixed18.json",
    }
    case = load_circuit_case(built_in.get(args.case, Path(args.case)))
    observation = load_observation(case, args.record)
    result = identify_circuit(case, observation, hypothesis=args.hypothesis, prior=args.prior)
    result["record_index"] = args.record
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
