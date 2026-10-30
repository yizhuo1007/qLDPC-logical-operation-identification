"""Evaluate a complete candidate-score CSV against separately stored labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--candidates", type=int, required=True)
    args = parser.parse_args()
    with args.truth.open(newline="") as stream:
        labels = {int(row["record_index"]): row["true_candidate"]
                  for row in csv.DictReader(stream)}
    grouped = defaultdict(list)
    with args.scores.open(newline="") as stream:
        for row in csv.DictReader(stream):
            grouped[int(row["record_index"])].append(row)
    unique_correct = certified_correct = certified_wrong = uncertified = 0
    complete = 0
    for index, rows in grouped.items():
        if len(rows) != args.candidates or len({row["candidate"] for row in rows}) != args.candidates:
            raise ValueError(f"record {index} does not have all candidate scores")
        if index not in labels:
            raise ValueError(f"missing evaluation label for record {index}")
        complete += 1
        points = [float(row["point"]) for row in rows]
        tolerance = 1e-12 if rows[0]["score_kind"] == "normalized_approximate_score" else 0.0
        peak = max(points)
        winners = [row["candidate"] for row, value in zip(rows, points)
                   if peak - value <= tolerance]
        if len(winners) == 1 and winners[0] == labels[index]:
            unique_correct += 1
        if rows[0]["full_lower"]:
            certified = [row["candidate"] for row in rows
                         if float(row["full_lower"]) > max(
                             float(other["full_upper"]) for other in rows if other is not row
                         )]
            if certified:
                if certified[0] == labels[index]:
                    certified_correct += 1
                else:
                    certified_wrong += 1
            else:
                uncertified += 1
    result = {"evaluated_records": complete, "strict_unique_correct": unique_correct,
              "strict_unique_accuracy": unique_correct / complete if complete else None}
    if certified_correct + certified_wrong + uncertified:
        result.update(full_certified_correct=certified_correct,
                      full_certified_wrong=certified_wrong,
                      full_uncertified=uncertified)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
