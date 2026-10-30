"""Small GF(2) checks for the frozen [[18,4,2]] example code."""

from __future__ import annotations

import json
from pathlib import Path


def gf2_rank(rows: list[int] | tuple[int, ...]) -> int:
    pivots: dict[int, int] = {}
    for original in rows:
        row = original
        while row:
            pivot = row.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = row
                break
            row ^= pivots[pivot]
    return len(pivots)


def _mask(support: list[int]) -> int:
    return sum(1 << qubit for qubit in support)


def validate_code_18_4_2(path: Path) -> dict:
    """Check the published fixture's CSS ranks and commutation relation."""
    code = json.loads(path.read_text())
    n = code["n"]
    hx = [_mask(row) for row in code["Hx_row_supports"]]
    hz = [_mask(row) for row in code["Hz_row_supports"]]
    if (n, code["k"]) != (18, 4):
        raise ValueError("this release expects the [[18,4,2]] fixture")
    if gf2_rank(hx) != code["rank_Hx"] or gf2_rank(hz) != code["rank_Hz"]:
        raise ValueError("stabilizer rank differs from the frozen fixture")
    if gf2_rank(hx) + gf2_rank(hz) != n - code["k"]:
        raise ValueError("stabilizer ranks do not encode four logical qubits")
    if any((x & z).bit_count() % 2 for x in hx for z in hz):
        raise ValueError("X and Z stabilizers do not commute")
    return code
