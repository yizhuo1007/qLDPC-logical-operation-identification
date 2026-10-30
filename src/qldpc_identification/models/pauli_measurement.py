"""Ideal [[18,4,2]] gauging law for 15 two-port Pauli measurements."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from qldpc_identification.codes import gf2_rank, validate_code_18_4_2


@dataclass(frozen=True)
class IdealMeasurementModel:
    candidates: tuple[str, ...]
    edge_masks: tuple[int, ...]
    post_masks: dict[str, tuple[int, ...]]
    hidden_dimension: int

    def record(self, candidate: str, hidden: int) -> tuple[int, int]:
        """Return (edge-Z outcomes, post-BB syndrome), bit 0 first."""
        if candidate not in self.post_masks:
            raise KeyError(candidate)
        if not 0 <= hidden < 1 << self.hidden_dimension:
            raise ValueError("hidden branch is outside the ideal support")
        edge = sum(((mask & hidden).bit_count() & 1) << i
                   for i, mask in enumerate(self.edge_masks))
        post = sum(((mask & hidden).bit_count() & 1) << i
                   for i, mask in enumerate(self.post_masks[candidate]))
        return edge, post


def load_ideal_code_18_4_2(data_dir: Path) -> IdealMeasurementModel:
    """Derive candidate maps from frozen graph matchings and code ports."""
    code = validate_code_18_4_2(data_dir / "code_and_ports.json")
    graph = json.loads((data_dir / "fixed_lpu.json").read_text())
    with (data_dir / "measurements_15.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidates = tuple(row["candidate"] for row in rows)
    if len(candidates) != 15 or len(set(candidates)) != 15:
        raise ValueError("expected 15 distinct nonidentity two-port Paulis")
    if set(candidates) != set(graph["matchings_edge_ids"]):
        raise ValueError("candidate catalogue and graph matchings disagree")
    if graph["affine_model"]["offset"] != "all zero":
        raise ValueError("nonzero affine offsets are not implemented")
    if graph["gauging_rounds"] != 2 or graph["pre_bb_rounds"] != 1 or graph["post_bb_rounds"] != 1:
        raise ValueError("unexpected gauging or BB syndrome schedule")
    free = [v for v in graph["vertices_data_qubits"] if v != graph["root"]]
    bit = {vertex: 1 << i for i, vertex in enumerate(free)}
    edge_masks = tuple(bit.get(a, 0) ^ bit.get(b, 0) for a, b in graph["edges"])
    if gf2_rank(list(edge_masks)) != len(free):
        raise ValueError("the edge outcomes must identify each hidden branch")
    post_masks: dict[str, tuple[int, ...]] = {}
    for row in rows:
        label = row["candidate"]
        for component in ("x", "z"):
            support: set[int] = set()
            for port, axis in zip(("a", "b"), label):
                if axis == "I":
                    continue
                generators = ("X", "Z") if axis == "Y" else (axis,)
                for generator in generators:
                    support.symmetric_difference_update(
                        code["selected_ports"][generator + port][component + "_support"]
                    )
            if support != set(json.loads(row[f"physical_{component}_support"])):
                raise ValueError(f"physical {component} representative differs for {label}")
        matchings = graph["matchings_edge_ids"][label]
        masks = tuple(_xor(edge_masks[e] for e in matching) for matching in matchings)
        if len(masks) != len(code["Hx_row_supports"]) + len(code["Hz_row_supports"]):
            raise ValueError(f"unexpected post-BB channel count for {label}")
        if gf2_rank(list(masks)) != int(row["raw_post_syndrome_dimension"]):
            raise ValueError(f"post-BB dimension differs for {label}")
        if len(free) != int(row["edge_dimension"]):
            raise ValueError(f"edge dimension differs for {label}")
        post_masks[label] = masks
    return IdealMeasurementModel(candidates, edge_masks, post_masks, len(free))


def _xor(values) -> int:
    result = 0
    for value in values:
        result ^= value
    return result
