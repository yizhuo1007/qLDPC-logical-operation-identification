"""Validate the inputs shared by supported compiled circuit models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from qldpc_identification.inference.scalable_bv import load_public_records
from qldpc_identification.inference.scalable_mixed import load_public_case


@dataclass(frozen=True)
class CircuitCase:
    name: str
    code: str
    backend: str
    data_dir: Path
    candidates: tuple[str, ...]
    target_of: dict[str, str]
    intervals: tuple[str, ...] = ()
    bits_per_interval: int = 0


@dataclass(frozen=True)
class CircuitObservation:
    bits: tuple[int, ...]
    channels: tuple[str, ...] = ()


def load_circuit_case(path: str | Path) -> CircuitCase:
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    backend = payload["backend"]
    if backend not in ("clifford_order2", "precompiled_mixed"):
        raise ValueError("unsupported circuit backend")
    candidates = tuple(payload["candidates"])
    if not candidates or any(not isinstance(c, str) or not c for c in candidates):
        raise ValueError("candidate IDs must be nonempty strings")
    if len(candidates) != len(set(candidates)):
        raise ValueError("candidate IDs must be unique")
    target_of = payload.get("target_of", {candidate: candidate for candidate in candidates})
    if set(target_of) != set(candidates):
        raise ValueError("target mapping must cover every candidate")
    if any(not isinstance(target, str) or not target for target in target_of.values()):
        raise ValueError("target IDs must be nonempty strings")
    intervals = tuple(payload.get("intervals", ()))
    bits_per_interval = int(payload.get("bits_per_interval", 0))
    if backend == "clifford_order2" and (not intervals or bits_per_interval < 1):
        raise ValueError("Clifford case requires intervals and bits_per_interval")
    if backend == "precompiled_mixed" and (intervals or bits_per_interval):
        raise ValueError("precompiled mixed case does not use interval slicing")
    data_dir = (path.parent / payload["data_dir"]).resolve()
    if not data_dir.is_dir():
        raise ValueError("circuit data directory does not exist")
    if backend == "precompiled_mixed":
        mixed = json.loads((data_dir / "mixed_config.json").read_text())
        if tuple(mixed["candidates"]) != candidates:
            raise ValueError("candidate IDs differ from the compiled mixed model")
    return CircuitCase(payload["name"], payload["code"], backend, data_dir,
                       candidates, dict(target_of), intervals, bits_per_interval)


def load_observation(case: CircuitCase, index: int = 0) -> CircuitObservation:
    if case.backend == "clifford_order2":
        records = load_public_records(case.data_dir)
        if not 0 <= index < len(records):
            raise IndexError("observation index outside public population")
        width = len(case.intervals) * case.bits_per_interval
        if records.shape[1] < width:
            raise ValueError("public history is shorter than the declared circuit")
        return CircuitObservation(tuple(int(bit) for bit in records[index, :width]))
    config, records, channels = load_public_case(case.data_dir)
    if tuple(config["candidates"]) != case.candidates:
        raise ValueError("observation candidates differ from the case")
    if not 0 <= index < len(records):
        raise IndexError("observation index outside public population")
    return CircuitObservation(tuple(int(bit) for bit in records[index]), channels)
