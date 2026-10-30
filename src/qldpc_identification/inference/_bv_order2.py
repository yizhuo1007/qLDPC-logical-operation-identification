"""Sparse order-two fault factors for the [[72,12,6]] BV circuit."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
from typing import Any, Iterable

def parity(value: int) -> int:
    return int(value).bit_count() & 1

PROFILES = {"LOW": {"p_cx": 1e-4, "p_1q": 1e-5, "p_measure": 1e-4, "p_reset": 5e-5, "p_idle_per_unit": 1e-5}}

class SyndromeCodec:
    """Independent sparse GF(2) check rows with a deterministic pivot section."""

    def __init__(self, n_data: int, rows: Iterable[int]):
        self.n_data = int(n_data)
        self.width = 2 * self.n_data
        self.rows = tuple(int(row) for row in rows)
        if not self.rows or any(row < 0 or row.bit_length() > self.width for row in self.rows):
            raise ValueError("invalid syndrome rows")
        work = [[row, 1 << index] for index, row in enumerate(self.rows)]
        pivots: list[int] = []
        r = 0
        for column in range(self.width):
            found = next((i for i in range(r, len(work)) if (work[i][0] >> column) & 1), None)
            if found is None:
                continue
            work[r], work[found] = work[found], work[r]
            for i in range(len(work)):
                if i != r and ((work[i][0] >> column) & 1):
                    work[i][0] ^= work[r][0]
                    work[i][1] ^= work[r][1]
            pivots.append(column)
            r += 1
            if r == len(work):
                break
        if r != len(self.rows):
            raise ValueError("syndrome rows must be independent")
        self.rref_rows = tuple(work[i][0] for i in range(r))
        self.row_combinations = tuple(work[i][1] for i in range(r))
        self.pivots = tuple(pivots)
        self.free_columns = tuple(q for q in range(self.width) if q not in set(self.pivots))

    @classmethod
    def from_json(cls, path: str | Path) -> "SyndromeCodec":
        payload = json.loads(Path(path).read_text())
        return cls(int(payload["n_data"]), [int(value, 16) for value in payload["independent_rows_hex"]])

    def syndrome_vector(self, vector: int) -> int:
        vector = int(vector)
        if vector < 0 or vector.bit_length() > self.width:
            raise ValueError("Pauli vector outside codec width")
        return sum(parity(row & vector) << i for i, row in enumerate(self.rows))

    def syndrome(self, x: int, z: int) -> int:
        mask = (1 << self.n_data) - 1
        return self.syndrome_vector((int(x) & mask) | ((int(z) & mask) << self.n_data))

    def section_vector(self, syndrome: int) -> int:
        syndrome = int(syndrome)
        if syndrome < 0 or syndrome.bit_length() > len(self.rows):
            raise ValueError("syndrome outside codec width")
        vector = 0
        for pivot, combination in zip(self.pivots, self.row_combinations):
            if parity(combination & syndrome):
                vector |= 1 << pivot
        if self.syndrome_vector(vector) != syndrome:
            raise AssertionError("syndrome section verification failed")
        return vector

    def section(self, syndrome: int) -> tuple[int, int]:
        vector = self.section_vector(syndrome)
        mask = (1 << self.n_data) - 1
        return vector & mask, (vector >> self.n_data) & mask

    def kernel_basis(self) -> tuple[int, ...]:
        out = []
        for free in self.free_columns:
            vector = 1 << free
            for row, pivot in zip(self.rref_rows, self.pivots):
                if (row >> free) & 1:
                    vector |= 1 << pivot
            if self.syndrome_vector(vector):
                raise AssertionError("kernel construction failed")
            out.append(vector)
        return tuple(out)


@dataclass(frozen=True)
class Step:
    tick: int
    location_id: str
    mechanism: str
    support: tuple[int, ...]
    p_key: str
    measurement_pos: int | None = None
    interval: str = ""


@dataclass(frozen=True)
class Alt:
    weight: float
    key: tuple
    obs_delta: int
    out_x: int
    out_z: int


def _rate_key(mechanism: str) -> str:
    if mechanism == "CX": return "p_cx"
    if mechanism in ("H", "X", "Y", "Z"): return "p_1q"
    if mechanism.startswith("RESET"): return "p_reset"
    if mechanism.startswith("MEASURE"): return "p_measure"
    if mechanism == "IDLE": return "p_idle_per_unit"
    raise ValueError(mechanism)


def parse_stim_intervals(path: str | Path, total_qubits: int = 144) -> dict[str, list[Step]]:
    """Expand a grouped schedule into physical locations and idles."""
    by_interval: dict[str, list[Step]] = defaultdict(list)
    tick = -1
    interval = ""
    pending: list[tuple[str, tuple[int, ...]]] = []
    measurement_count: dict[str, int] = defaultdict(int)
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# tick "):
            fields = line.split()
            tick = int(fields[2])
            interval = line.split("interval=", 1)[1]
            continue
        if line.startswith("#"):
            continue
        if line == "TICK":
            used: set[int] = set()
            index = 0
            for mechanism, support in pending:
                if used.intersection(support):
                    raise ValueError(f"qubit collision in tick {tick}")
                used.update(support)
                pos = None
                if mechanism.startswith("MEASURE"):
                    pos = measurement_count[interval]
                    measurement_count[interval] += 1
                by_interval[interval].append(Step(tick, f"T{tick:03d}:L{index:03d}", mechanism, support, _rate_key(mechanism), pos, interval))
                index += 1
            for qubit in range(total_qubits):
                if qubit not in used:
                    by_interval[interval].append(Step(tick, f"T{tick:03d}:L{index:03d}", "IDLE", (qubit,), "p_idle_per_unit", None, interval))
                    index += 1
            pending = []
            continue
        fields = line.split(); op = fields[0]; values = tuple(int(value) for value in fields[1:])
        name = {"R": "RESET_Z", "RX": "RESET_X", "M": "MEASURE_Z", "MX": "MEASURE_X", "I": "IDLE"}.get(op, op)
        if name == "CX":
            if len(values) % 2: raise ValueError("odd CX arity")
            pending.extend((name, values[i:i + 2]) for i in range(0, len(values), 2))
        else:
            pending.extend((name, (value,)) for value in values)
    if pending:
        raise ValueError("unterminated final tick")
    return dict(by_interval)


def _ideal_apply(x: int, z: int, step: Step) -> tuple[int, int, int]:
    obs = 0
    mechanism = step.mechanism
    if mechanism == "CX":
        c, t = step.support
        if (x >> c) & 1: x ^= 1 << t
        if (z >> t) & 1: z ^= 1 << c
    elif mechanism == "H":
        q = step.support[0]; xb = (x >> q) & 1; zb = (z >> q) & 1
        if xb != zb: x ^= 1 << q; z ^= 1 << q
    elif mechanism in ("X", "Y", "Z", "IDLE"):
        pass
    elif mechanism in ("RESET_Z", "RESET_X"):
        q = step.support[0]; x &= ~(1 << q); z &= ~(1 << q)
    elif mechanism in ("MEASURE_Z", "MEASURE_X"):
        if step.measurement_pos is None: raise AssertionError("measurement index absent")
        q = step.support[0]
        if ((x if mechanism == "MEASURE_Z" else z) >> q) & 1:
            obs = 1 << step.measurement_pos
        x &= ~(1 << q); z &= ~(1 << q)
    else:
        raise ValueError(f"unsupported mechanism {mechanism}")
    return x, z, obs


def _fault_outcomes(step: Step) -> tuple[tuple[str, tuple[int, int], int], ...]:
    mechanism = step.mechanism
    if mechanism in ("IDLE", "H", "X", "Y", "Z"):
        return tuple((p, (int(p in "XY"), int(p in "YZ")), 0) for p in "XYZ")
    if mechanism == "CX":
        out = []
        for left in "IXYZ":
            for right in "IXYZ":
                if left == right == "I": continue
                x = int(left in "XY") | (int(right in "XY") << 1)
                z = int(left in "YZ") | (int(right in "YZ") << 1)
                out.append((left + right, (x, z), 0))
        return tuple(out)
    if mechanism == "RESET_Z": return (("X", (1, 0), 0),)
    if mechanism == "RESET_X": return (("Z", (0, 1), 0),)
    if mechanism in ("MEASURE_Z", "MEASURE_X"):
        return (("FLIP", (0, 0), 1 << int(step.measurement_pos)),)
    raise ValueError(f"unsupported fault catalogue {mechanism}")


def _inject(x: int, z: int, step: Step, outcome: tuple[int, int], flip: int) -> tuple[int, int, int]:
    ox, oz = outcome
    if step.mechanism.startswith("MEASURE"):
        return x, z, flip
    if step.mechanism == "CX":
        a, b = step.support
        if ox & 1: x ^= 1 << a
        if oz & 1: z ^= 1 << a
        if ox & 2: x ^= 1 << b
        if oz & 2: z ^= 1 << b
    else:
        q = step.support[0]
        if ox: x ^= 1 << q
        if oz: z ^= 1 << q
    return x, z, flip


class WideOrder2IntervalFactor:
    """Lazy order-two fault search over arbitrary-width integer states."""

    order = 2

    def __init__(self, steps: Iterable[Step], codec: SyndromeCodec, profile: str = "LOW", local_cap: int = 256, state_cap: int = 1024):
        self.steps = tuple(sorted(steps, key=lambda item: (item.tick, item.location_id)))
        self.codec = codec
        self.profile = str(profile).upper()
        if self.profile not in PROFILES: raise ValueError("unknown profile")
        self.local_cap = int(local_cap); self.state_cap = int(state_cap)
        if not (1 <= self.local_cap <= 256 and 1 <= self.state_cap <= 1024):
            raise ValueError("ORDER2_WIDE caps may not exceed local=256/state=1024")
        self.by_obs: dict[int, list[Alt]] = defaultdict(list)
        self._single_by_obs: dict[int, list[Alt]] = defaultdict(list)
        self._all_single: list[Alt] = []
        self._location_rank: dict[str, int] = {}
        self._pair_query_cache: dict[int, tuple[tuple[Alt, ...], float]] = {}
        self._pair_query_count_cache: dict[int, int] = {}
        self._incoming_cache: dict[int, tuple[int, int, int, int]] = {}
        self.pair_query_count = 0; self.pair_alternative_count = 0
        self._compile()

    def _p(self, step: Step) -> float:
        return float(PROFILES[self.profile][step.p_key])

    def _run(self, start_x: int, start_z: int, inject_at: int | None = None, outcome: tuple[int, int] = (0, 0), flip: int = 0) -> tuple[int, int, int]:
        x, z, obs = int(start_x), int(start_z), 0
        for index, step in enumerate(self.steps):
            x, z, observed = _ideal_apply(x, z, step); obs ^= observed
            if index == inject_at:
                x, z, observed = _inject(x, z, step, outcome, flip); obs ^= observed
        mask = (1 << self.codec.n_data) - 1
        return obs, x & mask, z & mask

    def _compile(self) -> None:
        no_fault = 1.0
        for step in self.steps: no_fault *= 1.0 - self._p(step)
        self.no_fault_weight = no_fault
        self.by_obs[0].append(Alt(no_fault, ("NO_FAULT",), 0, 0, 0))
        single_mass = 0.0
        seen: set[str] = set()
        for index, step in enumerate(self.steps):
            if step.location_id in seen: raise ValueError(f"duplicate location {step.location_id}")
            seen.add(step.location_id); self._location_rank[step.location_id] = index
            p = self._p(step)
            if p <= 0: continue
            outcomes = _fault_outcomes(step); weight = no_fault * p / ((1.0 - p) * len(outcomes)); single_mass += weight * len(outcomes)
            for label, fault, flip in outcomes:
                obs, out_x, out_z = self._run(0, 0, index, fault, flip)
                alt = Alt(weight, (step.location_id, label), obs, out_x, out_z)
                self._single_by_obs[obs].append(alt); self._all_single.append(alt); self.by_obs[obs].append(alt)
        for values in self._single_by_obs.values(): values.sort(key=lambda item: (-item.weight, item.key))
        for values in self.by_obs.values(): values.sort(key=lambda item: (-item.weight, item.key))
        self._all_single.sort(key=lambda item: (self._location_rank[item.key[0]], item.key))
        self._single_obs_mass = {obs: sum(item.weight for item in values) for obs, values in self._single_by_obs.items()}
        self._single_obs_location_mass = {obs: {loc: sum(item.weight for item in values if item.key[0] == loc) for loc in {item.key[0] for item in values}} for obs, values in self._single_by_obs.items()}
        self._single_obs_location_count = {obs: {loc: sum(item.key[0] == loc for item in values) for loc in {item.key[0] for item in values}} for obs, values in self._single_by_obs.items()}
        suffix = 0.0; pair_ratio = 0.0
        for step in reversed(self.steps):
            p = self._p(step); ratio = p / (1.0 - p)
            pair_ratio += ratio * suffix; suffix += ratio
        self.exact_single_mass = single_mass
        self.exact_pair_mass = no_fault * pair_ratio
        self.exact_order_le2_mass = no_fault + single_mass + self.exact_pair_mass
        self.local_order_ge3 = max(0.0, 1.0 - self.exact_order_le2_mass)
        if abs(self.exact_order_le2_mass + self.local_order_ge3 - 1.0) > 2e-11:
            raise ArithmeticError("P0/P1/P2 mass invariant failed")

    def _pair_values(self, required_delta: int) -> tuple[tuple[Alt, ...], float]:
        if required_delta in self._pair_query_cache: return self._pair_query_cache[required_delta]
        base = tuple(self.by_obs.get(required_delta, ())); pair_mass = 0.0; pair_count = 0
        heap: list[tuple[float, tuple, int, int]] = []; rows: list[tuple[Alt, tuple[Alt, ...]]] = []
        for left in self._all_single:
            location = left.key[0]; complement = required_delta ^ left.obs_delta
            rights = tuple(self._single_by_obs.get(complement, ()))
            if not rights: continue
            pair_mass += left.weight * (self._single_obs_mass[complement] - self._single_obs_location_mass[complement].get(location, 0.0)) / self.no_fault_weight
            pair_count += len(rights) - self._single_obs_location_count[complement].get(location, 0)
            ranked = tuple(right for right in rights if self._location_rank[right.key[0]] > self._location_rank[location])
            if ranked:
                row_id = len(rows); rows.append((left, ranked)); first = ranked[0]
                heapq.heappush(heap, (-left.weight * first.weight / self.no_fault_weight, ("PAIR", left.key, first.key), row_id, 0))
        pair_mass *= 0.5
        if pair_count % 2: raise ArithmeticError("unordered pair count is not integral")
        pair_count //= 2
        top: list[Alt] = []
        while heap and len(top) < self.local_cap:
            neg, _, row_id, right_index = heapq.heappop(heap); left, rights = rows[row_id]; right = rights[right_index]
            if right_index + 1 < len(rights):
                nxt = rights[right_index + 1]
                heapq.heappush(heap, (-left.weight * nxt.weight / self.no_fault_weight, ("PAIR", left.key, nxt.key), row_id, right_index + 1))
            top.append(Alt(-neg, ("PAIR", left.key, right.key), left.obs_delta ^ right.obs_delta, left.out_x ^ right.out_x, left.out_z ^ right.out_z))
        values = list(base) + top; values.sort(key=lambda item: (-item.weight, item.key))
        result = (tuple(values), pair_mass); self._pair_query_cache[required_delta] = result; self._pair_query_count_cache[required_delta] = pair_count
        self.pair_query_count += 1; self.pair_alternative_count += len(top)
        return result

    def incoming(self, state: int) -> tuple[int, int, int, int]:
        if state not in self._incoming_cache:
            x, z = self.codec.section(state); obs, out_x, out_z = self._run(x, z)
            self._incoming_cache[state] = (obs, out_x, out_z, self.codec.syndrome(out_x, out_z))
        return self._incoming_cache[state]

    def update(self, frontier: dict[int, float], observed: int, baseline: int) -> tuple[dict[int, float], dict[str, float | int]]:
        nxt: dict[int, float] = defaultdict(float); local_discard = 0.0; compatible = 0; kept = 0
        for state, mass in sorted(frontier.items()):
            incoming_obs, tx, tz, _ = self.incoming(state); need = int(observed) ^ int(baseline) ^ incoming_obs
            values, compatible_pair_mass = self._pair_values(need); compatible += len(self.by_obs.get(need, ())) + self._pair_query_count_cache[need]
            omitted_pair_mass = compatible_pair_mass - sum(item.weight for item in values if item.key[0] == "PAIR")
            compatible_mass = sum(item.weight for item in values) + max(0.0, omitted_pair_mass)
            retained_mass = sum(item.weight for item in values[:self.local_cap])
            local_discard += max(0.0, compatible_mass - retained_mass) * mass
            for alt in values[:self.local_cap]:
                key = self.codec.syndrome(tx ^ alt.out_x, tz ^ alt.out_z); nxt[key] += mass * alt.weight; kept += 1
        pre_state = float(sum(nxt.values()))
        if len(nxt) > self.state_cap:
            ranked = sorted(nxt.items(), key=lambda item: (-item[1], item[0])); state_discard = sum(value for _, value in ranked[self.state_cap:]); nxt = dict(ranked[:self.state_cap])
        else: state_discard = 0.0
        return dict(nxt), {"order_le2_cap_discarded_mass": local_discard, "state_pruned_mass": state_discard, "compatible_paths": compatible, "retained_paths": kept, "pre_state_mass": pre_state, "analytic_ge3_omitted_probability": self.local_order_ge3}


def normalize_scores(scores: dict[str, float], catalogue: tuple[str, ...]) -> tuple[dict[str, float | None], str]:
    if set(scores) != set(catalogue): return {cid: None for cid in catalogue}, "INCOMPLETE_CANDIDATE_COVERAGE"
    if any(math.isnan(float(value)) or float(value) == math.inf for value in scores.values()): return {cid: None for cid in catalogue}, "NUMERICAL_INPUT_FAILURE"
    finite = {cid: float(value) for cid, value in scores.items() if math.isfinite(float(value))}
    if not finite: return {cid: None for cid in catalogue}, "ABSTAIN_ZERO_RETAINED_MASS"
    peak = max(finite.values()); denominator = sum(math.exp(value - peak) for value in finite.values())
    out = {cid: (math.exp(finite[cid] - peak) / denominator if cid in finite else 0.0) for cid in catalogue}
    return out, "ZERO_RETAINED_MASS" if len(finite) != len(catalogue) else "OK_APPROXIMATE_UNCALIBRATED_SCORE"
