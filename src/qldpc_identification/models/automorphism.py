"""[[18,4,2]] four-step automorphism sequences with a known Y0 marker."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

from qldpc_identification.codes import gf2_rank, validate_code_18_4_2

SYMBOLS = ("I", "x", "x^-1", "y", "y^-1")
Word = tuple[str, str, str, str]
History = tuple[int, int, int, int, int]


def _mask(support: list[int]) -> int:
    return sum(1 << qubit for qubit in support)


def _inverse(perm: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * len(perm)
    for old, new in enumerate(perm):
        result[new] = old
    return tuple(result)


@dataclass(frozen=True)
class AutomorphismModel:
    n: int
    hx: tuple[int, ...]
    hz: tuple[int, ...]
    permutations: dict[str, tuple[int, ...]]
    logical_actions: dict[str, tuple[int, ...]]

    def _permute(self, error: int, symbol: str) -> int:
        perm = self.permutations[symbol]
        return sum(1 << perm[qubit] for qubit in range(self.n)
                   if (error >> qubit) & 1)

    def syndrome(self, x_error: int, z_error: int) -> int:
        z_checks = sum(((row & x_error).bit_count() & 1) << i
                       for i, row in enumerate(self.hz))
        x_checks = sum(((row & z_error).bit_count() & 1) << i
                       for i, row in enumerate(self.hx))
        return z_checks | (x_checks << len(self.hz))

    def history(self, word: Word) -> History:
        """Measure an SEC at entry and after each of four automorphisms."""
        if len(word) != 4 or any(symbol not in SYMBOLS for symbol in word):
            raise ValueError("expected a four-symbol word over I,x,x^-1,y,y^-1")
        x_error = z_error = 1  # Known Y error on data qubit 0.
        observations = [self.syndrome(x_error, z_error)]
        for symbol in word:
            x_error = self._permute(x_error, symbol)
            z_error = self._permute(z_error, symbol)
            observations.append(self.syndrome(x_error, z_error))
        return (observations[0], observations[1], observations[2],
                observations[3], observations[4])

    def net_logical_action(self, word: Word) -> tuple[int, ...]:
        """Compose the frozen-basis logical maps in execution order."""
        action = self.logical_actions["I"]
        for symbol in word:
            action = _compose_binary_rows(self.logical_actions[symbol], action)
        return action

    def logical_action_id(self, word: Word) -> str:
        rows = self.net_logical_action(word)
        return "/".join(f"{row:0{len(rows)}b}" for row in rows)


def _compose_binary_rows(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    size = len(left)
    result = []
    for row in left:
        product = 0
        for column in range(size):
            if (row >> (size - 1 - column)) & 1:
                product ^= right[column]
        result.append(product)
    return tuple(result)


def _inverse_binary_rows(rows: tuple[int, ...]) -> tuple[int, ...]:
    size = len(rows)
    left = list(rows)
    right = [1 << (size - 1 - i) for i in range(size)]
    for column in range(size):
        bit = 1 << (size - 1 - column)
        pivot = next((i for i in range(column, size) if left[i] & bit), None)
        if pivot is None:
            raise ValueError("logical action is not invertible")
        left[column], left[pivot] = left[pivot], left[column]
        right[column], right[pivot] = right[pivot], right[column]
        for row in range(size):
            if row != column and left[row] & bit:
                left[row] ^= left[column]
                right[row] ^= right[column]
    return tuple(right)


def load_automorphism_code_18_4_2(path: Path) -> AutomorphismModel:
    code = validate_code_18_4_2(path)
    n = code["n"]
    hx = tuple(_mask(row) for row in code["Hx_row_supports"])
    hz = tuple(_mask(row) for row in code["Hz_row_supports"])
    permutations = {"I": tuple(range(n))}
    logical_size = 2 * code["k"]
    logical_actions = {"I": tuple(1 << (logical_size - 1 - i)
                                  for i in range(logical_size))}
    for symbol in ("x", "y"):
        perm = tuple(code["automorphisms"][symbol]["physical_old_to_new"])
        if sorted(perm) != list(range(n)):
            raise ValueError(f"{symbol} is not a physical permutation")
        if gf2_rank(list(hx) + [_permute_mask(row, perm) for row in hx]) != code["rank_Hx"]:
            raise ValueError(f"{symbol} does not preserve the X stabilizer span")
        if gf2_rank(list(hz) + [_permute_mask(row, perm) for row in hz]) != code["rank_Hz"]:
            raise ValueError(f"{symbol} does not preserve the Z stabilizer span")
        permutations[symbol] = perm
        permutations[f"{symbol}^-1"] = _inverse(perm)
        binary_rows = code["automorphisms"][symbol]["logical_action_rows_output_by_input"]
        if len(binary_rows) != logical_size or any(
            len(row) != logical_size or set(row) - {"0", "1"} for row in binary_rows
        ):
            raise ValueError(f"{symbol} has an invalid frozen-basis logical action")
        logical_actions[symbol] = tuple(int(row, 2) for row in binary_rows)
        logical_actions[f"{symbol}^-1"] = _inverse_binary_rows(logical_actions[symbol])
    return AutomorphismModel(n, hx, hz, permutations, logical_actions)


def _permute_mask(mask: int, perm: tuple[int, ...]) -> int:
    return sum(1 << perm[qubit] for qubit in range(len(perm)) if (mask >> qubit) & 1)


def ideal_history_catalog(model: AutomorphismModel) -> dict[History, tuple[Word, ...]]:
    grouped: dict[History, list[Word]] = {}
    for word in itertools.product(SYMBOLS, repeat=4):
        history = model.history(word)
        grouped.setdefault(history, []).append(word)
    return {history: tuple(words) for history, words in grouped.items()}
