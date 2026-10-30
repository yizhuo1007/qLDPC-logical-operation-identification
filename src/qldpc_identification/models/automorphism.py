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


def load_automorphism_code_18_4_2(path: Path) -> AutomorphismModel:
    code = validate_code_18_4_2(path)
    n = code["n"]
    hx = tuple(_mask(row) for row in code["Hx_row_supports"])
    hz = tuple(_mask(row) for row in code["Hz_row_supports"])
    permutations = {"I": tuple(range(n))}
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
    return AutomorphismModel(n, hx, hz, permutations)


def _permute_mask(mask: int, perm: tuple[int, ...]) -> int:
    return sum(1 << perm[qubit] for qubit in range(len(perm)) if (mask >> qubit) & 1)


def ideal_history_catalog(model: AutomorphismModel) -> dict[History, tuple[Word, ...]]:
    grouped: dict[History, list[Word]] = {}
    for word in itertools.product(SYMBOLS, repeat=4):
        history = model.history(word)
        grouped.setdefault(history, []).append(word)
    return {history: tuple(words) for history, words in grouped.items()}
