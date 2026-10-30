"""Exact fault marginalization for automorphisms between perfect SECs."""

from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict

from qldpc_identification.models.automorphism import (
    SYMBOLS,
    AutomorphismModel,
    History,
    Word,
)


def _fault_signatures(model: AutomorphismModel) -> tuple[tuple[int, ...], tuple[int, ...]]:
    x = tuple(model.syndrome(1 << qubit, 0) for qubit in range(model.n))
    z = tuple(model.syndrome(0, 1 << qubit) for qubit in range(model.n))
    return x, z


def exact_syndrome_noise(model: AutomorphismModel, p_x: float, p_z: float) -> dict[int, float]:
    """Sum all 2^(2n) independent X/Z fault patterns by syndrome DP."""
    if not (0 <= p_x < 1 and 0 <= p_z < 1):
        raise ValueError("fault probabilities must lie in [0,1)")
    x, z = _fault_signatures(model)
    mass: dict[int, float] = {0: 1.0}
    for signature, probability in (*((v, p_x) for v in x), *((v, p_z) for v in z)):
        if probability == 0:
            continue
        next_mass: dict[int, float] = defaultdict(float)
        for syndrome, weight in mass.items():
            next_mass[syndrome] += weight * (1 - probability)
            next_mass[syndrome ^ signature] += weight * probability
        mass = dict(next_mass)
    if not math.isclose(sum(mass.values()), 1.0, abs_tol=1e-12, rel_tol=1e-12):
        raise ArithmeticError("exact syndrome noise distribution is not normalized")
    return mass


def _transport_basis(model: AutomorphismModel, symbol: str) -> dict[int, tuple[int, int]]:
    x, z = _fault_signatures(model)
    perm = model.permutations[symbol]
    transformed = (
        *(model.syndrome(1 << perm[q], 0) for q in range(model.n)),
        *(model.syndrome(0, 1 << perm[q]) for q in range(model.n)),
    )
    basis: dict[int, tuple[int, int]] = {}
    for source, target in zip((*x, *z), transformed):
        while source:
            pivot = source.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = (source, target)
                break
            old_source, old_target = basis[pivot]
            source ^= old_source
            target ^= old_target
        if source == 0 and target != 0:
            raise ValueError("physical permutation has no well-defined syndrome action")
    return basis


def _transport(basis: dict[int, tuple[int, int]], syndrome: int) -> int:
    target = 0
    while syndrome:
        pivot = syndrome.bit_length() - 1
        if pivot not in basis:
            raise ValueError("observed syndrome is unreachable")
        source, image = basis[pivot]
        syndrome ^= source
        target ^= image
    return target


def simulate_history(model: AutomorphismModel, word: Word, p_x: float, p_z: float,
                     seed: int) -> History:
    """Generate an observation; the generating word is never passed to inference."""
    if not (0 <= p_x < 1 and 0 <= p_z < 1):
        raise ValueError("fault probabilities must lie in [0,1)")
    rng = random.Random(seed)
    x_error = z_error = 1  # Known Y marker on physical qubit 0.
    observed = [model.syndrome(x_error, z_error)]
    for symbol in word:
        x_error = model._permute(x_error, symbol)
        z_error = model._permute(z_error, symbol)
        for qubit in range(model.n):
            if rng.random() < p_x:
                x_error ^= 1 << qubit
            if rng.random() < p_z:
                z_error ^= 1 << qubit
        observed.append(model.syndrome(x_error, z_error))
    return (observed[0], observed[1], observed[2], observed[3], observed[4])


def exact_word_likelihoods(model: AutomorphismModel, history: History,
                           p_x: float, p_z: float) -> dict[Word, float]:
    """Evaluate every four-symbol word under the declared noise model."""
    if history[0] != model.syndrome(1, 1):
        raise ValueError("entry syndrome differs from the known Y0 marker")
    noise = exact_syndrome_noise(model, p_x, p_z)
    actions = {symbol: _transport_basis(model, symbol) for symbol in SYMBOLS}
    local = [
        {symbol: noise.get(history[i + 1] ^ _transport(actions[symbol], history[i]), 0.0)
         for symbol in SYMBOLS}
        for i in range(4)
    ]
    return {word: math.prod(local[i][symbol] for i, symbol in enumerate(word))
            for word in itertools.product(SYMBOLS, repeat=4)}
