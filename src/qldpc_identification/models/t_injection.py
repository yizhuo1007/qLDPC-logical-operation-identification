"""Ideal six-axis T-injection law at the BB boundary observation interface."""

from __future__ import annotations

from fractions import Fraction

from qldpc_identification.inference.exact import Distributions, ideal_distributions
from qldpc_identification.models.pauli_measurement import IdealMeasurementModel

AXES = ("XI", "YI", "ZI", "IX", "IY", "IZ")


def validate_joint_instrument() -> None:
    """Check the ideal joint-measurement branches symbolically.

    The factory state is cos(pi/8)|+> + i sin(pi/8)|->. Each pair of
    joint-measurement and destructive-X outcomes has probability 1/4,
    independent of the unknown Pauli axis. The BB boundary record therefore
    carries the axis information in the selected ideal interface.
    """
    c = (Fraction(1), Fraction(0))
    i_s = (Fraction(0), Fraction(1))
    factory = (c, i_s)
    for m in (0, 1):
        sign = (-1) ** m
        for n in (0, 1):
            a, b = factory[n], factory[1 - n]
            actual = (a[0] / 2, a[1] / 2, sign * b[0] / 2, sign * b[1] / 2)
            target = (Fraction(1, 2), Fraction(0), Fraction(0), Fraction(sign, 2))
            if n:
                target = target[2:] + target[:2]
            if actual != tuple((-1) ** (m * n) * value for value in target):
                raise ArithmeticError("joint T-injection branch differs")
            ar, ai, br, bi = actual
            if (ar * ar + br * br, ai * ai + bi * bi,
                    2 * ar * br, 2 * ai * bi) != (
                    Fraction(1, 4), Fraction(1, 4), Fraction(0), Fraction(0)):
                raise ArithmeticError("joint T-injection branch is not axis independent")


def ideal_axis_distributions(model: IdealMeasurementModel) -> Distributions:
    """Compute the six candidate laws seen through BB syndromes and edge-Z bits."""
    validate_joint_instrument()
    if not set(AXES).issubset(model.candidates):
        raise ValueError("the BB fixture lacks one or more T-injection axes")
    return ideal_distributions(model, AXES)
