from __future__ import annotations

from collections.abc import Sequence
from typing import TypeGuard


def percentile(values: Sequence[float], at: float) -> float:
    """Interpolated percentile."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * at / 100.0
    below = int(position)
    above = min(below + 1, len(ordered) - 1)
    return ordered[below] + (ordered[above] - ordered[below]) * (position - below)


def is_number(held: object) -> TypeGuard[int | float]:
    return isinstance(held, int | float) and not isinstance(held, bool)


def as_number(held: object) -> float:
    if not is_number(held):
        raise TypeError(f"{held!r} is not a number")
    return float(held)
