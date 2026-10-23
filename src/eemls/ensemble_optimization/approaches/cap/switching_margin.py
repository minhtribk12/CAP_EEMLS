from __future__ import annotations

import math


class SwitchingMargin:
    def __init__(self, *, initial: float, contributions_before_margin: int) -> None:
        self._initial = initial
        self._enough = contributions_before_margin
        self._mean = 0.0
        self._squares = 0.0
        self.contributions = 0

    def value(self) -> float:
        held = self.measured()
        return self._initial if held is None else held

    def measured(self) -> float | None:
        if self.contributions < self._enough:
            return None
        return math.sqrt(max(self._squares / self.contributions, 0.0))

    def learn(self, contribution: float) -> None:
        self.contributions += 1
        was = contribution - self._mean
        self._mean += was / self.contributions
        self._squares += was * (contribution - self._mean)
