from __future__ import annotations

from eemls.domain.types import Condition
from eemls.ensemble_optimization.approaches.cap.context_signals import mean_confidence
from eemls.evaluation_engine.window import Window


class NightDetector:
    def __init__(self, *, night_confidence_drop: float) -> None:
        self._drop = night_confidence_drop
        self._day_level = 0.0
        self._day_windows = 0
        self._last: Condition = "day"

    def read(self, window: Window) -> Condition:
        confidence = mean_confidence(window)
        if confidence is None:
            return self._last
        if self._day_windows and confidence < self._day_level - self._drop:
            self._last = "night"
            return self._last
        self._day_windows += 1
        self._day_level += (confidence - self._day_level) / self._day_windows
        self._last = "day"
        return self._last
