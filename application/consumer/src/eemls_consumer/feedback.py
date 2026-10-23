from __future__ import annotations

from collections import deque
from collections.abc import Callable

from eemls.interfaces.control_plane_api import ReturnedLabel


class LabelsToReturn:
    def __init__(
        self, *, fraction: float, label_delay_queries: int, draw: Callable[[], float]
    ) -> None:
        self._fraction = fraction
        self._label_delay_queries = label_delay_queries
        self._draw = draw
        self._waiting: deque[tuple[int, ReturnedLabel]] = deque()

    def kept(self) -> bool:
        return self._draw() < self._fraction

    def recorded(self, *, query_id: str, true_label: str, at_query: int, kept: bool) -> None:
        if not kept:
            return
        self._waiting.append(
            (
                at_query + self._label_delay_queries,
                ReturnedLabel(query_id=query_id, true_label=true_label),
            )
        )

    def due(self, *, after_queries: int) -> tuple[ReturnedLabel, ...]:
        ready: list[ReturnedLabel] = []
        while self._waiting and self._waiting[0][0] <= after_queries:
            ready.append(self._waiting.popleft()[1])
        return tuple(ready)

    def rest(self) -> tuple[ReturnedLabel, ...]:
        ready = tuple(returned for _, returned in self._waiting)
        self._waiting.clear()
        return ready
