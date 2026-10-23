from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Literal

from eemls.domain.errors import EemlsError
from eemls.domain.types import FrozenModel, Milliseconds, ResultOutcome

Closed = Literal["matched", "lost"]


class ResultError(EemlsError): ...


class RoundTrip(FrozenModel):
    round_trip_ms: Milliseconds


class Answered(FrozenModel):
    outcome: ResultOutcome
    round_trip: RoundTrip | None = None


class Pending:
    def __init__(self, *, timeout_ms: float, remember: int = 4096) -> None:
        self._timeout_s = timeout_ms / 1000.0
        self._remember = remember
        self._lock = threading.Lock()
        self._open: OrderedDict[str, float] = OrderedDict()
        self._closed: OrderedDict[str, Closed] = OrderedDict()

    @property
    def in_flight(self) -> int:
        with self._lock:
            return len(self._open)

    def sent(self, *, correlation_id: str, at_s: float) -> None:
        with self._lock:
            if correlation_id in self._open:
                raise ResultError(f"{correlation_id} already names a request in flight")
            self._open[correlation_id] = at_s

    def withdrawn(self, *, correlation_id: str) -> None:
        with self._lock:
            self._open.pop(correlation_id, None)

    def arrived(self, *, correlation_id: str, at_s: float) -> Answered:
        with self._lock:
            sent_at = self._open.pop(correlation_id, None)
            if sent_at is None:
                return self._nothing_open(correlation_id)
            self._keep(correlation_id, "matched")
            return Answered(
                outcome="matched",
                round_trip=RoundTrip(round_trip_ms=max((at_s - sent_at) * 1000.0, 0.0)),
            )

    def expired(self, *, now_s: float) -> tuple[str, ...]:
        with self._lock:
            gone = tuple(
                correlation_id
                for correlation_id, sent_at in self._open.items()
                if now_s - sent_at > self._timeout_s
            )
            for correlation_id in gone:
                del self._open[correlation_id]
                self._keep(correlation_id, "lost")
            return gone

    def _nothing_open(self, correlation_id: str) -> Answered:
        closed = self._closed.get(correlation_id)
        if closed == "matched":
            return Answered(outcome="duplicate")
        if closed == "lost":
            return Answered(outcome="late")
        return Answered(outcome="unknown")

    def _keep(self, correlation_id: str, closed: Closed) -> None:
        self._closed[correlation_id] = closed
        while len(self._closed) > self._remember:
            self._closed.popitem(last=False)
