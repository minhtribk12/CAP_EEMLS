from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Protocol, runtime_checkable


@runtime_checkable
class Measuring(Protocol):
    def timer(
        self, name: str, *, clock: Callable[[], float] = time.perf_counter, **labels: str
    ) -> AbstractContextManager[None]: ...

    def observe(self, name: str, *, seconds: float, **labels: str) -> None: ...

    def count(self, name: str, *, amount: float = 1.0, **labels: str) -> None: ...

    def gauge(self, name: str, *, value: float, **labels: str) -> None: ...


class NoMeasuring:
    def timer(
        self, name: str, *, clock: Callable[[], float] = time.perf_counter, **labels: str
    ) -> AbstractContextManager[None]:
        return nullcontext()

    def observe(self, name: str, *, seconds: float, **labels: str) -> None: ...

    def count(self, name: str, *, amount: float = 1.0, **labels: str) -> None: ...

    def gauge(self, name: str, *, value: float, **labels: str) -> None: ...


def milliseconds_since(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
