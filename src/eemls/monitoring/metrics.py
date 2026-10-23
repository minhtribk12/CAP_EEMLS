from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager

from eemls.config.services import MonitoringConfig
from eemls.domain.errors import EemlsError
from eemls.monitoring.declared_metrics import ALLOWED_VALUES, DECLARED, build_instruments, expose


class MetricError(EemlsError): ...


class Monitoring:
    def __init__(
        self,
        *,
        metrics_path: str,
        histogram_buckets_ms: Sequence[float],
        fast_histogram_buckets_ms: Sequence[float],
        enabled: bool = True,
    ) -> None:
        self.path = metrics_path
        self.enabled = enabled
        self._registry, self._instruments = build_instruments(
            [edge / 1000.0 for edge in histogram_buckets_ms],
            [edge / 1000.0 for edge in fast_histogram_buckets_ms],
            enabled=enabled,
        )

    @classmethod
    def from_config(cls, settings: MonitoringConfig) -> Monitoring:
        return cls(
            metrics_path=settings.metrics_path,
            histogram_buckets_ms=settings.histogram_buckets_ms,
            fast_histogram_buckets_ms=settings.fast_histogram_buckets_ms,
            enabled=settings.enabled,
        )

    def _check(self, name: str, kind: str, labels: Mapping[str, str]) -> None:
        declared = DECLARED.get(name)
        if declared is None:
            raise MetricError(f"{name} is not a declared metric")
        if declared.kind != kind:
            raise MetricError(f"{name} is a {declared.kind}, so it cannot be used as a {kind}")
        given, expected = set(labels), set(declared.labels)
        if given != expected:
            missing = sorted(expected - given)
            unknown = sorted(given - expected)
            raise MetricError(
                f"{name} takes the labels {sorted(expected)}: missing {missing}, unknown {unknown}"
            )
        for label, value in labels.items():
            allowed = ALLOWED_VALUES.get(label)
            if allowed is not None and value not in allowed:
                raise MetricError(f"{name} label {label}={value!r} is not one of {list(allowed)}")

    @staticmethod
    def _number(name: str, value: float, *, allow_negative: bool) -> float:
        if not math.isfinite(value):
            raise MetricError(f"{name} was given {value}, which is not a finite number")
        if not allow_negative and value < 0:
            raise MetricError(f"{name} was given {value}, and it only counts upwards")
        return value

    @contextmanager
    def timer(
        self, name: str, *, clock: Callable[[], float] = time.perf_counter, **labels: str
    ) -> Iterator[None]:
        self._check(name, "histogram", labels)
        histogram = self._instruments.histogram(name, labels) if self.enabled else None
        started = clock()
        try:
            yield
        finally:
            if histogram is not None:
                histogram.observe(clock() - started)

    def observe(self, name: str, *, seconds: float, **labels: str) -> None:
        self._check(name, "histogram", labels)
        checked = self._number(name, seconds, allow_negative=False)
        if self.enabled:
            self._instruments.histogram(name, labels).observe(checked)

    def count(self, name: str, *, amount: float = 1.0, **labels: str) -> None:
        self._check(name, "counter", labels)
        checked = self._number(name, amount, allow_negative=False)
        if self.enabled:
            self._instruments.counter(name, labels).inc(checked)

    def gauge(self, name: str, *, value: float, **labels: str) -> None:
        self._check(name, "gauge", labels)
        checked = self._number(name, value, allow_negative=True)
        if self.enabled:
            self._instruments.gauge(name, labels).set(checked)

    def exposition(self) -> tuple[str, str]:
        return expose(self._registry)


Monitoring = Monitoring
