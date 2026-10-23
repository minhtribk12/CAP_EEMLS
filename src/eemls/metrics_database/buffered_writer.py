from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence

from eemls.domain.errors import EemlsError
from eemls.domain.measuring import Measuring
from eemls.domain.types import Row
from eemls.metrics_database.schema import TABLES

Batch = Mapping[str, Sequence[Row]]

logger = logging.getLogger(__name__)

WRITE_ORDER: Sequence[str] = tuple(table for table in TABLES if table != "run_owners")


class WriteError(EemlsError): ...


class BufferedWriter:
    def __init__(
        self,
        sink: Callable[[Batch], None],
        *,
        monitoring: Measuring,
        buffer_rows: int,
        flush_interval_ms: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sink = sink
        self._monitoring = monitoring
        self._buffer_rows = buffer_rows
        self._flush_interval_s = flush_interval_ms / 1000.0
        self._clock = clock
        self._lock = threading.Lock()
        self._held: MutableMapping[str, list[Row]] = {}
        self._oldest: float | None = None

    @property
    def depth(self) -> int:
        with self._lock:
            return sum(len(rows) for rows in self._held.values())

    def add(self, table: str, row: Row) -> None:
        if table not in WRITE_ORDER:
            raise ValueError(f"{table} is not a table this writer holds rows for")
        with self._lock:
            self._held.setdefault(table, []).append(row)
            if self._oldest is None:
                self._oldest = self._clock()
            due = self._is_due()
        self._report_depth()
        if due:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            if not self._held:
                return
            batch = {
                table: tuple(self._held[table]) for table in WRITE_ORDER if self._held.get(table)
            }
            self._held = {}
            self._oldest = None
        try:
            with self._monitoring.timer("eemls_metrics_database_write_seconds"):
                self._sink(batch)
        except Exception as refused:
            self._put_back(batch)
            self._monitoring.count("eemls_metrics_database_write_failures_total")
            self._report_depth()
            raise WriteError(
                f"the store refused {sum(len(rows) for rows in batch.values())} rows: {refused}"
            ) from refused
        self._report_depth()

    def _put_back(self, batch: Batch) -> None:
        with self._lock:
            for table, rows in batch.items():
                self._held[table] = [*rows, *self._held.get(table, ())]
            if self._oldest is None:
                self._oldest = self._clock()

    def close(self) -> None:
        self.flush()

    def flush_quietly(self) -> None:
        try:
            self.flush()
        except WriteError as refused:
            logger.warning(
                f"error_type=write_refused_before_read recovery_action=read_what_is_stored "
                f"detail={refused}"
            )

    def close_quietly(self) -> None:
        try:
            self.close()
        except WriteError as refused:
            logger.error(
                f"error_type=write_refused_on_close rows={self.depth} "
                f"recovery_action=dropped detail={refused}"
            )

    def _is_due(self) -> bool:
        rows = sum(len(held) for held in self._held.values())
        if rows >= self._buffer_rows:
            return True
        oldest = self._oldest
        return oldest is not None and self._clock() - oldest >= self._flush_interval_s

    def _report_depth(self) -> None:
        self._monitoring.gauge("eemls_metrics_database_buffer_rows", value=float(self.depth))
