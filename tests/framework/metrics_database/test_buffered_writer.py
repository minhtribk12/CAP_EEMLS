from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from eemls.config.reading import Reader, validate
from eemls.config.services import MonitoringConfig
from eemls.metrics_database.buffered_writer import Batch, BufferedWriter, WriteError
from eemls.monitoring.metrics import Monitoring
from eemls.testing.metrics import series
from eemls.testing.trees import TREE

MONITORING = "services/monitoring.yaml"


def monitoring_settings() -> MonitoringConfig:
    return validate(MonitoringConfig, Reader(TREE).read_file(MONITORING), MONITORING)


class Sink:
    def __init__(self) -> None:
        self.batches: list[Batch] = []
        self.failures = 0

    def __call__(self, batch: Batch) -> None:
        self.batches.append(batch)

    def rows_of(self, table: str) -> list[Mapping[str, object]]:
        return [row for batch in self.batches for row in batch.get(table, ())]


class FailingSink(Sink):
    def __call__(self, batch: Batch) -> None:
        self.failures += 1
        if self.failures == 1:
            raise RuntimeError("the database went away")
        super().__call__(batch)


@pytest.fixture
def monitoring() -> Monitoring:
    return Monitoring.from_config(monitoring_settings())


class Clock:
    def __init__(self, at: float = 0.0) -> None:
        self.at = at

    def __call__(self) -> float:
        return self.at


def writer(
    sink: Sink,
    monitoring: Monitoring,
    *,
    rows: int = 3,
    interval_ms: float = 200.0,
    now: Sequence[float] | None = None,
    clock: Clock | None = None,
) -> BufferedWriter:
    ticks = iter(now or [0.0] * 50)
    return BufferedWriter(
        sink,
        monitoring=monitoring,
        buffer_rows=rows,
        flush_interval_ms=interval_ms,
        clock=clock or (lambda: next(ticks)),
    )


class TestWhenItFlushes:
    def test_it_holds_rows_until_the_count_is_reached(self, monitoring: Monitoring) -> None:
        sink = Sink()
        held = writer(sink, monitoring, rows=3)

        held.add("queries", {"query_id": "q-1"})
        held.add("queries", {"query_id": "q-2"})

        assert sink.batches == []
        assert held.depth == 2

    def test_it_flushes_once_the_count_is_reached(self, monitoring: Monitoring) -> None:
        sink = Sink()
        held = writer(sink, monitoring, rows=3)

        for index in range(3):
            held.add("queries", {"query_id": f"q-{index}"})

        assert len(sink.batches) == 1
        assert len(sink.rows_of("queries")) == 3
        assert held.depth == 0

    def test_it_flushes_once_what_it_holds_is_old_enough(self, monitoring: Monitoring) -> None:
        sink = Sink()
        held = writer(sink, monitoring, rows=99, interval_ms=200.0, now=[0.0, 0.1, 0.25, 0.25])

        held.add("queries", {"query_id": "q-1"})
        held.add("queries", {"query_id": "q-2"})

        assert len(sink.rows_of("queries")) == 2

    def test_an_age_that_has_not_been_reached_does_not_flush(self, monitoring: Monitoring) -> None:
        sink = Sink()
        held = writer(sink, monitoring, rows=99, interval_ms=200.0, now=[0.0, 0.1, 0.1])

        held.add("queries", {"query_id": "q-1"})
        held.add("queries", {"query_id": "q-2"})

        assert sink.batches == []

    def test_closing_writes_out_what_it_still_holds(self, monitoring: Monitoring) -> None:
        sink = Sink()
        held = writer(sink, monitoring, rows=99)

        held.add("queries", {"query_id": "q-1"})
        held.close()

        assert len(sink.rows_of("queries")) == 1
        assert held.depth == 0

    def test_closing_an_empty_writer_writes_nothing(self, monitoring: Monitoring) -> None:
        sink = Sink()
        held = writer(sink, monitoring)

        held.close()

        assert sink.batches == []

    def test_a_flush_on_request_writes_out_what_it_holds(self, monitoring: Monitoring) -> None:
        sink = Sink()
        held = writer(sink, monitoring, rows=99)

        held.add("queries", {"query_id": "q-1"})
        held.flush()

        assert len(sink.rows_of("queries")) == 1


class TestWhatItKeepsTogether:
    def test_rows_of_several_tables_travel_in_one_batch(self, monitoring: Monitoring) -> None:
        sink = Sink()
        held = writer(sink, monitoring, rows=2)

        held.add("queries", {"query_id": "q-1"})
        held.add("member_predictions", {"query_id": "q-1", "instance_id": "i-1"})

        assert len(sink.batches) == 1
        assert set(sink.batches[0]) == {"queries", "member_predictions"}

    def test_a_query_is_written_before_the_members_that_point_at_it(
        self, monitoring: Monitoring
    ) -> None:
        sink = Sink()
        held = writer(sink, monitoring, rows=2)

        held.add("member_predictions", {"query_id": "q-1", "instance_id": "i-1"})
        held.add("queries", {"query_id": "q-1"})

        assert list(sink.batches[0]) == ["queries", "member_predictions"]

    def test_a_table_the_store_does_not_hold_is_refused(self, monitoring: Monitoring) -> None:
        held = writer(Sink(), monitoring)

        with pytest.raises(ValueError, match="unicorns"):
            held.add("unicorns", {"horn": 1})

    def test_a_run_owner_is_refused_because_no_flush_writes_it(
        self, monitoring: Monitoring
    ) -> None:
        held = writer(Sink(), monitoring)

        with pytest.raises(ValueError, match="run_owners"):
            held.add("run_owners", {"run_id": "r1"})


class TestWhenTheDatabaseFails:
    def test_a_failed_flush_raises_and_keeps_the_rows(self, monitoring: Monitoring) -> None:
        sink = FailingSink()
        held = writer(sink, monitoring, rows=2)

        held.add("queries", {"query_id": "q-1"})
        with pytest.raises(WriteError, match="refused 2 rows"):
            held.add("queries", {"query_id": "q-2"})

        assert held.depth == 2

    def test_the_rows_it_kept_are_written_by_the_next_flush(self, monitoring: Monitoring) -> None:
        sink = FailingSink()
        held = writer(sink, monitoring, rows=2)

        held.add("queries", {"query_id": "q-1"})
        with pytest.raises(WriteError):
            held.add("queries", {"query_id": "q-2"})
        held.flush()

        assert len(sink.rows_of("queries")) == 2


class TestWhatItReports:
    def test_the_depth_appears_on_the_metrics_endpoint(self, monitoring: Monitoring) -> None:
        held = writer(Sink(), monitoring, rows=99)

        held.add("queries", {"query_id": "q-1"})
        held.add("queries", {"query_id": "q-2"})

        assert series(monitoring.exposition()[0], "eemls_metrics_database_buffer_rows") == 2.0

    def test_the_depth_falls_back_to_zero_after_a_flush(self, monitoring: Monitoring) -> None:
        held = writer(Sink(), monitoring, rows=2)

        held.add("queries", {"query_id": "q-1"})
        held.add("queries", {"query_id": "q-2"})

        assert series(monitoring.exposition()[0], "eemls_metrics_database_buffer_rows") == 0.0

    def test_a_flush_is_timed(self, monitoring: Monitoring) -> None:
        held = writer(Sink(), monitoring, rows=1)

        held.add("queries", {"query_id": "q-1"})

        body = monitoring.exposition()[0]

        assert series(body, "eemls_metrics_database_write_seconds_count") == 1.0


class TestWhenTheStoreRefusesABatch:
    def test_the_failure_is_counted_on_the_metrics_endpoint(self, monitoring: Monitoring) -> None:
        held = writer(FailingSink(), monitoring, rows=1)

        with pytest.raises(WriteError):
            held.add("queries", {"query_id": "q-1"})

        assert (
            series(monitoring.exposition()[0], "eemls_metrics_database_write_failures_total") == 1.0
        )

    def test_the_rows_are_still_counted_as_waiting(self, monitoring: Monitoring) -> None:
        held = writer(FailingSink(), monitoring, rows=1)

        with pytest.raises(WriteError):
            held.add("queries", {"query_id": "q-1"})

        assert series(monitoring.exposition()[0], "eemls_metrics_database_buffer_rows") == 1.0

    def test_the_rows_keep_their_order_when_they_are_put_back(self, monitoring: Monitoring) -> None:
        sink = FailingSink()
        held = writer(sink, monitoring, rows=2)

        held.add("queries", {"query_id": "q-1"})
        with pytest.raises(WriteError):
            held.add("queries", {"query_id": "q-2"})
        held.flush()

        assert [row["query_id"] for row in sink.rows_of("queries")] == ["q-1", "q-2"]


class TestARowHandedOverDuringAFlush:
    class ReentrantSink(Sink):
        def __init__(self) -> None:
            super().__init__()
            self.writer: BufferedWriter | None = None
            self.handed_over = False

        def __call__(self, batch: Batch) -> None:
            super().__call__(batch)
            if self.writer is not None and not self.handed_over:
                self.handed_over = True
                self.writer.add("queries", {"query_id": "late"})

    def test_the_late_row_is_not_thrown_away(self, monitoring: Monitoring) -> None:
        sink = self.ReentrantSink()
        held = writer(sink, monitoring, rows=2)
        sink.writer = held

        held.add("queries", {"query_id": "q-1"})
        held.add("queries", {"query_id": "q-2"})

        assert held.depth == 1

    def test_the_late_row_is_written_by_the_next_flush(self, monitoring: Monitoring) -> None:
        sink = self.ReentrantSink()
        held = writer(sink, monitoring, rows=2)
        sink.writer = held

        held.add("queries", {"query_id": "q-1"})
        held.add("queries", {"query_id": "q-2"})
        held.flush()

        assert [row["query_id"] for row in sink.rows_of("queries")] == ["q-1", "q-2", "late"]

    def test_the_late_row_can_still_age_out(self, monitoring: Monitoring) -> None:
        clock = Clock()
        sink = self.ReentrantSink()
        held = writer(sink, monitoring, rows=2, interval_ms=200.0, clock=clock)
        sink.writer = held

        held.add("queries", {"query_id": "q-1"})
        held.add("queries", {"query_id": "q-2"})
        assert held.depth == 1

        clock.at = 5.0
        held.add("queries", {"query_id": "q-3"})

        assert held.depth == 0


class TestTheIntervalBoundary:
    def test_an_age_exactly_at_the_interval_flushes(self, monitoring: Monitoring) -> None:
        sink = Sink()
        held = writer(sink, monitoring, rows=99, interval_ms=200.0, now=[0.0, 0.1, 0.2, 0.2])

        held.add("queries", {"query_id": "q-1"})
        held.add("queries", {"query_id": "q-2"})

        assert len(sink.rows_of("queries")) == 2
