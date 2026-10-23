from __future__ import annotations

import threading
import time
from datetime import timedelta

import psycopg
import pytest

from eemls.config.services import MonitoringConfig
from eemls.domain.records import Feedback, ServedQuery
from eemls.metrics_database.buffered_writer import WriteError
from eemls.metrics_database.database import MetricsDatabase
from eemls.metrics_database.postgres import DatabaseUnavailableError, PostgresDatabase
from eemls.metrics_database.schema import TABLES, run_row
from eemls.metrics_database.testing import window_columns
from eemls.monitoring.metrics import Monitoring
from eemls.testing.metrics import series
from eemls.testing.records import decision, instance, prediction, profile, served
from tests.framework.metrics_database.conftest import PASSWORD, WRITING
from tests.support.docker import a_free_port

pytestmark = pytest.mark.integration

WINDOW = 500


def opened(dsn: str, run_id: str) -> PostgresDatabase:
    held = PostgresDatabase(dsn=dsn, run_id=run_id, **WRITING)
    held.apply_schema()
    return held


def answered(index: int, consumer_id: str = "c1_navigation") -> ServedQuery:
    return served(query_id=f"q-{consumer_id}-{index}", consumer_id=consumer_id)


class TestTheInterface:
    def test_it_is_a_metrics_database(self, dsn: str, run_id: str) -> None:
        assert isinstance(opened(dsn, run_id), MetricsDatabase)

    def test_applying_the_schema_twice_changes_nothing(self, dsn: str, run_id: str) -> None:
        held = opened(dsn, run_id)
        held.apply_schema()

        held.write_query(answered(1))
        held.flush()

        assert len(held.read_window("c1_navigation", WINDOW)) == 1

    def test_a_statement_timeout_of_zero_reads_without_a_limit(self, dsn: str, run_id: str) -> None:
        held = PostgresDatabase(dsn=dsn, run_id=run_id, **(WRITING | {"statement_timeout_ms": 0}))

        with held._pool.connection() as connection:
            shown = connection.execute("SHOW statement_timeout").fetchone()

        assert shown == ("0",)


class TestWhatItStores:
    def test_every_writer_round_trips(self, dsn: str, run_id: str) -> None:
        held = opened(dsn, run_id)
        held.register_instance(instance())
        held.write_query(answered(1))
        held.write_member_predictions((prediction(query_id="q-c1_navigation-1"),))
        held.write_feedback(
            Feedback(
                query_id="q-c1_navigation-1",
                consumer_id="c1_navigation",
                true_label="car",
                t_feedback=answered(1).t_arrive,
                delay_ms=1.0,
            )
        )
        held.write_decision(decision())
        held.write_profile(profile())
        held.flush()

        rows = held.read_window("c1_navigation", WINDOW)

        assert len(rows) == 1
        assert rows[0]["true_label"] == "car"
        assert len(held.read_profiles()) == 1

    def test_a_copy_s_stop_is_set_on_its_row_once(self, dsn: str, run_id: str) -> None:
        held = opened(dsn, run_id)
        held.register_instance(instance())
        stopped = instance().started_at + timedelta(minutes=5)

        held.write_instance_stopped("i-1", stopped)
        held.write_instance_stopped("i-1", stopped + timedelta(minutes=1))

        with psycopg.connect(dsn) as connection:
            rows = connection.execute(
                "SELECT stopped_at FROM instances WHERE run_id = %s", (run_id,)
            ).fetchall()
        assert rows == [(stopped,)]

    def test_a_window_read_returns_every_column_a_metric_may_name(
        self, dsn: str, run_id: str
    ) -> None:
        held = opened(dsn, run_id)
        held.write_query(answered(1))
        held.flush()

        rows = held.read_window("c1_navigation", WINDOW)

        assert set(window_columns()) <= set(rows[0])

    def test_it_returns_the_most_recent_queries_of_one_consumer_only(
        self, dsn: str, run_id: str
    ) -> None:
        held = opened(dsn, run_id)
        for index in range(12):
            held.write_query(answered(index))
        held.write_query(answered(1, consumer_id="c2_safety"))
        held.flush()

        rows = held.read_window("c1_navigation", 10)

        assert len(rows) == 10
        assert {str(row["consumer_id"]) for row in rows} == {"c1_navigation"}


class TestWritingFromSeveralServicesAtOnce:
    def test_no_row_is_lost_when_eight_writers_run_together(self, dsn: str, run_id: str) -> None:
        held = opened(dsn, run_id)
        per_writer = 50

        def write(worker: int) -> None:
            for index in range(per_writer):
                held.write_query(answered(worker * per_writer + index))

        workers = [threading.Thread(target=write, args=(worker,)) for worker in range(8)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        held.flush()

        rows = held.read_window("c1_navigation", 8 * per_writer)

        assert len(rows) == 8 * per_writer

    def test_nothing_held_is_lost_when_the_database_is_closed(self, dsn: str, run_id: str) -> None:
        held = opened(dsn, run_id)
        for index in range(7):
            held.write_query(answered(index))

        held.close()

        after = opened(dsn, run_id)
        assert len(after.read_window("c1_navigation", WINDOW)) == 7


class TestWhatItCosts:
    def test_a_window_read_of_500_rows_per_consumer_is_quick_enough_to_decide_on(
        self, dsn: str, run_id: str
    ) -> None:
        held = opened(dsn, run_id)
        for consumer in ("c1_navigation", "c2_safety", "c3_surveillance"):
            for index in range(WINDOW):
                held.write_query(answered(index, consumer_id=consumer))
                held.write_member_predictions((prediction(query_id=f"q-{consumer}-{index}"),))
        held.flush()
        held.read_window("c1_navigation", WINDOW)

        taken: list[float] = []
        for _ in range(9):
            start = time.perf_counter()
            rows = held.read_window("c1_navigation", WINDOW)
            taken.append((time.perf_counter() - start) * 1000.0)
        assert len(rows) == WINDOW

        middle = sorted(taken)[len(taken) // 2]
        assert middle < 25.0, f"the median window read took {middle:.1f} ms"


class TestWhatItReports:
    def test_the_buffer_depth_and_the_write_are_published(self, dsn: str, run_id: str) -> None:
        monitoring = Monitoring.from_config(
            MonitoringConfig(
                metrics_path="/metrics",
                histogram_buckets_ms=(1.0, 10.0, 100.0),
                fast_histogram_buckets_ms=(0.1, 1.0, 10.0),
                enabled=True,
            )
        )
        held = PostgresDatabase(
            dsn=dsn,
            run_id=run_id,
            monitoring=monitoring,
            buffer_rows=1000,
            flush_interval_ms=WRITING["flush_interval_ms"],
            statement_timeout_ms=WRITING["statement_timeout_ms"],
            max_connections=WRITING["max_connections"],
        )
        held.apply_schema()

        held.write_query(answered(1))
        held.write_query(answered(2))

        assert series(monitoring.exposition()[0], "eemls_metrics_database_buffer_rows") == 2.0

        held.flush()
        body = monitoring.exposition()[0]

        assert series(body, "eemls_metrics_database_buffer_rows") == 0.0
        assert series(body, "eemls_metrics_database_write_seconds_count") == 1.0


class TestWhenSomethingIsWrong:
    def poisoned(self, held: PostgresDatabase) -> None:
        row = dict(run_row("queries", answered(999), run_id=held.run_id))
        row["response_time_ms"] = "not a number"
        held._writer.add("queries", row)

    def test_one_row_the_server_refuses_does_not_block_the_others(
        self, dsn: str, run_id: str
    ) -> None:
        held = opened(dsn, run_id)
        self.poisoned(held)

        for index in range(3):
            held.write_query(answered(index))
        held.flush()

        assert len(held.read_window("c1_navigation", WINDOW)) == 3

    def test_closing_writes_out_what_it_can_and_does_not_raise(self, dsn: str, run_id: str) -> None:
        held = opened(dsn, run_id)
        self.poisoned(held)
        held.write_query(answered(1))

        held.close()

        after = opened(dsn, run_id)
        assert len(after.read_window("c1_navigation", WINDOW)) == 1

    def test_a_write_after_closing_is_refused_rather_than_accepted_and_dropped(
        self, dsn: str, run_id: str
    ) -> None:
        held = opened(dsn, run_id)
        held.close()

        with pytest.raises(WriteError, match="closed"):
            held.write_query(answered(1))

    def test_a_read_answers_what_is_stored_even_when_a_write_is_stuck(
        self, dsn: str, run_id: str
    ) -> None:
        held = opened(dsn, run_id)
        held.write_query(answered(1))
        held.flush()
        self.poisoned(held)

        assert len(held.read_window("c1_navigation", WINDOW)) == 1

    def test_eight_services_applying_the_schema_together_all_succeed(
        self, dsn: str, run_id: str
    ) -> None:
        first = opened(dsn, run_id)
        with first._pool.connection() as connection:
            for table in TABLES:
                connection.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        first.close()
        failed: list[str] = []

        def apply() -> None:
            try:
                PostgresDatabase(dsn=dsn, run_id=run_id, **WRITING).apply_schema()
            except Exception as refused:
                failed.append(type(refused).__name__)

        workers = [threading.Thread(target=apply) for _ in range(8)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        assert not failed


class TestADatabaseThatWillNotTakeTheConnection:
    def test_a_wrong_password_is_refused_with_the_database_s_own_reason(
        self, dsn: str, run_id: str
    ) -> None:
        wrong = dsn.replace(f":{PASSWORD}@", ":not-the-password@")

        with pytest.raises(DatabaseUnavailableError, match="password authentication failed"):
            PostgresDatabase(dsn=wrong, run_id=run_id, connect_timeout_s=2.0, **WRITING)

    def test_a_database_nobody_listens_for_is_refused(self, run_id: str) -> None:
        unused = f"postgresql://eemls:any@127.0.0.1:{a_free_port()}/eemls"

        with pytest.raises(DatabaseUnavailableError, match="refused the connection"):
            PostgresDatabase(dsn=unused, run_id=run_id, connect_timeout_s=2.0, **WRITING)

    def test_the_reason_never_carries_the_password(self, dsn: str, run_id: str) -> None:
        wrong = dsn.replace(f":{PASSWORD}@", ":a-wrong-password@")

        with pytest.raises(DatabaseUnavailableError) as refused:
            PostgresDatabase(dsn=wrong, run_id=run_id, connect_timeout_s=2.0, **WRITING)

        assert "a-wrong-password" not in str(refused.value)
