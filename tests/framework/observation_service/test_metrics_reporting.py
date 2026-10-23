from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import Any

import httpx
import pytest

from eemls.config.reading import Reader, validate
from eemls.config.services import MonitoringConfig
from eemls.domain.types import Row, Timestamp
from eemls.metrics_database.buffered_writer import WriteError
from eemls.metrics_database.testing import MemoryMetricsDatabase
from eemls.monitoring.metrics import Monitoring
from eemls.observation_service.metrics_reporting import MetricsReporter, ReportingDatabase
from eemls.testing.records import AT, decision, feedback, prediction, served
from eemls.testing.trees import TREE

MONITORING = "services/monitoring.yaml"
URL = "http://observation-service:8083"


@pytest.fixture
def monitoring() -> Monitoring:
    return Monitoring.from_config(
        validate(MonitoringConfig, Reader(TREE).read_file(MONITORING), MONITORING)
    )


class Service:
    def __init__(self, refusals: Sequence[str] = ()) -> None:
        self.refusals = list(refusals)
        self.bodies: list[dict[str, Any]] = []
        self.paths: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if self.refusals:
            refusal = self.refusals.pop(0)
            if refusal == "timeout":
                raise httpx.ReadTimeout("the service did not answer", request=request)
            return httpx.Response(int(refusal))
        body = json.loads(request.content)
        self.bodies.append(body)
        return httpx.Response(200, json={"stored": 0})

    def ids(self, table: str) -> list[str]:
        return [str(row["query_id"]) for body in self.bodies for row in body.get(table, ())]


def writer(service: Service, monitoring: Monitoring, *, rows: int = 100) -> MetricsReporter:
    return MetricsReporter(
        base_url=URL,
        run_id="run-1",
        sender="control-plane-1",
        monitoring=monitoring,
        buffer_rows=rows,
        flush_interval_ms=60_000.0,
        request_timeout_ms=500.0,
        transport=httpx.MockTransport(service),
    )


class TestTheMetricsReporter:
    def test_a_flush_posts_the_held_rows_as_one_batch(self, monitoring: Monitoring) -> None:
        service = Service()
        held = writer(service, monitoring)
        held.add("queries", served())
        held.add("member_predictions", prediction())

        held.flush()

        assert service.paths == ["/rows"]
        assert service.bodies[0]["run_id"] == "run-1"
        assert service.bodies[0]["sender"] == "control-plane-1"
        assert service.ids("queries") == ["q-1"]
        assert service.ids("member_predictions") == ["q-1"]

    @pytest.mark.parametrize("refusal", ["503", "timeout"])
    def test_refused_rows_are_put_back_and_sent_again(
        self, monitoring: Monitoring, refusal: str
    ) -> None:
        service = Service([refusal])
        held = writer(service, monitoring)
        held.add("queries", served())

        with pytest.raises(WriteError):
            held.flush()
        held.flush()
        held.flush()

        assert service.ids("queries") == ["q-1"]

    def test_a_request_carries_at_most_the_buffer_rows(self, monitoring: Monitoring) -> None:
        service = Service()
        held = writer(service, monitoring, rows=2)
        held.add("queries", served(query_id="q-1"))
        held.add("queries", served(query_id="q-2"))

        assert [len(body["queries"]) for body in service.bodies] == [2]

    def test_put_back_rows_go_out_in_pieces_and_in_order(self, monitoring: Monitoring) -> None:
        service = Service(["503", "503"])
        held = writer(service, monitoring, rows=2)
        for index in range(1, 5):
            held.add("queries", served(query_id=f"q-{index}"))
        held.add("feedback", feedback(query_id="q-1"))

        held.flush()

        sizes = [
            sum(len(rows) for rows in body.values() if isinstance(rows, list))
            for body in service.bodies
        ]
        assert max(sizes) <= 2
        assert service.ids("queries") == ["q-1", "q-2", "q-3", "q-4"]
        assert service.ids("feedback") == ["q-1"]

    def test_a_refused_send_while_adding_holds_the_row_without_raising(
        self, monitoring: Monitoring, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = Service(["503"])
        held = writer(service, monitoring, rows=1)

        with caplog.at_level(logging.WARNING):
            held.add("queries", served())

        assert "rows_held_for_the_next_send" in caplog.text
        held.flush()
        assert service.ids("queries") == ["q-1"]

    def test_rows_of_a_wrong_run_are_dropped(self, monitoring: Monitoring) -> None:
        service = Service(["409"])
        held = writer(service, monitoring)
        held.add("decisions", decision())

        held.flush()
        held.flush()

        assert service.paths == ["/rows"]

    def test_a_wrong_run_is_logged_once_per_batch(
        self, monitoring: Monitoring, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = Service(["503", "409", "409"])
        held = writer(service, monitoring, rows=2)
        held.add("queries", served(query_id="q-1"))
        held.add("queries", served(query_id="q-2"))
        caplog.clear()

        with caplog.at_level(logging.ERROR):
            held.add("queries", served(query_id="q-3"))

        assert service.paths == ["/rows"] * 3
        assert caplog.text.count("error_type=wrong_run") == 1
        assert "rows=3" in caplog.text
        held.flush()
        assert service.paths == ["/rows"] * 3

    def test_a_table_the_service_does_not_take_is_refused(self, monitoring: Monitoring) -> None:
        with pytest.raises(ValueError, match="instances"):
            writer(Service(), monitoring).add("instances", served())


class Counted(MemoryMetricsDatabase):
    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self.log = log

    def read_window(self, consumer_id: str, limit: int) -> Sequence[Row]:
        self.log.append("read")
        return super().read_window(consumer_id, limit)

    def read_member_answers(self, *, since: Timestamp, until: Timestamp) -> Sequence[Row]:
        self.log.append("read")
        return super().read_member_answers(since=since, until=until)


class TestTheReportingDatabase:
    def test_writes_go_through_the_metrics_reporter(self, monitoring: Monitoring) -> None:
        service = Service()
        stored = MemoryMetricsDatabase()
        database = ReportingDatabase(stored, writer(service, monitoring))

        database.write_query(served())
        database.write_member_predictions((prediction(),))
        database.write_feedback(feedback())
        database.write_decision(decision())
        database.flush()

        assert stored.queries == {}
        assert service.ids("queries") == ["q-1"]
        assert service.ids("member_predictions") == ["q-1"]
        assert service.ids("feedback") == ["q-1"]
        assert len(service.bodies[0]["decisions"]) == 1

    def test_the_metrics_reporter_is_flushed_before_every_read(
        self, monitoring: Monitoring
    ) -> None:
        log: list[str] = []
        service = Service()

        def recorded(request: httpx.Request) -> httpx.Response:
            log.append("sent")
            return service(request)

        database = ReportingDatabase(Counted(log), writer_with(recorded, monitoring))
        database.write_query(served())
        database.read_window("c1_navigation", 10)
        database.write_query(served(query_id="q-2"))
        database.read_member_answers(since=AT, until=AT + timedelta(minutes=1))

        assert log == ["sent", "read", "sent", "read"]

    def test_a_refused_flush_before_a_read_still_reads(
        self, monitoring: Monitoring, caplog: pytest.LogCaptureFixture
    ) -> None:
        stored = MemoryMetricsDatabase()
        stored.write_query(served())
        database = ReportingDatabase(stored, writer(Service(["503"]), monitoring))
        database.write_query(served(query_id="q-2"))

        with caplog.at_level(logging.WARNING):
            rows = database.read_window("c1_navigation", 10)

        assert [row["query_id"] for row in rows] == ["q-1"]
        assert "write_refused_before_read" in caplog.text

    def test_close_sends_what_is_held(self, monitoring: Monitoring) -> None:
        service = Service()
        stored = MemoryMetricsDatabase()
        database = ReportingDatabase(stored, writer(service, monitoring))
        database.write_query(served())

        database.close()

        assert service.ids("queries") == ["q-1"]
        assert stored.closed


def writer_with(
    handler: Callable[[httpx.Request], httpx.Response], monitoring: Monitoring
) -> MetricsReporter:
    return MetricsReporter(
        base_url=URL,
        run_id="run-1",
        sender="control-plane-1",
        monitoring=monitoring,
        buffer_rows=100,
        flush_interval_ms=60_000.0,
        request_timeout_ms=500.0,
        transport=httpx.MockTransport(handler),
    )
