from __future__ import annotations

from datetime import timedelta

import pytest

from eemls.config.reading import Reader, validate
from eemls.config.services import MonitoringConfig
from eemls.domain.measuring import NoMeasuring
from eemls.domain.records import ServedQuery
from eemls.domain.types import Query
from eemls.metrics_database.testing import MemoryMetricsDatabase
from eemls.monitoring.metrics import Monitoring
from eemls.observation_service.consumer_feedback import ReturnedLabels
from eemls.testing.metrics import series
from eemls.testing.records import AT, served
from eemls.testing.records import query as inbound
from eemls.testing.trees import TREE

MONITORING = "services/monitoring.yaml"


def monitoring_settings() -> MonitoringConfig:
    return validate(MonitoringConfig, Reader(TREE).read_file(MONITORING), MONITORING)


def answered(index: int, consumer_id: str = "c1_navigation") -> ServedQuery:
    arrived = AT + timedelta(seconds=index)
    return served(
        query_id=f"q-{index}",
        consumer_id=consumer_id,
        t_arrive=arrived,
        t_result_ready=arrived + timedelta(milliseconds=174),
        t_publish=arrived + timedelta(milliseconds=180),
    )


def arriving(index: int, consumer_id: str = "c1_navigation") -> Query:
    return inbound(
        query_id=f"q-{index}",
        consumer_id=consumer_id,
        t_arrive=AT + timedelta(seconds=index),
    )


@pytest.fixture
def database() -> MemoryMetricsDatabase:
    return MemoryMetricsDatabase(run_id="feedback")


@pytest.fixture
def returned(database: MemoryMetricsDatabase) -> ReturnedLabels:
    return ReturnedLabels(database, remember=100, monitoring=NoMeasuring())


class TestWhatItRecords:
    def test_a_returned_label_is_stored_with_the_delay_it_took(
        self, database: MemoryMetricsDatabase, returned: ReturnedLabels
    ) -> None:
        returned.arrived(arriving(1))

        kept = returned.label(
            "q-1", consumer_id="c1_navigation", true_label="car", at=AT + timedelta(seconds=16)
        )

        assert kept is True
        held = database.feedback["q-1"]
        assert held.true_label == "car"
        assert held.delay_ms == pytest.approx(15_000.0)
        assert held.t_feedback == AT + timedelta(seconds=16)

    def test_it_never_touches_the_query_it_labels(
        self, database: MemoryMetricsDatabase, returned: ReturnedLabels
    ) -> None:
        query = answered(1)
        database.write_query(query)
        returned.arrived(arriving(1))
        before = database.queries["q-1"]

        returned.label(
            "q-1", consumer_id="c1_navigation", true_label="car", at=AT + timedelta(seconds=9)
        )

        assert database.queries["q-1"] is before


class TestWhatItIgnores:
    def test_a_second_label_for_one_query_changes_nothing(
        self, database: MemoryMetricsDatabase, returned: ReturnedLabels
    ) -> None:
        returned.arrived(arriving(1))
        returned.label(
            "q-1", consumer_id="c1_navigation", true_label="car", at=AT + timedelta(seconds=9)
        )

        again = returned.label(
            "q-1", consumer_id="c1_navigation", true_label="bus", at=AT + timedelta(seconds=10)
        )

        assert again is False
        assert database.feedback["q-1"].true_label == "car"

    def test_a_label_for_a_query_it_never_served_is_ignored(
        self, returned: ReturnedLabels, database: MemoryMetricsDatabase
    ) -> None:
        kept = returned.label("q-unknown", consumer_id="c1_navigation", true_label="car", at=AT)

        assert kept is False
        assert database.feedback == {}

    def test_a_label_from_another_consumer_is_ignored(
        self, returned: ReturnedLabels, database: MemoryMetricsDatabase
    ) -> None:
        returned.arrived(arriving(1))

        kept = returned.label(
            "q-1",
            consumer_id="c3_surveillance",
            true_label="car",
            at=AT + timedelta(seconds=9),
        )

        assert kept is False
        assert database.feedback == {}

    def test_a_label_for_a_query_long_dropped_is_ignored(
        self, database: MemoryMetricsDatabase
    ) -> None:
        returned = ReturnedLabels(database, remember=3, monitoring=NoMeasuring())
        for index in range(5):
            returned.arrived(arriving(index))

        kept = returned.label(
            "q-0", consumer_id="c1_navigation", true_label="car", at=AT + timedelta(seconds=9)
        )

        assert kept is False
        assert set(database.feedback) == set()

    def test_another_consumer_s_queries_never_push_a_query_out(
        self, database: MemoryMetricsDatabase
    ) -> None:
        returned = ReturnedLabels(database, remember=3, monitoring=NoMeasuring())
        returned.arrived(arriving(0, consumer_id="c1_navigation"))
        for index in range(1, 50):
            returned.arrived(arriving(index, consumer_id="c2_safety"))

        kept = returned.label(
            "q-0", consumer_id="c1_navigation", true_label="car", at=AT + timedelta(seconds=9)
        )

        assert kept is True

    def test_a_label_that_arrives_before_the_query_did_is_ignored(
        self, database: MemoryMetricsDatabase, returned: ReturnedLabels
    ) -> None:
        returned.arrived(arriving(5))

        kept = returned.label("q-5", consumer_id="c1_navigation", true_label="car", at=AT)

        assert kept is False
        assert database.feedback == {}


class TestOutOfOrder:
    def test_two_labels_arriving_in_either_order_leave_the_same_rows(
        self, database: MemoryMetricsDatabase, returned: ReturnedLabels
    ) -> None:
        other = MemoryMetricsDatabase(run_id="feedback")
        reversed_order = ReturnedLabels(other, remember=100, monitoring=NoMeasuring())
        for index in (1, 2):
            returned.arrived(arriving(index))
            reversed_order.arrived(arriving(index))

        late = AT + timedelta(seconds=9)
        returned.label("q-1", consumer_id="c1_navigation", true_label="car", at=late)
        returned.label("q-2", consumer_id="c1_navigation", true_label="bus", at=late)
        reversed_order.label("q-2", consumer_id="c1_navigation", true_label="bus", at=late)
        reversed_order.label("q-1", consumer_id="c1_navigation", true_label="car", at=late)

        assert {name: held.true_label for name, held in database.feedback.items()} == {
            name: held.true_label for name, held in other.feedback.items()
        }


class TestWhatItReports:
    def test_what_it_kept_and_what_it_ignored_are_counted(
        self, database: MemoryMetricsDatabase
    ) -> None:
        monitoring = Monitoring.from_config(monitoring_settings())
        returned = ReturnedLabels(database, remember=100, monitoring=monitoring)
        returned.arrived(arriving(1))

        returned.label(
            "q-1", consumer_id="c1_navigation", true_label="car", at=AT + timedelta(seconds=9)
        )
        returned.label(
            "q-unknown",
            consumer_id="c1_navigation",
            true_label="car",
            at=AT + timedelta(seconds=9),
        )

        body = monitoring.exposition()[0]
        assert series(body, "eemls_feedback_total", consumer="c1_navigation") == 1.0
        assert (
            series(
                body,
                "eemls_feedback_ignored_total",
                consumer="c1_navigation",
                reason="no_such_query",
            )
            == 1.0
        )
