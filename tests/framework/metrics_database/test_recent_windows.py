from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta

import pytest

from eemls.domain.records import Feedback, ServedQuery
from eemls.domain.types import MemberPrediction
from eemls.metrics_database.postgres import PostgresDatabase
from eemls.metrics_database.recent_windows import RecentWindows
from eemls.testing.records import AT, member, prediction, served
from tests.framework.metrics_database.conftest import WRITING

pytestmark = pytest.mark.integration

KEEP = 40
CONSUMERS = ("c1_navigation", "c2_safety")
MEMBERS = (("infer-a", "x86_cpu"), ("infer-b", "arm_cpu"), ("infer-c", "jetson"))


@pytest.fixture
def stored(dsn: str, run_id: str) -> Iterator[PostgresDatabase]:
    database = PostgresDatabase(dsn=dsn, run_id=run_id, **WRITING)
    database.apply_schema()
    try:
        yield database
    finally:
        database.close()


def query(index: int, consumer_id: str, **changes: object) -> ServedQuery:
    arrived = AT + timedelta(seconds=index // 2)
    return served(
        query_id=f"q-{consumer_id}-{index:03d}",
        consumer_id=consumer_id,
        t_arrive=arrived,
        t_result_ready=arrived + timedelta(milliseconds=174),
        t_publish=arrived + timedelta(milliseconds=180),
        **changes,
    )


def answer(query_id: str, index: int, position: int, name: str, device: str) -> MemberPrediction:
    timed_out = (index + position) % 11 == 0
    return prediction(
        query_id=query_id,
        instance_id=f"{name}-1",
        member=member(name, device),
        total_ms=1000.0 if timed_out else 30.0 + 7.0 * position + (index % 5),
        status="timeout" if timed_out else "ok",
        predicted_label=None if timed_out else ("car", "bus", "truck")[(index + position) % 3],
        confidence=None if timed_out else 0.5 + 0.1 * position,
        fetch_ms=0.0 if timed_out else 2.0,
        inference_ms=0.0 if timed_out else 18.0,
        in_ensemble=position < 2 or index % 3 == 0,
    )


def traffic(into: RecentWindows, count: int = 60) -> None:
    for index in range(count):
        for consumer_id in CONSUMERS:
            one = query(index, consumer_id)
            into.write_query(one)
            into.write_member_predictions(
                tuple(
                    answer(one.query_id, index, position, name, device)
                    for position, (name, device) in enumerate(MEMBERS)
                )
            )
            if index % 2 == 0:
                into.write_feedback(
                    Feedback(
                        query_id=one.query_id,
                        consumer_id=consumer_id,
                        true_label="car" if index % 4 else "bus",
                        t_feedback=AT + timedelta(minutes=5),
                        delay_ms=1000.0 + index,
                    )
                )


@pytest.mark.parametrize("limit", [1, 10, KEEP])
def test_a_query_window_read_from_memory_is_the_one_postgresql_reads(
    stored: PostgresDatabase, limit: int
) -> None:
    recent = RecentWindows(stored, keep=KEEP)
    traffic(recent)

    for consumer_id in CONSUMERS:
        assert recent.read_window(consumer_id, limit) == stored.read_window(consumer_id, limit)


@pytest.mark.parametrize("limit", [1, 10, KEEP])
def test_a_member_window_read_from_memory_is_the_one_postgresql_reads(
    stored: PostgresDatabase, limit: int
) -> None:
    recent = RecentWindows(stored, keep=KEEP)
    traffic(recent)

    for consumer_id in CONSUMERS:
        assert recent.read_member_window(consumer_id, limit) == stored.read_member_window(
            consumer_id, limit
        )


def test_a_window_longer_than_what_is_kept_is_read_from_postgresql(
    stored: PostgresDatabase,
) -> None:
    recent = RecentWindows(stored, keep=KEEP)
    traffic(recent)

    assert recent.read_window("c1_navigation", 55) == stored.read_window("c1_navigation", 55)


def test_a_store_started_on_a_run_already_written_reads_the_same_windows(
    stored: PostgresDatabase,
) -> None:
    traffic(RecentWindows(stored, keep=KEEP))

    restarted = RecentWindows(stored, keep=KEEP)
    restarted.seed(CONSUMERS)

    for consumer_id in CONSUMERS:
        assert restarted.read_window(consumer_id, KEEP) == stored.read_window(consumer_id, KEEP)
        assert restarted.read_member_window(consumer_id, KEEP) == stored.read_member_window(
            consumer_id, KEEP
        )


def test_a_label_returned_after_its_query_was_read_reaches_the_next_read(
    stored: PostgresDatabase,
) -> None:
    recent = RecentWindows(stored, keep=KEEP)
    one = query(0, "c1_navigation")
    recent.write_query(one)
    before = recent.read_window("c1_navigation", KEEP)
    recent.write_feedback(
        Feedback(
            query_id=one.query_id,
            consumer_id="c1_navigation",
            true_label="car",
            t_feedback=AT,
            delay_ms=1.0,
        )
    )

    after = recent.read_window("c1_navigation", KEEP)

    assert before[0]["true_label"] is None
    assert after == stored.read_window("c1_navigation", KEEP)


def test_a_label_returned_before_its_query_was_stored_is_still_joined(
    stored: PostgresDatabase,
) -> None:
    recent = RecentWindows(stored, keep=KEEP)
    one = query(0, "c1_navigation")
    recent.write_feedback(
        Feedback(
            query_id=one.query_id,
            consumer_id="c1_navigation",
            true_label="car",
            t_feedback=AT,
            delay_ms=1.0,
        )
    )
    recent.write_query(one)
    recent.write_member_predictions((answer(one.query_id, 1, 1, "infer-a", "x86_cpu"),))

    assert recent.read_window("c1_navigation", KEEP) == stored.read_window("c1_navigation", KEEP)
    assert recent.read_member_window("c1_navigation", KEEP) == stored.read_member_window(
        "c1_navigation", KEEP
    )
