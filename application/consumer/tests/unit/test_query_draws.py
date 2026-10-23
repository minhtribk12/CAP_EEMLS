from __future__ import annotations

import pytest

from eemls.domain.random_seeds import consumer_seed
from eemls_consumer.main import random_generators_of
from eemls_consumer.query_schedule import QuerySurge, Schedule
from support.consumer import DAY_LENGTH_S, PERIODS, Stub, built


def test_each_query_takes_its_period_and_quality_from_its_planned_time() -> None:
    stub = Stub()
    client, _, moment = built(stub, rate=1.0)

    moment.at = 10.5
    client.tick()

    assert [query["image_id"] for query in stub.queries] == [
        "ped-clear-0",
        "ped-clear-0",
        "ped-clear-0",
        "ped-clear-0",
        "ped-blurred-0",
        "ped-blurred-0",
        "ped-clear-0",
        "ped-clear-0",
        "ped-clear-0",
        "car-clear-0",
    ]


def test_a_dropped_query_still_takes_its_draws() -> None:
    drawn: list[float] = []

    def draw() -> float:
        drawn.append(0.0)
        return 0.0

    stub = Stub()
    client, _, moment = built(stub, send_window=0, rate=1.0, draw=draw)

    moment.at = 3.0
    client.tick()

    assert stub.queries == []
    assert len(drawn) == 6


def test_the_queries_drawn_do_not_depend_on_the_labels_returned() -> None:
    queries, labels = random_generators_of(3, "c1_navigation")
    for _ in range(5):
        labels.random()

    assert queries.random() == random_generators_of(3, "c1_navigation")[0].random()


def test_the_same_run_seed_gives_a_consumer_the_same_draws() -> None:
    first, again = (
        random_generators_of(3, "c1_navigation"),
        random_generators_of(3, "c1_navigation"),
    )

    assert [first[0].random(), first[1].random()] == [again[0].random(), again[1].random()]


def test_each_consumer_of_a_run_draws_its_own_queries_and_labels() -> None:
    one, other = random_generators_of(3, "c1_navigation"), random_generators_of(3, "c2_safety")

    assert one[0].random() != other[0].random()
    assert one[1].random() != other[1].random()


def test_a_consumer_s_seed_is_the_same_on_every_machine() -> None:
    assert consumer_seed(3, "c1_navigation") == int.from_bytes(
        bytes.fromhex("e5c7e65e19fadb91661fba14b6d7c824e8058fc10b18e804fe11a2590b2d1b69"[:16]),
        "big",
    )


@pytest.mark.parametrize("query", [1, 7, 20, 31, 45])
def test_a_query_is_planned_at_the_moment_it_falls_due(query: int) -> None:
    schedule = Schedule(
        request_rate_per_s=2.0,
        periods=PERIODS,
        day_length_s=DAY_LENGTH_S,
        surges=(QuerySurge(starts_at_s=5.0, ends_at_s=10.0, multiplier=3.0),),
    )

    planned = schedule.planned_at(query)

    assert schedule.due_by(planned + 1e-9) >= query
    assert schedule.due_by(planned - 1e-6) < query
