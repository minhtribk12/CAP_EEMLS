from __future__ import annotations

import httpx
import pytest

from eemls.image_catalogue import Catalogue, CatalogueError
from eemls.messaging.result_listener import Delivered
from eemls.monitoring.metrics import Monitoring
from eemls.testing.metrics import histogram_count, histogram_sum, series
from eemls_consumer.query_sender import QuerySender
from support.consumer import (
    CONSUMER,
    DAY_LENGTH_S,
    IDS,
    TOPIC,
    Moment,
    Stub,
    built,
    label_count,
)


def driven(client: QuerySender, moment: Moment, moments: list[float]) -> None:
    for at in moments:
        moment.at = at
        client.tick()


def failures(client: QuerySender, monitoring: Monitoring, moment: Moment) -> float | None:
    moment.at = 10.0
    assert client.expire() == 0
    body, _ = monitoring.exposition()
    return series(body, "eemls_client_requests_total", consumer=CONSUMER, outcome="failed")


class TestWhatItAsksAbout:
    def test_the_class_mix_of_the_period_decides_what_it_asks_about(self) -> None:
        stub = Stub()
        client, _, moment = built(stub, rate=1.0)

        driven(client, moment, [1.0, 11.0])

        assert stub.queries[0]["image_id"] == "ped-clear-0"
        assert stub.queries[-1]["image_id"] == "car-clear-0"

    def test_a_data_quality_event_turns_it_over_to_the_blurred_copies_and_back(self) -> None:
        stub = Stub()
        client, _, moment = built(stub, rate=1.0)

        driven(client, moment, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        assert [one["image_quality"] for one in stub.queries] == [
            "clear",
            "clear",
            "clear",
            "clear",
            "blurred",
            "blurred",
            "clear",
            "clear",
        ]
        assert stub.queries[4]["image_id"] == "ped-blurred-0"

    def test_every_query_carries_its_place_in_the_simulated_day(self) -> None:
        stub = Stub()
        client, _, moment = built(stub, rate=1.0)

        driven(client, moment, [1.0, 5.0, 21.0])

        assert [one["time_of_day"] for one in stub.queries] == pytest.approx(
            [index / DAY_LENGTH_S for index in (1, 2, 3, 4, 5)]
            + [(index % DAY_LENGTH_S) / DAY_LENGTH_S for index in range(6, 22)]
        )

    def test_a_query_carries_nothing_of_the_class_the_image_really_is(self) -> None:
        stub = Stub()
        client, _, moment = built(stub, rate=1.0)

        driven(client, moment, [1.0])

        assert set(stub.queries[0]) == {
            "consumer_id",
            "image_id",
            "reply_topic",
            "correlation_id",
            "image_quality",
            "time_of_day",
        }

    def test_every_query_carries_the_reply_address_and_its_own_correlation_id(self) -> None:
        stub = Stub()
        client, _, moment = built(stub)

        driven(client, moment, [0.1])

        assert [one["reply_topic"] for one in stub.queries] == [TOPIC, TOPIC]
        assert [one["correlation_id"] for one in stub.queries] == ["corr-1", "corr-2"]

    def test_a_class_the_hub_holds_no_copy_of_is_a_typed_refusal(self) -> None:
        with pytest.raises(CatalogueError, match="truck"):
            Catalogue(IDS).pick(label="truck", condition="day", image_quality="clear", draw=0.0)


class TestTheRate:
    def test_the_rate_of_the_contract_is_held_over_the_run(self) -> None:
        stub = Stub()
        client, _, moment = built(stub, rate=20.0, send_window=1000)

        driven(client, moment, [10.0])

        assert len(stub.queries) == 200

    def test_it_sends_only_what_has_come_due_since_the_last_tick(self) -> None:
        stub = Stub()
        client, _, moment = built(stub, rate=20.0, send_window=1000)

        driven(client, moment, [1.0, 1.5])

        assert len(stub.queries) == 30


class TestWhatItKeepsOpen:
    def test_it_keeps_no_more_of_its_own_requests_open_than_its_window(self) -> None:
        stub = Stub()
        client, monitoring, moment = built(stub, send_window=2)

        driven(client, moment, [0.25])

        assert (client.accepted, len(stub.queries)) == (2, 2)
        moment.at = 10.0
        assert client.expire() == 2
        body, _ = monitoring.exposition()
        assert series(body, "eemls_client_dropped_total", consumer=CONSUMER) == 3

    def test_a_control_plane_over_its_cap_refuses_and_nothing_is_left_open(self) -> None:
        stub = Stub(status=503)
        client, monitoring, moment = built(stub)

        driven(client, moment, [0.25])

        assert client.accepted == 0
        moment.at = 10.0
        assert client.expire() == 0
        body, _ = monitoring.exposition()
        assert (
            series(body, "eemls_client_requests_total", consumer=CONSUMER, outcome="refused") == 5
        )

    def test_an_answer_no_control_plane_gives_is_counted_a_failure(self) -> None:
        stub = Stub(status=500)
        client, monitoring, moment = built(stub)

        driven(client, moment, [0.25])

        assert client.accepted == 0
        assert failures(client, monitoring, moment) == 5

    def test_an_acceptance_with_no_query_id_is_counted_a_failure(self) -> None:
        stub = Stub(body={"accepted": True})
        client, monitoring, moment = built(stub)

        driven(client, moment, [0.25])

        assert client.accepted == 0
        assert failures(client, monitoring, moment) == 5

    def test_a_control_plane_that_is_not_there_is_counted_a_failure(self) -> None:
        stub = Stub(reachable=False)
        client, monitoring, moment = built(stub)

        driven(client, moment, [0.25])

        assert failures(client, monitoring, moment) == 5


class TestTheResults:
    def test_a_result_is_matched_to_its_request_and_its_round_trip_recorded(self) -> None:
        stub = Stub()
        client, monitoring, moment = built(stub)
        moment.at = 0.05
        client.tick()

        moment.at = 0.08
        client.result(Delivered(topic=TOPIC, correlation_id="corr-1", payload=b"{}"))

        body, _ = monitoring.exposition()
        assert series(body, "eemls_client_results_total", consumer=CONSUMER, result="matched") == 1
        assert histogram_count(body, "eemls_client_round_trip_seconds", consumer=CONSUMER) == 1
        assert histogram_sum(
            body, "eemls_client_round_trip_seconds", consumer=CONSUMER
        ) == pytest.approx(0.030)

    def test_a_result_that_arrives_before_the_send_returns_is_still_matched(self) -> None:
        held: list[Delivered] = []

        def answered(body: dict[str, object]) -> None:
            held.append(
                Delivered(topic=TOPIC, correlation_id=str(body["correlation_id"]), payload=b"{}")
            )
            client.result(held[-1])

        stub = Stub(while_accepting=answered)
        client, monitoring, moment = built(stub)

        driven(client, moment, [0.05])

        body, _ = monitoring.exposition()
        assert series(body, "eemls_client_results_total", consumer=CONSUMER, result="matched") == 1
        assert (
            series(body, "eemls_client_results_total", consumer=CONSUMER, result="unknown") is None
        )

    def test_a_second_delivery_of_one_result_is_dropped(self) -> None:
        stub = Stub()
        client, monitoring, moment = built(stub)
        driven(client, moment, [0.05])
        delivered = Delivered(topic=TOPIC, correlation_id="corr-1", payload=b"{}")

        client.result(delivered)
        client.result(delivered)

        body, _ = monitoring.exposition()
        assert (
            series(body, "eemls_client_results_total", consumer=CONSUMER, result="duplicate") == 1
        )

    def test_a_request_whose_result_never_arrives_is_counted_lost(self) -> None:
        stub = Stub()
        client, monitoring, moment = built(stub)
        driven(client, moment, [0.05])

        moment.at = 3.0
        assert client.expire() == 1
        body, _ = monitoring.exposition()
        assert series(body, "eemls_client_results_total", consumer=CONSUMER, result="lost") == 1

    def test_a_result_for_a_request_already_given_up_on_is_counted_late(self) -> None:
        stub = Stub()
        client, monitoring, moment = built(stub)
        driven(client, moment, [0.05])
        moment.at = 3.0
        client.expire()

        moment.at = 3.5
        client.result(Delivered(topic=TOPIC, correlation_id="corr-1", payload=b"{}"))

        body, _ = monitoring.exposition()
        assert series(body, "eemls_client_results_total", consumer=CONSUMER, result="late") == 1
        assert (
            series(body, "eemls_client_results_total", consumer=CONSUMER, result="duplicate")
            is None
        )


class TestTheReturnedLabels:
    def test_no_label_comes_back_before_its_own_delay_and_about_half_come_back_after(self) -> None:
        stub = Stub()
        client, _, moment = built(
            stub, fraction=0.5, label_delay_queries=500, rate=100.0, send_window=2000
        )
        driven(client, moment, [round(0.1 * step, 2) for step in range(1, 41)])
        early = sum(label_count(one) for one in stub.returned)
        driven(client, moment, [round(0.1 * step, 2) for step in range(41, 101)])

        due = sum(label_count(one) for one in stub.returned)

        assert len(stub.queries) == 1000
        assert early == 0
        assert 200 <= due <= 300

    def test_a_surge_brings_the_labels_back_sooner_in_time(self) -> None:
        stub = Stub()
        client, _, moment = built(
            stub, fraction=1.0, label_delay_queries=100, rate=100.0, send_window=200
        )
        driven(client, moment, [1.0])

        assert sum(label_count(one) for one in stub.returned) == 1

    def test_a_returned_label_carries_the_class_the_image_really_is(self) -> None:
        stub = Stub()
        client, _, moment = built(stub, fraction=1.0, label_delay_queries=0)
        driven(client, moment, [0.05])

        client.returned()

        assert stub.returned[0]["consumer_id"] == CONSUMER
        labels = stub.returned[0]["labels"]
        assert isinstance(labels, list)
        assert labels[0]["true_label"] == "pedestrian"

    def test_the_labels_go_back_in_batches_of_the_configured_size(self) -> None:
        stub = Stub()
        client, _, moment = built(
            stub, fraction=1.0, label_delay_queries=200, rate=20.0, send_window=200
        )
        driven(client, moment, [5.0])

        client.returned_the_rest()

        assert [label_count(one) for one in stub.returned] == [32, 32, 32, 4]

    def test_a_long_tick_hands_labels_back_within_a_decision_interval_of_their_delay(self) -> None:
        stub = Stub()
        client, _, moment = built(
            stub, fraction=1.0, label_delay_queries=75, rate=100.0, send_window=1000
        )
        driven(client, moment, [3.0])
        client.returned()

        ages = [
            after - (int(str(label["query_id"]).removeprefix("q-")) - 1)
            for one, after in zip(stub.returned, stub.returned_after, strict=True)
            for label in one["labels"]  # type: ignore[attr-defined]
        ]
        assert len(stub.queries) == 300
        assert len(ages) == 226
        assert max(ages) <= 75 + 5 - 1

    def test_a_long_tick_hands_labels_back_once_every_decision_interval(self) -> None:
        stub = Stub()
        client, _, moment = built(
            stub, fraction=1.0, label_delay_queries=75, rate=100.0, send_window=1000
        )
        driven(client, moment, [3.0])

        assert len(stub.queries) == 300
        assert len(stub.returned) <= 300 // 5 + 3

    def test_short_ticks_hand_labels_back_within_a_decision_interval_of_their_delay(self) -> None:
        stub = Stub()
        client, _, moment = built(
            stub, fraction=1.0, label_delay_queries=75, rate=100.0, send_window=1000
        )
        driven(client, moment, [round(0.01 * step, 2) for step in range(1, 301)])

        ages = [
            after - (int(str(label["query_id"]).removeprefix("q-")) - 1)
            for one, after in zip(stub.returned, stub.returned_after, strict=True)
            for label in one["labels"]  # type: ignore[attr-defined]
        ]
        assert len(stub.queries) == 300
        assert len(ages) >= 300 - 75 - 5
        assert max(ages) <= 75 + 5 - 1

    def test_short_ticks_hand_labels_back_once_every_decision_interval(self) -> None:
        stub = Stub()
        client, _, moment = built(
            stub, fraction=1.0, label_delay_queries=75, rate=100.0, send_window=1000
        )
        driven(client, moment, [round(0.01 * step, 2) for step in range(1, 301)])

        assert len(stub.queries) == 300
        assert 0 < len(stub.returned) <= 300 // 5 + 3

    def test_labels_the_control_plane_does_not_take_are_counted_every_one(self) -> None:
        stub = Stub(takes_feedback=False)
        client, monitoring, moment = built(stub, fraction=1.0, label_delay_queries=0, rate=20.0)
        driven(client, moment, [2.0])

        assert client.returned() == 0
        assert stub.returned == []
        body, _ = monitoring.exposition()
        assert (
            series(body, "eemls_feedback_ignored_total", consumer=CONSUMER, reason="not_taken")
            == 40
        )

    def test_labels_the_control_plane_refuses_are_counted_every_one(self) -> None:
        stub = Stub(feedback_status=422)
        client, monitoring, moment = built(stub, fraction=1.0, label_delay_queries=0, rate=20.0)
        driven(client, moment, [2.0])

        assert client.returned() == 0
        body, _ = monitoring.exposition()
        assert (
            series(body, "eemls_feedback_ignored_total", consumer=CONSUMER, reason="not_taken")
            == 40
        )


def returned_ids(stub: Stub) -> set[str]:
    ids: set[str] = set()
    for one in stub.returned:
        labels = one["labels"]
        assert isinstance(labels, list)
        ids |= {label["query_id"] for label in labels}
    return ids


class RefusingEveryThird(Stub):
    def answer(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/classify" and len(self.queries) % 3 == 2:
            self.queries.append({})
            return httpx.Response(503, json={"detail": "over the cap"})
        return super().answer(request)


class TestTheSameLabelledQueries:
    def test_a_refused_query_does_not_change_which_later_queries_return_labels(self) -> None:
        plain, refusing = Stub(), RefusingEveryThird()
        for stub in (plain, refusing):
            client, _, moment = built(
                stub, fraction=0.5, label_delay_queries=0, rate=20.0, send_window=200
            )
            driven(client, moment, [5.0])
            client.returned()
        returned = [returned_ids(stub) for stub in (plain, refusing)]
        refused = {f"q-{number}" for number in range(3, 101, 3)}

        assert returned[1] == returned[0] - refused
        assert returned[1]
