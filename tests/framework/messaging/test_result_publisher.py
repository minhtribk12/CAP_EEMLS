from __future__ import annotations

import pytest

from eemls.messaging.broker import BrokerError
from eemls.messaging.result_publisher import ResultPublisher
from eemls.messaging.topics import TopicError
from eemls.testing.fake_broker import FakeMqttClient, broker_settings

TOPIC = "eemls/results/c2_safety/client-1"
CONSUMER = "c2_safety"


def publishing(
    *, reachable: bool = True, max_queued_results: int = 10_000
) -> tuple[ResultPublisher, FakeMqttClient]:
    held = FakeMqttClient("eemls-control-plane-1", reachable=reachable)
    publisher = ResultPublisher(
        broker=broker_settings(max_queued_results=max_queued_results),
        client_id="eemls-control-plane-1",
        topic_prefix="eemls/results",
        factory=lambda client_id: held,
    )
    return publisher, held


class TestOpeningIt:
    def test_it_opens_the_connection_and_lets_the_library_hold_it(self) -> None:
        publisher, held = publishing()

        publisher.start()

        assert [(one.host, one.port, one.keepalive) for one in held.opened] == [
            ("broker", 1883, 30)
        ]
        assert held.loops_started == 1

    def test_starting_twice_opens_one_connection(self) -> None:
        publisher, held = publishing()

        publisher.start()
        publisher.start()

        assert (len(held.opened), held.loops_started) == (1, 1)

    def test_the_backoff_of_the_configuration_is_handed_to_the_library(self) -> None:
        publisher, held = publishing()

        publisher.start()

        assert held.backoff == (0.5, 8.0)

    def test_a_broker_nothing_is_listening_on_leaves_the_publisher_alive(self) -> None:
        publisher, held = publishing(reachable=False)

        publisher.start()

        assert held.async_opens == 1
        assert held.loops_started == 1


class TestPublishingOne:
    def test_a_result_goes_to_the_topic_the_request_named(self) -> None:
        publisher, held = publishing()
        publisher.start()

        publisher.publish(
            topic=TOPIC,
            payload=b'{"query_id": "q-1"}',
            correlation_id="corr-1",
            consumer_id=CONSUMER,
        )

        sent = held.publishes[0]
        assert (sent.topic, sent.payload, sent.correlation_id) == (
            TOPIC,
            b'{"query_id": "q-1"}',
            "corr-1",
        )

    def test_a_result_is_sent_at_the_configured_quality_of_service_and_never_retained(self) -> None:
        publisher, held = publishing()
        publisher.start()

        publisher.publish(topic=TOPIC, payload=b"{}", correlation_id="corr-1", consumer_id=CONSUMER)

        assert (held.publishes[0].qos, held.publishes[0].retain) == (1, False)

    def test_publishing_with_no_connection_is_a_typed_refusal(self) -> None:
        publisher, _ = publishing(reachable=False)
        publisher.start()

        with pytest.raises(BrokerError, match="holds no connection"):
            publisher.publish(
                topic=TOPIC, payload=b"{}", correlation_id="corr-1", consumer_id=CONSUMER
            )

    def test_the_number_of_held_results_is_bounded_and_over_it_the_broker_takes_no_more(
        self,
    ) -> None:
        publisher, held = publishing(max_queued_results=2)
        publisher.start()
        for number in range(2):
            publisher.publish(
                topic=TOPIC,
                payload=b"{}",
                correlation_id=f"corr-{number}",
                consumer_id=CONSUMER,
            )

        with pytest.raises(BrokerError, match="refused"):
            publisher.publish(
                topic=TOPIC, payload=b"{}", correlation_id="corr-2", consumer_id=CONSUMER
            )

        assert held.queue_bound == 2

    def test_a_refused_publish_leaves_the_publisher_usable(self) -> None:
        publisher, held = publishing(reachable=False)
        publisher.start()
        with pytest.raises(BrokerError):
            publisher.publish(
                topic=TOPIC, payload=b"{}", correlation_id="corr-1", consumer_id=CONSUMER
            )

        publisher.close()

        assert held.loops_stopped == 1


class TestTheReplyAddress:
    def test_a_consumer_is_not_answered_on_another_consumer_s_topic(self) -> None:
        publisher, held = publishing()
        publisher.start()

        with pytest.raises(TopicError, match="is not answered on"):
            publisher.publish(
                topic="eemls/results/c1_navigation/client-1",
                payload=b"{}",
                correlation_id="corr-1",
                consumer_id=CONSUMER,
            )

        assert held.publishes == []

    def test_a_reply_address_outside_the_prefix_is_refused(self) -> None:
        publisher, held = publishing()
        publisher.start()

        with pytest.raises(TopicError, match="is not under"):
            publisher.publish(
                topic="somewhere/else/c2_safety/client-1",
                payload=b"{}",
                correlation_id="corr-1",
                consumer_id=CONSUMER,
            )

        assert held.publishes == []

    def test_a_reply_address_naming_no_client_is_refused(self) -> None:
        publisher, _ = publishing()
        publisher.start()

        with pytest.raises(TopicError, match="names 1 part"):
            publisher.publish(
                topic="eemls/results/c2_safety",
                payload=b"{}",
                correlation_id="corr-1",
                consumer_id=CONSUMER,
            )


class TestFlushingWhatTheLibraryStillHolds:
    def test_closing_waits_for_a_result_the_broker_has_confirmed(self) -> None:
        publisher, held = publishing()
        publisher.start()
        publisher.publish(topic=TOPIC, payload=b"{}", correlation_id="c-1", consumer_id=CONSUMER)

        unsent = publisher.flush(0.0)

        assert (unsent, len(held.publishes)) == (0, 1)

    def test_a_result_the_broker_never_confirms_is_counted_and_closing_still_returns(
        self,
    ) -> None:
        publisher, held = publishing()
        held.confirms = False
        publisher.start()
        publisher.publish(topic=TOPIC, payload=b"{}", correlation_id="c-1", consumer_id=CONSUMER)

        unsent = publisher.flush(0.0)
        publisher.close()

        assert unsent == 1
        assert (held.disconnects, held.loops_stopped) == (1, 1)

    def test_a_result_confirmed_while_the_flush_waits_is_not_counted_unsent(self) -> None:
        publisher, held = publishing()
        held.confirms = False
        publisher.start()
        publisher.publish(topic=TOPIC, payload=b"{}", correlation_id="c-1", consumer_id=CONSUMER)
        held.confirm()

        assert publisher.flush(0.0) == 0

    def test_flushing_with_nothing_held_waits_for_nothing(self) -> None:
        publisher, _ = publishing()
        publisher.start()

        assert publisher.flush(0.0) == 0

    def test_closing_flushes_before_it_disconnects(self) -> None:
        publisher, held = publishing()
        publisher.start()
        publisher.publish(topic=TOPIC, payload=b"{}", correlation_id="c-1", consumer_id=CONSUMER)
        held.confirms = False
        publisher.publish(topic=TOPIC, payload=b"{}", correlation_id="c-2", consumer_id=CONSUMER)

        publisher.close()

        assert publisher.unsent_on_close == 1

    def test_what_the_library_confirmed_is_not_held_against_the_next_flush(self) -> None:
        publisher, held = publishing()
        publisher.start()
        for index in range(50):
            publisher.publish(
                topic=TOPIC, payload=b"{}", correlation_id=f"c-{index}", consumer_id=CONSUMER
            )
        held.confirms = False
        publisher.publish(topic=TOPIC, payload=b"{}", correlation_id="c-50", consumer_id=CONSUMER)

        assert publisher.flush(0.0) == 1
        assert len(held.publishes) == 51


class TestClosingIt:
    def test_closing_disconnects_and_stops_the_loop(self) -> None:
        publisher, held = publishing()
        publisher.start()

        publisher.close()

        assert (held.disconnects, held.loops_stopped) == (1, 1)

    def test_closing_twice_is_the_same_as_closing_once(self) -> None:
        publisher, held = publishing()
        publisher.start()

        publisher.close()
        publisher.close()

        assert held.loops_stopped == 1
