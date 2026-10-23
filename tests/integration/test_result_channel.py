from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Iterator

import pytest
from paho.mqtt.packettypes import PacketTypes

from eemls.messaging.broker import (
    BrokerError,
    client_of,
    close_wakeup_sockets,
    properties_for,
)
from eemls.messaging.result_listener import Delivered, ResultListener
from eemls.messaging.result_publisher import ResultPublisher
from eemls.messaging.topics import result_topic
from eemls.testing.fake_broker import broker_settings
from eemls_consumer.results import Pending
from tests.support.docker import a_free_port

pytestmark = pytest.mark.integration

PREFIX = "eemls/results"
CONSUMER = "c2_safety"
READY_SECONDS = 10.0
QUIET_SECONDS = 0.2


class Taken:
    def __init__(self) -> None:
        self.results: list[Delivered] = []
        self.arrived = threading.Event()

    def take(self, delivered: Delivered) -> None:
        self.results.append(delivered)
        self.arrived.set()

    def wait(self, count: int = 1) -> bool:
        while len(self.results) < count:
            self.arrived.clear()
            if not self.arrived.wait(READY_SECONDS):
                return False
        return True


def listener_on(
    host: str, port: int, *, client_id: str, topic: str, taken: Taken
) -> ResultListener:
    return ResultListener(
        broker=broker_settings(host=host, port=port),
        client_id=client_id,
        topic=topic,
        on_result=taken.take,
    )


def publisher_on(host: str, port: int) -> ResultPublisher:
    return ResultPublisher(
        broker=broker_settings(host=host, port=port),
        client_id=f"eemls-control-plane-{uuid.uuid4().hex[:8]}",
        topic_prefix=PREFIX,
    )


@pytest.fixture
def channel(
    broker: tuple[str, int],
) -> Iterator[tuple[ResultPublisher, ResultListener, Taken, str]]:
    host, port = broker
    client_id = f"client-{uuid.uuid4().hex[:8]}"
    topic = result_topic(prefix=PREFIX, consumer_id=CONSUMER, client_id=client_id)
    taken = Taken()
    listener = listener_on(host, port, client_id=client_id, topic=topic, taken=taken)
    publisher = publisher_on(host, port)
    listener.start()
    assert listener.wait_until_ready(READY_SECONDS)
    publisher.start()
    try:
        yield publisher, listener, taken, topic
    finally:
        listener.close()
        publisher.close()


class TestARoundTrip:
    def test_a_result_reaches_the_client_that_asked_for_it(
        self, channel: tuple[ResultPublisher, ResultListener, Taken, str]
    ) -> None:
        publisher, _, taken, topic = channel

        publisher.publish(
            topic=topic,
            payload=b'{"query_id": "q-1"}',
            correlation_id="corr-1",
            consumer_id=CONSUMER,
        )

        assert taken.wait()
        assert taken.results[0].correlation_id == "corr-1"
        assert taken.results[0].payload == b'{"query_id": "q-1"}'

    def test_the_client_measures_the_trip_from_sending_to_receiving(
        self, channel: tuple[ResultPublisher, ResultListener, Taken, str]
    ) -> None:
        publisher, _, taken, topic = channel
        pending = Pending(timeout_ms=10_000.0)
        started = time.monotonic()
        pending.sent(correlation_id="corr-1", at_s=time.monotonic() - started)

        publisher.publish(topic=topic, payload=b"{}", correlation_id="corr-1", consumer_id=CONSUMER)

        assert taken.wait()
        answered = pending.arrived(correlation_id="corr-1", at_s=time.monotonic() - started)
        assert answered.outcome == "matched"
        assert answered.round_trip is not None
        assert 0.0 < answered.round_trip.round_trip_ms < 1000.0


class TestARepeat:
    def test_a_result_delivered_twice_is_matched_once(
        self, channel: tuple[ResultPublisher, ResultListener, Taken, str]
    ) -> None:
        publisher, _, taken, topic = channel
        pending = Pending(timeout_ms=10_000.0)
        pending.sent(correlation_id="corr-1", at_s=0.0)

        publisher.publish(topic=topic, payload=b"{}", correlation_id="corr-1", consumer_id=CONSUMER)
        publisher.publish(topic=topic, payload=b"{}", correlation_id="corr-1", consumer_id=CONSUMER)

        assert taken.wait(2)
        outcomes = [
            pending.arrived(correlation_id=held.correlation_id, at_s=0.01).outcome
            for held in taken.results
        ]
        assert outcomes == ["matched", "duplicate"]


class TestAClientThatComesBack:
    def test_a_result_published_while_it_was_away_still_reaches_it(
        self, broker: tuple[str, int]
    ) -> None:
        host, port = broker
        client_id = f"client-{uuid.uuid4().hex[:8]}"
        topic = result_topic(prefix=PREFIX, consumer_id=CONSUMER, client_id=client_id)
        away = Taken()
        gone = listener_on(host, port, client_id=client_id, topic=topic, taken=away)
        gone.start()
        assert gone.wait_until_ready(READY_SECONDS)
        gone.close()
        taken = Taken()
        back = listener_on(host, port, client_id=client_id, topic=topic, taken=taken)
        publisher = publisher_on(host, port)
        publisher.start()
        try:
            publisher.publish(
                topic=topic,
                payload=b'{"query_id": "q-9"}',
                correlation_id="corr-9",
                consumer_id=CONSUMER,
            )

            back.start()

            assert back.wait_until_ready(READY_SECONDS)
            assert taken.wait()
            assert taken.results[0].correlation_id == "corr-9"
            assert away.results == []
        finally:
            back.close()
            publisher.close()


class TestADeliveryThatNamesNoRequest:
    def test_a_correlation_id_no_request_could_carry_does_not_end_the_session(
        self, broker: tuple[str, int], channel: tuple[ResultPublisher, ResultListener, Taken, str]
    ) -> None:
        host, port = broker
        publisher, listener, taken, topic = channel
        stray = client_of(f"stray-{uuid.uuid4().hex[:8]}")
        stray.connect(host, port, keepalive=30)
        stray.loop_start()
        try:
            for unreadable in (b"", b" corr-1", b"\xff\xfe"):
                properties = properties_for(PacketTypes.PUBLISH)
                properties.CorrelationData = unreadable
                sent = stray.publish(topic, b"{}", qos=1, retain=False, properties=properties)
                sent.wait_for_publish(READY_SECONDS)

            publisher.publish(
                topic=topic, payload=b"{}", correlation_id="corr-after", consumer_id=CONSUMER
            )

            assert taken.wait()
            assert [one.correlation_id for one in taken.results] == ["corr-after"]
            assert listener.unaddressed == 3
        finally:
            stray.disconnect()
            stray.loop_stop()
            close_wakeup_sockets(stray)


class TestABrokerThatIsNotThere:
    def test_the_publisher_stays_alive_and_says_so(self) -> None:
        publisher = publisher_on("127.0.0.1", a_free_port())
        publisher.start()
        try:
            with pytest.raises(BrokerError, match="holds no connection"):
                publisher.publish(
                    topic="eemls/results/c2_safety/client-1",
                    payload=b"{}",
                    correlation_id="c-1",
                    consumer_id=CONSUMER,
                )
        finally:
            publisher.close()

    def test_the_listener_stays_alive_and_is_never_ready(self) -> None:
        taken = Taken()
        listener = listener_on(
            "127.0.0.1",
            a_free_port(),
            client_id="client-1",
            topic="eemls/results/c2_safety/client-1",
            taken=taken,
        )
        listener.start()
        try:
            assert listener.wait_until_ready(0.5) is False
            assert taken.results == []
        finally:
            listener.close()


class TestAResultThatNeverComes:
    def test_a_request_no_one_answers_is_counted_lost(
        self, channel: tuple[ResultPublisher, ResultListener, Taken, str]
    ) -> None:
        _, _, taken, _ = channel
        pending = Pending(timeout_ms=50.0)
        started = time.monotonic()
        pending.sent(correlation_id="corr-5", at_s=time.monotonic() - started)

        assert taken.arrived.wait(QUIET_SECONDS) is False

        assert taken.results == []
        assert pending.expired(now_s=time.monotonic() - started) == ("corr-5",)
        assert pending.in_flight == 0
