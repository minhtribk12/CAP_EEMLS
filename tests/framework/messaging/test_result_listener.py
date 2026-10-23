from __future__ import annotations

from collections.abc import Callable

from eemls.messaging.result_listener import Delivered, ResultListener
from eemls.testing.fake_broker import FakeMqttClient, broker_settings

TOPIC = "eemls/results/c2_safety/client-1"


def listening(
    on_result: Callable[[Delivered], None] | None = None,
    *,
    reachable: bool = True,
    grants: bool = True,
) -> tuple[ResultListener, FakeMqttClient, list[Delivered]]:
    taken: list[Delivered] = []
    held = FakeMqttClient("eemls-client-1", reachable=reachable, grants=grants)
    hook = on_result if on_result is not None else taken.append
    listener = ResultListener(
        broker=broker_settings(),
        client_id="eemls-client-1",
        topic=TOPIC,
        on_result=hook,
        factory=lambda client_id: held,
    )
    return listener, held, taken


class TestTheSession:
    def test_the_session_outlives_the_connection(self) -> None:
        listener, held, _ = listening()

        listener.start()

        assert (held.opened[0].clean_start, held.opened[0].session_expiry_s) == (False, 300)

    def test_it_subscribes_to_its_own_topic_once_the_broker_takes_it(self) -> None:
        listener, held, _ = listening()
        listener.start()

        held.accept()

        assert held.subscriptions == [(TOPIC, 1)]

    def test_it_subscribes_again_after_a_reconnection(self) -> None:
        listener, held, _ = listening()
        listener.start()

        held.accept()
        held.accept()

        assert held.subscriptions == [(TOPIC, 1), (TOPIC, 1)]

    def test_starting_twice_opens_one_connection(self) -> None:
        listener, held, _ = listening()

        listener.start()
        listener.start()

        assert (len(held.opened), held.loops_started) == (1, 1)

    def test_a_broker_that_turns_the_connection_down_leaves_it_unready(self) -> None:
        listener, held, _ = listening()
        listener.start()

        held.refuse()

        assert listener.wait_until_ready(0.0) is False
        assert held.subscriptions == []

    def test_a_subscription_the_broker_does_not_grant_leaves_it_unready(self) -> None:
        listener, held, _ = listening(grants=False)
        listener.start()

        held.accept()

        assert held.subscriptions == [(TOPIC, 1)]
        assert listener.wait_until_ready(0.0) is False

    def test_it_is_ready_once_the_broker_has_granted_the_subscription(self) -> None:
        listener, held, _ = listening()
        listener.start()

        held.accept()

        assert listener.wait_until_ready(0.0) is True


class TestADelivery:
    def test_a_result_is_handed_on_with_its_correlation_id(self) -> None:
        listener, held, taken = listening()
        listener.start()
        held.accept()

        held.deliver(topic=TOPIC, payload=b'{"query_id": "q-1"}', correlation_id="corr-1")

        assert taken == [
            Delivered(topic=TOPIC, correlation_id="corr-1", payload=b'{"query_id": "q-1"}')
        ]

    def test_a_result_naming_no_request_is_dropped_and_counted(self) -> None:
        listener, held, taken = listening()
        listener.start()
        held.accept()

        held.deliver(topic=TOPIC, payload=b"{}", correlation_id=None)

        assert taken == []
        assert listener.unaddressed == 1

    def test_a_correlation_id_no_request_could_have_carried_does_not_end_the_session(self) -> None:
        listener, held, taken = listening()
        listener.start()
        held.accept()

        for unreadable in (b"", b" corr-1", b"\xff\xfe"):
            held.deliver(topic=TOPIC, payload=b"{}", correlation_id=unreadable)
        held.deliver(topic=TOPIC, payload=b"{}", correlation_id="corr-2")

        assert listener.unaddressed == 3
        assert [one.correlation_id for one in taken] == ["corr-2"]

    def test_a_client_that_cannot_take_a_result_does_not_end_the_session(self) -> None:
        def refuse(delivered: Delivered) -> None:
            raise RuntimeError("the client could not take it")

        listener, held, _ = listening(refuse)
        listener.start()
        held.accept()

        held.deliver(topic=TOPIC, payload=b"{}", correlation_id="corr-1")

        assert listener.refused == 1


class TestClosingIt:
    def test_closing_disconnects_and_stops_the_loop(self) -> None:
        listener, held, _ = listening()
        listener.start()

        listener.close()

        assert (held.disconnects, held.loops_stopped) == (1, 1)

    def test_a_broker_nothing_is_listening_on_leaves_the_listener_alive(self) -> None:
        listener, held, _ = listening(reachable=False)

        listener.start()

        assert held.async_opens == 1
        assert listener.wait_until_ready(0.0) is False
