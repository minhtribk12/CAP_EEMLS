from __future__ import annotations

import json

import pytest

from eemls.domain.types import AggregatedResult
from eemls.messaging.broker import BrokerError
from eemls.messaging.result_publisher import ResultPublisher
from eemls.messaging.topics import TopicError
from eemls.orchestration.result_delivery import ResultDelivery
from eemls.testing.fake_broker import FakeMqttClient, broker_settings
from eemls.testing.records import query

TOPIC = "eemls/results/c1_navigation/client-1"


def result(**changes: object) -> AggregatedResult:
    fields: dict[str, object] = {
        "query_id": "q-1",
        "predicted_label": "car",
        "confidence": 0.82,
        "deployment_id": "c1_navigation-0003",
        "n_members": 3,
        "n_answered": 2,
    }
    return AggregatedResult.model_validate(fields | changes)


def publishing(*, reachable: bool = True) -> tuple[ResultDelivery, FakeMqttClient]:
    held = FakeMqttClient("eemls-control-plane-1", reachable=reachable)
    broker = ResultPublisher(
        broker=broker_settings(),
        client_id="eemls-control-plane-1",
        topic_prefix="eemls/results",
        factory=lambda client_id: held,
    )
    broker.start()
    return ResultDelivery(broker), held


class TestWhereOneResultGoes:
    def test_it_goes_to_the_topic_the_request_named_under_the_id_it_asked_with(self) -> None:
        publisher, held = publishing()

        publisher(query(reply_topic=TOPIC, correlation_id="corr-9"), result())

        assert [(one.topic, one.correlation_id) for one in held.publishes] == [(TOPIC, "corr-9")]

    def test_the_payload_is_the_answer_the_consumer_receives(self) -> None:
        publisher, held = publishing()

        publisher(query(reply_topic=TOPIC), result())

        assert json.loads(held.publishes[0].payload) == {
            "query_id": "q-1",
            "predicted_label": "car",
            "confidence": 0.82,
            "deployment_id": "c1_navigation-0003",
            "n_members": 3,
            "n_answered": 2,
        }


class TestWhenNobodyReceivesIt:
    def test_a_broker_holding_no_connection_refuses_out_loud(self) -> None:
        publisher, _ = publishing(reachable=False)

        with pytest.raises(BrokerError):
            publisher(query(reply_topic=TOPIC), result())

    def test_a_reply_address_of_another_consumer_is_refused(self) -> None:
        publisher, held = publishing()

        with pytest.raises(TopicError):
            publisher(
                query(consumer_id="c1_navigation", reply_topic="eemls/results/c2_safety/client-1"),
                result(),
            )

        assert held.publishes == []
