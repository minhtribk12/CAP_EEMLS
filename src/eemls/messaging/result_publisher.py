from __future__ import annotations

import logging
import time
from collections import deque

import paho.mqtt.client as mqtt
from paho.mqtt.packettypes import PacketTypes

from eemls.config.services import BrokerConfig
from eemls.messaging.broker import (
    BrokerError,
    Factory,
    client_of,
    connect_in_background,
    disconnect,
    properties_for,
    set_reconnect_delays,
)
from eemls.messaging.topics import TopicError, parts_of

logger = logging.getLogger(__name__)


class ResultPublisher:
    def __init__(
        self,
        *,
        broker: BrokerConfig,
        client_id: str,
        topic_prefix: str,
        factory: Factory = client_of,
    ) -> None:
        self._broker = broker
        self._topic_prefix = topic_prefix
        self._awaiting: deque[mqtt.MQTTMessageInfo] = deque()
        self.unsent_on_close = 0
        self._client = factory(client_id)
        self._client.max_queued_messages_set(broker.max_queued_results)
        set_reconnect_delays(self._client, broker)
        self._open = False

    def start(self) -> None:
        if self._open:
            return
        connect_in_background(self._client, self._broker, process_tag="result_publisher")
        self._open = True

    def publish(self, *, topic: str, payload: bytes, correlation_id: str, consumer_id: str) -> None:
        self._addressed(topic, consumer_id)
        properties = properties_for(PacketTypes.PUBLISH)
        properties.CorrelationData = correlation_id.encode("utf-8")
        properties.MessageExpiryInterval = self._broker.session_expiry_s
        info = self._client.publish(
            topic, payload, qos=self._broker.qos, retain=False, properties=properties
        )
        if info.rc == mqtt.MQTT_ERR_NO_CONN:
            logger.error(
                f"stage_tag=publish process_tag=result_publisher error_type=no_connection "
                f"topic={topic} recovery_action=held_until_the_connection_returns"
            )
            raise BrokerError(
                f"the broker holds no connection, so the result on {topic} waits for one"
            )
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.error(
                f"stage_tag=publish process_tag=result_publisher error_type=publish_refused "
                f"topic={topic} recovery_action=row_is_already_stored "
                f"detail={mqtt.error_string(info.rc)}"
            )
            raise BrokerError(
                f"the broker refused a result on {topic}: {mqtt.error_string(info.rc)}"
            )
        self._hold(info)

    def flush(self, timeout_s: float) -> int:
        deadline = time.monotonic() + max(timeout_s, 0.0)
        held, self._awaiting = self._awaiting, deque()
        unsent = 0
        for info in held:
            info.wait_for_publish(max(deadline - time.monotonic(), 0.0))
            if not info.is_published():
                unsent += 1
        if unsent:
            logger.error(
                f"stage_tag=flush process_tag=result_publisher error_type=results_not_sent "
                f"results={unsent} recovery_action=rows_are_already_stored"
            )
        return unsent

    def close(self) -> None:
        if not self._open:
            return
        self._open = False
        self.unsent_on_close = self.flush(self._broker.shutdown_flush_s)
        disconnect(self._client)

    def _hold(self, info: mqtt.MQTTMessageInfo) -> None:
        while self._awaiting and self._awaiting[0].is_published():
            self._awaiting.popleft()
        self._awaiting.append(info)

    def _addressed(self, topic: str, consumer_id: str) -> None:
        named, _ = parts_of(topic, prefix=self._topic_prefix)
        if named != consumer_id:
            raise TopicError(f"{consumer_id} is not answered on {topic}, which names {named}")
