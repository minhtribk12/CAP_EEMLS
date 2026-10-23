from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import paho.mqtt.client as mqtt
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.properties import Properties
from paho.mqtt.reasoncodes import ReasonCode
from pydantic import ValidationError

from eemls.config.services import BrokerConfig
from eemls.domain.types import FrozenModel, Identifier
from eemls.messaging.broker import (
    Factory,
    client_of,
    connect_in_background,
    disconnect,
    properties_for,
    set_reconnect_delays,
)

logger = logging.getLogger(__name__)


class Delivered(FrozenModel):
    topic: Identifier
    correlation_id: Identifier
    payload: bytes


Taking = Callable[[Delivered], None]


class ResultListener:
    def __init__(
        self,
        *,
        broker: BrokerConfig,
        client_id: str,
        topic: str,
        on_result: Taking,
        factory: Factory = client_of,
    ) -> None:
        self._broker = broker
        self._topic = topic
        self._on_result = on_result
        self.unaddressed = 0
        self.refused = 0
        self._ready = threading.Event()
        self._open = False
        self._awaited_subscription: int | None = None
        self._client = factory(client_id)
        set_reconnect_delays(self._client, broker)
        self._client.on_connect = self._connected
        self._client.on_disconnect = self._disconnected
        self._client.on_subscribe = self._subscribed
        self._client.on_message = self._message

    def start(self) -> None:
        if self._open:
            return
        properties = properties_for(PacketTypes.CONNECT)
        properties.SessionExpiryInterval = self._broker.session_expiry_s
        connect_in_background(
            self._client,
            self._broker,
            process_tag="result_listener",
            clean_start=False,
            properties=properties,
        )
        self._open = True

    def wait_until_ready(self, timeout_s: float) -> bool:
        return self._ready.wait(timeout_s)

    def close(self) -> None:
        if not self._open:
            return
        self._open = False
        self._ready.clear()
        disconnect(self._client)

    def _connected(
        self,
        client: mqtt.Client,
        userdata: object,
        flags: mqtt.ConnectFlags,
        reason_code: ReasonCode,
        properties: Properties | None,
    ) -> None:
        if reason_code.is_failure:
            logger.error(
                f"stage_tag=connect process_tag=result_listener error_type=connection_refused "
                f"topic={self._topic} recovery_action=retry_in_background detail={reason_code}"
            )
            return
        try:
            asked, mid = client.subscribe(self._topic, qos=self._broker.qos)
        except (ValueError, OSError) as refused:
            logger.error(
                f"stage_tag=subscribe process_tag=result_listener error_type=subscribe_failed "
                f"topic={self._topic} recovery_action=none detail={refused}"
            )
            return
        if asked != mqtt.MQTT_ERR_SUCCESS:
            logger.error(
                f"stage_tag=subscribe process_tag=result_listener error_type=subscribe_refused "
                f"topic={self._topic} recovery_action=retry_on_reconnect "
                f"detail={mqtt.error_string(asked)}"
            )
            return
        self._awaited_subscription = mid

    def _subscribed(
        self,
        client: mqtt.Client,
        userdata: object,
        mid: int,
        reason_codes: list[ReasonCode],
        properties: Properties | None,
    ) -> None:
        if mid != self._awaited_subscription:
            return
        turned_down = [one for one in reason_codes if one.is_failure]
        if turned_down:
            logger.error(
                f"stage_tag=subscribe process_tag=result_listener "
                f"error_type=subscription_not_granted topic={self._topic} "
                f"recovery_action=none detail={turned_down}"
            )
            return
        self._ready.set()

    def _disconnected(
        self,
        client: mqtt.Client,
        userdata: object,
        flags: mqtt.DisconnectFlags,
        reason_code: ReasonCode,
        properties: Properties | None,
    ) -> None:
        self._ready.clear()

    def _message(self, client: mqtt.Client, userdata: object, message: mqtt.MQTTMessage) -> None:
        try:
            delivered = self._read(message)
        except (UnicodeDecodeError, ValidationError, ValueError) as refused:
            self._unreadable(message, refused)
            return
        if delivered is None:
            return
        try:
            self._on_result(delivered)
        except Exception as refused:
            self.refused += 1
            logger.error(
                f"stage_tag=deliver process_tag=result_listener error_type=result_not_taken "
                f"topic={delivered.topic} recovery_action=dropped detail={refused}"
            )

    def _read(self, message: mqtt.MQTTMessage) -> Delivered | None:
        held = getattr(message.properties, "CorrelationData", None)
        if held is None:
            self._unaddressed(message.topic, "no_correlation_id")
            return None
        return Delivered(
            topic=message.topic,
            correlation_id=bytes(held).decode("utf-8"),
            payload=bytes(message.payload),
        )

    def _unreadable(self, message: mqtt.MQTTMessage, refused: Exception) -> None:
        self._unaddressed(message.topic, "unreadable_correlation_id", detail=str(refused))

    def _unaddressed(self, topic: object, error_type: str, *, detail: str = "") -> None:
        self.unaddressed += 1
        logger.error(
            f"stage_tag=deliver process_tag=result_listener error_type={error_type} "
            f"topic={topic} recovery_action=dropped" + (f" detail={detail}" if detail else "")
        )
