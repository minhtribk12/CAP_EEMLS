from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

import paho.mqtt.client as mqtt
from paho.mqtt.client import MQTTMessage
from paho.mqtt.properties import Properties
from paho.mqtt.reasoncodes import ReasonCode
from pydantic import ValidationError

from eemls.config.services import BrokerConfig
from eemls.domain.records import Instance
from eemls.domain.types import FrozenModel
from eemls.messaging.broker import (
    Factory,
    client_of,
    connect_in_background,
    disconnect,
    set_reconnect_delays,
)

logger = logging.getLogger(__name__)

STOP_WAIT_S = 2.0


class InstanceAnnouncement(FrozenModel):
    state: Literal["started", "stopped", "lost"]
    instance: Instance


def _now() -> datetime:
    return datetime.now(UTC)


def _announcement_payload(
    state: Literal["started", "stopped", "lost"], instance: Instance
) -> bytes:
    return InstanceAnnouncement(state=state, instance=instance).model_dump_json().encode("utf-8")


class Announcer:
    def __init__(
        self,
        *,
        broker: BrokerConfig,
        client_id: str,
        topic: str,
        instance: Instance,
        clock: Callable[[], datetime] = _now,
        factory: Factory = client_of,
    ) -> None:
        self._broker = broker
        self._topic = topic
        self._instance = instance
        self._clock = clock
        self._open = False
        self._client = factory(client_id)
        set_reconnect_delays(self._client, broker)
        self._client.will_set(
            topic, _announcement_payload("lost", instance), qos=broker.qos, retain=True
        )
        self._client.on_connect = self._connected

    def start(self) -> None:
        if self._open:
            return
        connect_in_background(self._client, self._broker, process_tag="announcer")
        self._open = True

    def stop(self) -> None:
        if not self._open:
            return
        self._open = False
        stopped = self._instance.model_copy(update={"stopped_at": self._clock()})
        info = self._client.publish(
            self._topic,
            _announcement_payload("stopped", stopped),
            qos=self._broker.qos,
            retain=True,
        )
        try:
            info.wait_for_publish(STOP_WAIT_S)
        except RuntimeError as unsent:
            logger.warning(
                f"stage_tag=stop process_tag=announcer error_type=not_connected "
                f"topic={self._topic} recovery_action=shut detail={unsent}"
            )
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
                f"stage_tag=connect process_tag=announcer error_type=connection_refused "
                f"topic={self._topic} recovery_action=retry_in_background detail={reason_code}"
            )
            return
        client.publish(
            self._topic,
            _announcement_payload("started", self._instance),
            qos=self._broker.qos,
            retain=True,
        )


class InstanceAnnouncementListener:
    def __init__(
        self,
        *,
        broker: BrokerConfig,
        client_id: str,
        topic_filter: str,
        on_announcement: Callable[[InstanceAnnouncement], None],
        factory: Factory = client_of,
    ) -> None:
        self._broker = broker
        self._topic_filter = topic_filter
        self._on_announcement = on_announcement
        self._open = False
        self.refused = 0
        self._client = factory(client_id)
        set_reconnect_delays(self._client, broker)
        self._client.on_connect = self._connected
        self._client.on_message = self._message

    def start(self) -> None:
        if self._open:
            return
        connect_in_background(self._client, self._broker, process_tag="presence_listener")
        self._open = True

    def close(self) -> None:
        if not self._open:
            return
        self._open = False
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
                f"stage_tag=connect process_tag=presence_listener error_type=connection_refused "
                f"topic={self._topic_filter} recovery_action=retry_in_background "
                f"detail={reason_code}"
            )
            return
        client.subscribe(self._topic_filter, qos=self._broker.qos)

    def _message(self, client: mqtt.Client, userdata: object, message: MQTTMessage) -> None:
        try:
            announcement = InstanceAnnouncement.model_validate_json(message.payload)
        except ValidationError as refused:
            self.refused += 1
            logger.error(
                f"stage_tag=receive process_tag=presence_listener error_type=not_a_presence "
                f"topic={message.topic} recovery_action=dropped detail={refused.errors()[:1]}"
            )
            return
        self._on_announcement(announcement)
