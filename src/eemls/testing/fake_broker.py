from __future__ import annotations

from typing import cast

import paho.mqtt.client as mqtt
from paho.mqtt.client import (
    CallbackOnConnect_v2,
    CallbackOnSubscribe_v2,
    MQTTMessage,
    MQTTMessageInfo,
)
from paho.mqtt.enums import CallbackAPIVersion, MQTTErrorCode, MQTTProtocolVersion
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.properties import Properties
from paho.mqtt.reasoncodes import ReasonCode
from paho.mqtt.subscribeoptions import SubscribeOptions

from eemls.config.services import BrokerConfig
from eemls.messaging.broker import properties_for

SUCCESS = MQTTErrorCode.MQTT_ERR_SUCCESS


def broker_settings(**changes: object) -> BrokerConfig:
    return BrokerConfig.model_validate(
        {
            "host": "broker",
            "port": 1883,
            "qos": 1,
            "keepalive_s": 30,
            "session_expiry_s": 300,
            "max_queued_results": 10_000,
            "client_id_prefix": "eemls",
            "instance_topic_prefix": "eemls/instances",
            "reconnect_min_s": 0.5,
            "reconnect_max_s": 8.0,
            "shutdown_flush_s": 8.0,
        }
        | changes
    )


class Published:
    __slots__ = ("correlation_id", "payload", "qos", "retain", "topic")

    def __init__(
        self, *, topic: str, payload: bytes, qos: int, retain: bool, correlation_id: str | None
    ) -> None:
        self.topic = topic
        self.payload = payload
        self.qos = qos
        self.retain = retain
        self.correlation_id = correlation_id


class Opened:
    __slots__ = ("clean_start", "host", "keepalive", "port", "session_expiry_s")

    def __init__(
        self,
        *,
        host: str,
        port: int,
        keepalive: int,
        clean_start: bool,
        session_expiry_s: int | None,
    ) -> None:
        self.host = host
        self.port = port
        self.keepalive = keepalive
        self.clean_start = clean_start
        self.session_expiry_s = session_expiry_s


def _bytes_of(payload: mqtt.PayloadType) -> bytes:
    if payload is None:
        return b""
    if isinstance(payload, str):
        return payload.encode("utf-8")
    if isinstance(payload, bytes | bytearray):
        return bytes(payload)
    return str(payload).encode("utf-8")


class FakeMqttClient(mqtt.Client):
    def __init__(self, client_id: str = "", *, reachable: bool = True, grants: bool = True) -> None:
        super().__init__(
            CallbackAPIVersion.VERSION2,
            client_id=client_id,
            protocol=MQTTProtocolVersion.MQTTv5,
        )
        self.reachable = reachable
        self.grants = grants
        self.opened: list[Opened] = []
        self.async_opens = 0
        self.loops_started = 0
        self.loops_stopped = 0
        self.disconnects = 0
        self.subscriptions: list[tuple[str, int]] = []
        self.publishes: list[Published] = []
        self.confirms = True
        self.infos: list[MQTTMessageInfo] = []
        self.backoff: tuple[float, float] | None = None
        self.queue_bound: int | None = None
        self.will: Published | None = None
        self._awaiting_suback: list[int] = []

    def connect(
        self,
        host: str,
        port: int = 1883,
        keepalive: int = 60,
        bind_address: str = "",
        bind_port: int = 0,
        clean_start: bool | int = mqtt.MQTT_CLEAN_START_FIRST_ONLY,
        properties: Properties | None = None,
    ) -> MQTTErrorCode:
        if not self.reachable:
            raise OSError(f"nothing is listening on {host}:{port}")
        self.opened.append(
            Opened(
                host=host,
                port=port,
                keepalive=keepalive,
                clean_start=bool(clean_start),
                session_expiry_s=getattr(properties, "SessionExpiryInterval", None),
            )
        )
        return SUCCESS

    def connect_async(
        self,
        host: str,
        port: int = 1883,
        keepalive: int = 60,
        bind_address: str = "",
        bind_port: int = 0,
        clean_start: bool | int = mqtt.MQTT_CLEAN_START_FIRST_ONLY,
        properties: Properties | None = None,
    ) -> None:
        self.async_opens += 1

    def loop_start(self) -> MQTTErrorCode:
        self.loops_started += 1
        return SUCCESS

    def loop_stop(self) -> MQTTErrorCode:
        self.loops_stopped += 1
        return SUCCESS

    def disconnect(
        self, reasoncode: ReasonCode | None = None, properties: Properties | None = None
    ) -> MQTTErrorCode:
        self.disconnects += 1
        return SUCCESS

    def subscribe(
        self,
        topic: str
        | tuple[str, int]
        | tuple[str, SubscribeOptions]
        | list[tuple[str, int]]
        | list[tuple[str, SubscribeOptions]],
        qos: int = 0,
        options: SubscribeOptions | None = None,
        properties: Properties | None = None,
    ) -> tuple[MQTTErrorCode, int | None]:
        self.subscriptions.append((str(topic), qos))
        mid = len(self.subscriptions)
        self._awaiting_suback.append(mid)
        return SUCCESS, mid

    def publish(
        self,
        topic: str,
        payload: mqtt.PayloadType = None,
        qos: int = 0,
        retain: bool = False,
        properties: Properties | None = None,
    ) -> MQTTMessageInfo:
        info = MQTTMessageInfo(1)
        if not self.reachable:
            info.rc = mqtt.MQTT_ERR_NO_CONN
            return info
        if self.queue_bound is not None and len(self.publishes) >= self.queue_bound:
            info.rc = mqtt.MQTT_ERR_QUEUE_SIZE
            return info
        held = getattr(properties, "CorrelationData", None)
        self.publishes.append(
            Published(
                topic=topic,
                payload=_bytes_of(payload),
                qos=qos,
                retain=retain,
                correlation_id=None if held is None else bytes(held).decode("utf-8"),
            )
        )
        info.rc = SUCCESS
        self.infos.append(info)
        if self.confirms:
            info._set_as_published()
        return info

    def will_set(
        self,
        topic: str,
        payload: mqtt.PayloadType = None,
        qos: int = 0,
        retain: bool = False,
        properties: Properties | None = None,
    ) -> None:
        self.will = Published(
            topic=topic, payload=_bytes_of(payload), qos=qos, retain=retain, correlation_id=None
        )

    def reconnect_delay_set(self, min_delay: int = 1, max_delay: int = 120) -> None:
        self.backoff = (min_delay, max_delay)

    def max_queued_messages_set(self, queue_size: int) -> mqtt.Client:
        self.queue_bound = queue_size
        return self

    def confirm(self) -> None:
        for info in self.infos:
            info._set_as_published()

    def accept(self) -> None:
        self._fire(ReasonCode(PacketTypes.CONNACK, "Success"))
        self._acknowledge()

    def refuse(self) -> None:
        self._fire(ReasonCode(PacketTypes.CONNACK, "Not authorized"))

    def deliver(self, *, topic: str, payload: bytes, correlation_id: str | bytes | None) -> None:
        message = MQTTMessage(mid=1, topic=topic.encode("utf-8"))
        message.payload = payload
        message.qos = 1
        properties = properties_for(PacketTypes.PUBLISH)
        if correlation_id is not None:
            properties.CorrelationData = (
                correlation_id.encode("utf-8")
                if isinstance(correlation_id, str)
                else correlation_id
            )
        message.properties = properties
        if self.on_message is not None:
            self.on_message(self, None, message)

    def _acknowledge(self) -> None:
        asked, self._awaiting_suback = self._awaiting_suback, []
        if self.on_subscribe is None:
            return
        hook = cast("CallbackOnSubscribe_v2", self.on_subscribe)
        granted = "Granted QoS 1" if self.grants else "Not authorized"
        for mid in asked:
            hook(self, None, mid, [ReasonCode(PacketTypes.SUBACK, granted)], None)

    def _fire(self, reason_code: ReasonCode) -> None:
        if self.on_connect is None:
            return
        hook = cast("CallbackOnConnect_v2", self.on_connect)
        hook(self, None, mqtt.ConnectFlags(session_present=False), reason_code, None)
