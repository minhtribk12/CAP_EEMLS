from __future__ import annotations

import logging
from collections.abc import Callable

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion, MQTTProtocolVersion
from paho.mqtt.properties import Properties

from eemls.config.services import BrokerConfig
from eemls.domain.errors import EemlsError

logger = logging.getLogger(__name__)

Factory = Callable[[str], mqtt.Client]


class BrokerError(EemlsError): ...


def properties_for(packet_type: int) -> Properties:
    return Properties(packet_type)  # type: ignore[no-untyped-call]


def close_wakeup_sockets(client: mqtt.Client) -> None:
    client._reset_sockets(sockpair_only=True)


def client_of(client_id: str) -> mqtt.Client:
    return mqtt.Client(
        CallbackAPIVersion.VERSION2,
        client_id=client_id,
        protocol=MQTTProtocolVersion.MQTTv5,
    )


def set_reconnect_delays(client: mqtt.Client, broker: BrokerConfig) -> None:
    client.reconnect_delay_set(
        min_delay=broker.reconnect_min_s,  # type: ignore[arg-type]
        max_delay=broker.reconnect_max_s,  # type: ignore[arg-type]
    )


def connect_in_background(
    client: mqtt.Client,
    broker: BrokerConfig,
    *,
    process_tag: str,
    clean_start: mqtt.CleanStartOption = mqtt.MQTT_CLEAN_START_FIRST_ONLY,
    properties: Properties | None = None,
) -> None:
    try:
        client.connect(
            broker.host,
            broker.port,
            keepalive=broker.keepalive_s,
            clean_start=clean_start,
            properties=properties,
        )
    except OSError as refused:
        logger.error(
            f"stage_tag=start process_tag={process_tag} error_type=broker_unreachable "
            f"host={broker.host} port={broker.port} recovery_action=retry_in_background "
            f"detail={refused}"
        )
        client.connect_async(
            broker.host,
            broker.port,
            keepalive=broker.keepalive_s,
            clean_start=clean_start,
            properties=properties,
        )
    started = client.loop_start()
    if started != mqtt.MQTT_ERR_SUCCESS:
        logger.error(
            f"stage_tag=start process_tag={process_tag} error_type=loop_not_started "
            f"recovery_action=none detail={mqtt.error_string(started)}"
        )


def disconnect(client: mqtt.Client) -> None:
    client.disconnect()
    client.loop_stop()
    close_wakeup_sockets(client)
