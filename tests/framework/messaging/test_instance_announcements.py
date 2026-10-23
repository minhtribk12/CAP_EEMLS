from __future__ import annotations

import json
import logging
from datetime import timedelta

import pytest

from eemls.messaging.instance_announcements import (
    Announcer,
    InstanceAnnouncement,
    InstanceAnnouncementListener,
)
from eemls.metrics_database.testing import MemoryMetricsDatabase
from eemls.observation_service.running_instances import instance_recorder
from eemls.testing.fake_broker import FakeMqttClient, broker_settings
from eemls.testing.records import AT, instance

TOPIC = "eemls/instances/r1/i-1"
LATER = AT + timedelta(minutes=5)


def announcer(client: FakeMqttClient) -> Announcer:
    return Announcer(
        broker=broker_settings(),
        client_id="eemls-i-1",
        topic=TOPIC,
        instance=instance(),
        clock=lambda: LATER,
        factory=lambda _: client,
    )


def said(payload: bytes) -> InstanceAnnouncement:
    return InstanceAnnouncement.model_validate_json(payload)


class TestACopyAnnouncingItself:
    def test_it_says_it_started_once_the_broker_takes_it_and_the_broker_keeps_it(self) -> None:
        client = FakeMqttClient()
        announcer(client).start()
        client.accept()

        [published] = client.publishes

        assert published.topic == TOPIC
        assert published.retain
        assert said(published.payload) == InstanceAnnouncement(state="started", instance=instance())

    def test_the_broker_holds_a_word_for_it_in_case_it_goes_without_one(self) -> None:
        client = FakeMqttClient()
        announcer(client).start()

        assert client.will is not None
        assert client.will.topic == TOPIC
        assert client.will.retain
        assert said(client.will.payload) == InstanceAnnouncement(state="lost", instance=instance())

    def test_stopping_says_when_it_stopped_and_then_lets_the_connection_go(self) -> None:
        client = FakeMqttClient()
        held = announcer(client)
        held.start()
        client.accept()

        held.stop()

        stopped = said(client.publishes[-1].payload)
        assert stopped.state == "stopped"
        assert stopped.instance.stopped_at == LATER
        assert client.publishes[-1].retain
        assert client.disconnects == 1
        assert client.loops_stopped == 1

    def test_stopping_after_the_broker_has_gone_logs_once_and_still_lets_the_connection_go(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = FakeMqttClient()
        held = announcer(client)
        held.start()
        client.accept()
        client.reachable = False

        held.stop()

        [logged] = [record for record in caplog.records if record.levelno >= logging.WARNING]
        assert "process_tag=announcer" in logged.getMessage()
        assert client.disconnects == 1
        assert client.loops_stopped == 1

    def test_a_broker_that_is_not_there_is_tried_again_in_the_background(self) -> None:
        client = FakeMqttClient(reachable=False)
        announcer(client).start()

        assert client.async_opens == 1


def listener(
    client: FakeMqttClient, heard: list[InstanceAnnouncement]
) -> InstanceAnnouncementListener:
    return InstanceAnnouncementListener(
        broker=broker_settings(),
        client_id="eemls-control-plane-1-instances",
        topic_filter="eemls/instances/r1/+",
        on_announcement=heard.append,
        factory=lambda _: client,
    )


class TestTheControlPlaneHearingThem:
    def test_it_listens_to_every_copy_of_its_run(self) -> None:
        client = FakeMqttClient()
        listener(client, []).start()
        client.accept()

        assert client.subscriptions == [("eemls/instances/r1/+", 1)]

    def test_each_word_of_a_copy_is_handed_on(self) -> None:
        client = FakeMqttClient()
        heard: list[InstanceAnnouncement] = []
        listener(client, heard).start()
        client.accept()
        word = InstanceAnnouncement(state="started", instance=instance())

        client.deliver(topic=TOPIC, payload=word.model_dump_json().encode(), correlation_id=None)

        assert heard == [word]

    def test_a_word_that_is_not_one_is_counted_and_dropped(self) -> None:
        client = FakeMqttClient()
        heard: list[InstanceAnnouncement] = []
        held = listener(client, heard)
        held.start()
        client.accept()

        client.deliver(
            topic=TOPIC, payload=json.dumps({"state": "x"}).encode(), correlation_id=None
        )

        assert heard == []
        assert held.refused == 1


class TestTheRowsItWrites:
    def test_a_copy_that_started_has_a_row_with_no_stop(self) -> None:
        database = MemoryMetricsDatabase()

        instance_recorder(database, clock=lambda: LATER)(
            InstanceAnnouncement(state="started", instance=instance())
        )

        assert database.instances["i-1"].stopped_at is None

    def test_a_copy_that_stopped_has_the_stop_it_said(self) -> None:
        database = MemoryMetricsDatabase()
        record = instance_recorder(database, clock=lambda: LATER + timedelta(hours=1))
        record(InstanceAnnouncement(state="started", instance=instance()))

        record(InstanceAnnouncement(state="stopped", instance=instance(stopped_at=LATER)))

        assert database.instances["i-1"].stopped_at == LATER

    def test_a_copy_gone_without_a_word_stops_when_the_broker_said_so(self) -> None:
        database = MemoryMetricsDatabase()
        record = instance_recorder(database, clock=lambda: LATER)
        record(InstanceAnnouncement(state="started", instance=instance()))

        record(InstanceAnnouncement(state="lost", instance=instance()))

        assert database.instances["i-1"].stopped_at == LATER

    def test_a_stop_heard_before_the_start_still_leaves_one_row(self) -> None:
        database = MemoryMetricsDatabase()

        instance_recorder(database, clock=lambda: LATER)(
            InstanceAnnouncement(state="stopped", instance=instance(stopped_at=LATER))
        )

        assert database.instances["i-1"].stopped_at == LATER
