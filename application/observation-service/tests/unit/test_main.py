from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import httpx
import pytest

from eemls.messaging.instance_announcements import InstanceAnnouncement
from eemls.messaging.topics import instance_topic
from eemls.metrics_database.postgres import DatabaseUnavailableError
from eemls.metrics_database.testing import MemoryMetricsDatabase
from eemls.testing.fake_broker import FakeMqttClient
from eemls.testing.records import AT, instance, served
from eemls_observation_service.config import load_observation_service_config
from eemls_observation_service.main import (
    announcement_listener_of,
    application,
    main,
    profile_settings_of,
)

GIVEN = "test"


def unreachable(*_: object, **__: object) -> MemoryMetricsDatabase:
    raise DatabaseUnavailableError("the metrics database refused the connection: no route")


class TestWhatItRefusesToStartOn:
    def test_a_missing_database_password_is_refused(
        self, tree: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("EEMLS_DATABASE_PASSWORD", raising=False)

        assert main(["--config", str(tree), "--run", "run-1"]) == 1
        assert "EEMLS_DATABASE_PASSWORD is not set" in capsys.readouterr().err

    def test_a_configuration_directory_that_is_not_there_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("EEMLS_DATABASE_PASSWORD", "test")

        assert main(["--config", "/nowhere/at/all", "--run", "run-1"]) == 1
        assert "refused" in capsys.readouterr().err

    def test_a_database_that_refuses_the_connection_is_refused_with_its_reason(
        self, tree: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("EEMLS_DATABASE_PASSWORD", "test")
        monkeypatch.setattr(
            "eemls_observation_service.main.PostgresDatabase.from_config", unreachable
        )

        assert main(["--config", str(tree), "--run", "run-1"]) == 1
        assert "refused: the metrics database refused the connection" in capsys.readouterr().err


class TestRecordingTheRunningInstances:
    def test_an_announced_instance_is_recorded(self, tree: Path) -> None:
        config = load_observation_service_config(tree)
        settings = config.services.broker
        client = FakeMqttClient("observation-service-1")
        database = MemoryMetricsDatabase()
        listener = announcement_listener_of(
            config,
            instance_id="observation-service-1",
            run_id="run-1",
            database=database,
            factory=lambda _: client,
        )
        listener.start()
        client.accept()

        client.deliver(
            topic=instance_topic(
                prefix=settings.instance_topic_prefix, run_id="run-1", instance_id="i-1"
            ),
            payload=InstanceAnnouncement(state="started", instance=instance())
            .model_dump_json()
            .encode(),
            correlation_id=None,
        )

        assert client.subscriptions == [(f"{settings.instance_topic_prefix}/run-1/+", 1)]
        assert list(database.instances) == ["i-1"]

    def test_a_lost_instance_is_stopped(self, tree: Path) -> None:
        config = load_observation_service_config(tree)
        settings = config.services.broker
        client = FakeMqttClient("observation-service-1")
        database = MemoryMetricsDatabase()
        announcement_listener_of(
            config,
            instance_id="observation-service-1",
            run_id="run-1",
            database=database,
            factory=lambda _: client,
        ).start()
        client.accept()

        client.deliver(
            topic=instance_topic(
                prefix=settings.instance_topic_prefix, run_id="run-1", instance_id="i-1"
            ),
            payload=InstanceAnnouncement(state="lost", instance=instance())
            .model_dump_json()
            .encode(),
            correlation_id=None,
        )

        assert database.instances["i-1"].stopped_at is not None


class TestTheProfileSettings:
    def test_the_window_ends_at_the_asked_moment_and_spans_the_profile_window(
        self, tree: Path
    ) -> None:
        config = load_observation_service_config(tree)
        keys = config.services.observation_service

        settings = profile_settings_of(config)(AT)

        assert settings.window_end == AT
        assert settings.window_start == AT - timedelta(seconds=keys.profile_window_s)
        assert settings.min_labeled_per_class == keys.profile_min_labeled_per_class
        assert settings.min_labeled_overall == keys.profile_min_labeled_overall
        assert set(settings.rates) == set(config.cost.devices)


class TestTheApplication:
    async def test_it_stores_rows_in_the_database_it_opened(
        self, tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        database = MemoryMetricsDatabase("run-1")
        monkeypatch.setattr(
            "eemls_observation_service.main.PostgresDatabase.from_config",
            lambda *_, **__: database,
        )
        app = application(
            load_observation_service_config(tree),
            instance_id="observation-service-1",
            run_id="run-1",
            password=GIVEN,
            factory=lambda name: FakeMqttClient(name),
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://observation-service"
        ) as client:
            answer = await client.post(
                "/rows",
                json={
                    "run_id": "run-1",
                    "sender": "control-plane-1",
                    "queries": [served().model_dump(mode="json")],
                },
            )

        assert answer.status_code == 200
        assert database.applied == 1
        assert list(database.queries) == ["q-1"]
