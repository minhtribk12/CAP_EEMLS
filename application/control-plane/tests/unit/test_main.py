from __future__ import annotations

import functools
from pathlib import Path

import httpx
import pytest
import yaml

from eemls.domain.cost import DeviceRate
from eemls.domain.types import Condition
from eemls.evaluation_engine.window import Window
from eemls.messaging.topics import TopicError
from eemls.metrics_database.postgres import DatabaseUnavailableError
from eemls.metrics_database.testing import MemoryMetricsDatabase
from eemls.observation_service.operating_conditions import ConditionLookup
from eemls.observation_service.testing import InProcessObservationService
from eemls.orchestration.testing import Answers, Members
from eemls.testing.fake_broker import FakeMqttClient
from eemls.testing.records import profile
from eemls.testing.trees import TREE, committed_tree
from eemls_control_plane import main as control_plane
from eemls_control_plane.config import (
    FILES_READ,
    ControlPlaneView,
    load_control_plane_config,
)
from eemls_control_plane.main import (
    application,
    condition_lookup_of,
    main,
    metrics_reporter_of,
    monitoring_of,
    profile_updates_of,
    publisher_of,
    runtime_context_reader_of,
)
from eemls_control_plane.startup import build_control_plane


class TestWhatItRefusesToStartOn:
    def test_a_missing_database_password_is_refused_by_the_name_of_the_variable(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("EEMLS_DATABASE_PASSWORD", raising=False)

        assert main(["--config", str(TREE), "--seed", "0"]) == 1
        assert "EEMLS_DATABASE_PASSWORD is not set" in capsys.readouterr().err

    def test_the_refusal_never_carries_the_password(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("EEMLS_DATABASE_PASSWORD", "")

        main(["--config", str(TREE), "--seed", "0"])

        assert "hunter2" not in capsys.readouterr().err

    def test_a_configuration_directory_that_is_not_there_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("EEMLS_DATABASE_PASSWORD", "test")

        assert main(["--config", "/nowhere/at/all", "--seed", "0"]) == 1
        assert "refused" in capsys.readouterr().err

    def test_a_database_that_refuses_the_connection_is_refused_with_its_reason(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("EEMLS_DATABASE_PASSWORD", "test")
        monkeypatch.setattr("eemls_control_plane.main.PostgresDatabase.from_config", unreachable)

        assert main(["--config", str(TREE), "--seed", "0"]) == 1
        assert "refused: the metrics database refused the connection" in capsys.readouterr().err

    def test_a_key_of_another_service_file_is_not_its_concern(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        tree = committed_tree(tmp_path / "config")
        path = tree / "services" / "data_hub.yaml"
        path.write_text(
            path.read_text(encoding="utf-8") + "a_key_no_service_knows: 1\n", encoding="utf-8"
        )
        monkeypatch.setenv("EEMLS_DATABASE_PASSWORD", "test")
        monkeypatch.setattr("eemls_control_plane.main.PostgresDatabase.from_config", unreachable)

        assert main(["--config", str(tree), "--seed", "0"]) == 1
        assert "refused: the metrics database refused the connection" in capsys.readouterr().err

    def test_a_scored_cdm_no_table_can_compute_is_refused_before_the_database(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        tree = committed_tree(tmp_path / "config")
        path = tree / "consumers" / "c3_surveillance" / "cdm.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["cdms"]["mean_unicorn_latency_ms"] = {
            "measured": "per_window",
            "window": 500,
            "unit": "milliseconds",
            "description": "Mean of a column nothing stores",
            "specification": {"function": "average", "attribute": "unicorn_latency_ms"},
        }
        path.write_text(yaml.safe_dump(body), encoding="utf-8")
        monkeypatch.setenv("EEMLS_DATABASE_PASSWORD", "test")
        monkeypatch.setattr("eemls_control_plane.main.PostgresDatabase.from_config", unreachable)

        assert main(["--config", str(tree), "--seed", "0"]) == 1
        assert "consumers/c3_surveillance/cdm.yaml" in capsys.readouterr().err


class TestTheSeedItDrawsFrom:
    def seeded(self, monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> object:
        given: dict[str, object] = {}

        def application(config: object, **passed: object) -> None:
            given.update(passed)

        monkeypatch.setenv("EEMLS_DATABASE_PASSWORD", "test")
        monkeypatch.setattr("eemls_control_plane.main.application", application)
        monkeypatch.setattr("eemls_control_plane.main.uvicorn.run", lambda *_, **__: None)
        assert main(["--config", str(TREE), *argv]) == 0
        return given["seed"]

    def test_it_draws_from_the_seed_the_run_sets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EEMLS_SEED", "5")

        assert self.seeded(monkeypatch, []) == 5

    def test_a_seed_on_the_command_line_is_the_one_drawn_from(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self.seeded(monkeypatch, ["--seed", "9"]) == 9

    def test_without_a_seed_it_is_refused_in_one_line(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("EEMLS_SEED", raising=False)
        monkeypatch.setenv("EEMLS_DATABASE_PASSWORD", "test")

        assert main(["--config", str(TREE)]) == 1
        assert capsys.readouterr().err == "refused: EEMLS_SEED is not set and no --seed is given\n"

    def test_an_empty_seed_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EEMLS_SEED", "")
        monkeypatch.setenv("EEMLS_DATABASE_PASSWORD", "test")

        assert main(["--config", str(TREE)]) == 1

    def test_a_seed_that_is_not_a_whole_number_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("EEMLS_SEED", "five")
        monkeypatch.setenv("EEMLS_DATABASE_PASSWORD", "test")

        with pytest.raises(SystemExit):
            main(["--config", str(TREE)])
        assert "invalid int value: 'five'" in capsys.readouterr().err


class TestWhatItBuildsFromTheTree:
    def test_the_measuring_surface_comes_from_the_observation_settings(self) -> None:
        config = load_control_plane_config(TREE)

        held = monitoring_of(config)

        assert held.path == config.services.monitoring.metrics_path

    def test_the_broker_connection_carries_the_committed_settings(self) -> None:
        config = load_control_plane_config(TREE)
        settings = config.services.broker
        client = FakeMqttClient("control-plane-1")

        publisher_of(config, instance_id="control-plane-1", factory=lambda _: client).start()

        assert [(one.host, one.port, one.keepalive) for one in client.opened] == [
            (settings.host, settings.port, settings.keepalive_s)
        ]
        assert client.backoff == (settings.reconnect_min_s, settings.reconnect_max_s)
        assert client.queue_bound == settings.max_queued_results

    def test_a_result_goes_out_at_the_committed_quality_of_service(self) -> None:
        config = load_control_plane_config(TREE)
        client = FakeMqttClient("control-plane-1")
        held = publisher_of(config, instance_id="control-plane-1", factory=lambda _: client)
        held.start()

        held.publish(
            topic="eemls/results/c1_navigation/client-1",
            payload=b"{}",
            correlation_id="corr-1",
            consumer_id="c1_navigation",
        )

        assert [one.qos for one in client.publishes] == [config.services.broker.qos]

    def test_a_result_of_one_consumer_cannot_go_out_on_another_topic(self) -> None:
        config = load_control_plane_config(TREE)
        client = FakeMqttClient("control-plane-1")
        held = publisher_of(config, instance_id="control-plane-1", factory=lambda _: client)
        held.start()

        with pytest.raises(TopicError, match="c1_navigation"):
            held.publish(
                topic="eemls/results/c2_safety/client-1",
                payload=b"{}",
                correlation_id="corr-1",
                consumer_id="c1_navigation",
            )

    def test_it_asks_the_data_hub_the_tree_names_on_a_member_budget(self) -> None:
        config = load_control_plane_config(TREE)
        given: dict[str, object] = {}

        def built_with(**passed: object) -> ConditionLookup:
            given.update(passed)
            return condition_lookup_of(config)

        condition_lookup_of(config, factory=built_with)

        assert given["base_url"] == str(config.services.control_plane.data_hub_url)
        assert given["timeout_ms"] == float(config.services.control_plane.member_timeout_ms)

    def test_the_periods_it_can_name_are_the_dominant_classes_of_the_dataset(self) -> None:
        config = load_control_plane_config(TREE)
        named = config.datasets.dominant_classes[2]
        held = runtime_context_reader_of(config)

        held.read_from("c1_navigation", Window(({"predicted_label": named, "condition": "night"},)))

        assert held("c1_navigation").period_name == named
        assert held("c1_navigation").period_index == 3
        assert held("c1_navigation").condition == "night"

    def test_nothing_has_been_served_so_no_period_is_named_yet(self) -> None:
        held = runtime_context_reader_of(load_control_plane_config(TREE))("c1_navigation")

        assert (held.period_name, held.period_index) == ("unnamed", 0)
        assert (held.condition, held.weather) == ("day", "clear")


def unreachable(*_: object, **__: object) -> None:
    raise DatabaseUnavailableError("the metrics database refused the connection: no one listens")


def unhurried_tree(target: Path) -> Path:
    tree = committed_tree(target)
    database = tree / "services" / "metrics_database.yaml"
    body = yaml.safe_load(database.read_text(encoding="utf-8"))
    body["flush_interval_ms"] = 3_600_000
    database.write_text(yaml.safe_dump(body), encoding="utf-8")
    return tree


def profiled(config: ControlPlaneView) -> MemoryMetricsDatabase:
    database = MemoryMetricsDatabase()
    for index, (name, service) in enumerate(
        sorted(config.models.inference_services.services.items())
    ):
        database.write_profile(
            profile(
                service_name=name,
                model=service.model,
                device=service.device_type,
                instance_id=f"i-{index}",
            )
        )
    return database


def observation_service(config: ControlPlaneView, run_id: str) -> InProcessObservationService:
    rates: dict[str, DeviceRate] = dict(config.cost.rates())
    return InProcessObservationService(
        MemoryMetricsDatabase(),
        run_id=run_id,
        rates=rates,
        window_s=config.services.observation_service.profile_window_s,
        min_labeled_per_class=config.services.observation_service.profile_min_labeled_per_class,
        min_labeled_overall=config.services.observation_service.profile_min_labeled_overall,
    )


class Unhurried:
    async def __call__(self, image_id: str) -> Condition:
        return "day"

    async def close(self) -> None: ...


class Started:
    def __init__(
        self,
        config: ControlPlaneView,
        monkeypatch: pytest.MonkeyPatch,
        *,
        run_id: str = "r1",
        service_run_id: str | None = None,
    ) -> None:
        self.config = config
        self.stored = profiled(config)
        self.service = observation_service(config, service_run_id or run_id)
        self.closed_when_sent: list[bool] = []
        self.clients: list[str] = []
        self.client = FakeMqttClient()
        pool = config.models.inference_services.services
        members = Members({name: Answers(label="car") for name in pool})
        monkeypatch.setattr(
            "eemls_control_plane.main.PostgresDatabase.from_config",
            lambda *_, **__: self.stored,
        )
        monkeypatch.setattr(control_plane, "condition_lookup_of", lambda *_, **__: Unhurried())
        monkeypatch.setattr(
            control_plane,
            "publisher_of",
            functools.partial(publisher_of, factory=self.opened),
        )
        monkeypatch.setattr(
            control_plane,
            "build_control_plane",
            functools.partial(build_control_plane, transport=members),
        )
        transport = httpx.MockTransport(self.sent)
        monkeypatch.setattr(
            control_plane,
            "metrics_reporter_of",
            functools.partial(metrics_reporter_of, transport=transport),
        )
        monkeypatch.setattr(
            control_plane,
            "profile_updates_of",
            functools.partial(profile_updates_of, transport=transport),
        )
        self.app = application(
            config, instance_id="control-plane-1", run_id=run_id, password="", seed=0
        )

    def opened(self, client_id: str) -> FakeMqttClient:
        self.clients.append(client_id)
        return self.client

    def sent(self, request: httpx.Request) -> httpx.Response:
        self.closed_when_sent.append(self.stored.closed)
        return self.service.handle_request(request)

    async def classify(self) -> None:
        consumer_id = sorted(self.config.consumers)[0]
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://control-plane"
        ) as client:
            answer = await client.post(
                "/classify",
                json={
                    "consumer_id": consumer_id,
                    "image_id": "img-7",
                    "reply_topic": f"eemls/results/{consumer_id}/client-1",
                    "correlation_id": "corr-1",
                },
            )
        assert answer.status_code == 202


class TestWhereItsRowsGo:
    def test_it_reads_where_the_observation_service_listens(self) -> None:
        config = load_control_plane_config(TREE)

        assert "services/observation_service.yaml" in FILES_READ
        assert str(config.services.observation_service.url) == "http://observation-service:8083/"

    async def test_the_rows_of_a_query_reach_the_observation_service_on_shutdown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        started = Started(
            load_control_plane_config(unhurried_tree(tmp_path / "config")), monkeypatch
        )

        async with started.app.router.lifespan_context(started.app):
            await started.classify()
            held = len(started.service.database.read_rows("queries"))

        assert held == 0
        assert len(started.service.database.read_rows("queries")) == 1
        assert started.service.database.read_rows("member_predictions")
        assert started.stored.read_rows("queries") == []

    async def test_the_metrics_reporter_is_flushed_before_the_database_closes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        started = Started(
            load_control_plane_config(unhurried_tree(tmp_path / "config")), monkeypatch
        )

        async with started.app.router.lifespan_context(started.app):
            await started.classify()

        assert started.closed_when_sent == [False]
        assert started.stored.closed

    def test_it_claims_the_run_in_the_database_itself(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        started = Started(
            load_control_plane_config(unhurried_tree(tmp_path / "config")), monkeypatch
        )

        assert started.stored.owner is not None
        assert started.stored.owner.instance_id == "control-plane-1"

    async def test_it_does_not_listen_for_the_running_instances(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = load_control_plane_config(unhurried_tree(tmp_path / "config"))
        started = Started(config, monkeypatch)

        async with started.app.router.lifespan_context(started.app):
            started.client.accept()

        prefix = config.services.broker.client_id_prefix
        assert started.clients == [f"{prefix}-control-plane-1"]
        assert not [
            topic
            for topic, _ in started.client.subscriptions
            if topic.startswith("eemls/instances")
        ]
