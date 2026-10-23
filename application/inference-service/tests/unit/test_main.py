from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eemls.config.reading import ConfigError
from eemls.domain.records import Instance
from eemls.monitoring.metrics import Monitoring
from eemls.testing.fake_broker import FakeMqttClient
from eemls.testing.records import AT
from eemls.testing.trees import committed_tree
from eemls_inference_service.config import load_inference_config
from eemls_inference_service.main import (
    announcer_of,
    inference_endpoints_of,
    instance_of,
    main,
    monitoring_of,
    providers_for,
)

TREE = Path(__file__).resolve().parents[4] / "config"


def asked(**changes: object) -> dict[str, object]:
    config = load_inference_config(TREE)
    fields: dict[str, object] = {
        "config": config,
        "service_name": "infer-b5-orin",
        "instance_id": "i-1",
        "node": "orin-1",
        "replica": 1,
        "models": Path("models"),
        "monitoring": Monitoring.from_config(config.services.monitoring),
    }
    return fields | changes


class TestWhatItRefusesBeforeServing:
    def test_a_service_name_the_tree_does_not_declare_is_refused_by_name(self) -> None:
        with pytest.raises(ConfigError, match="infer-b5-orin"):
            inference_endpoints_of(**asked())  # type: ignore[arg-type]

    def test_the_refusal_names_what_is_declared_instead(self) -> None:
        with pytest.raises(ConfigError, match="only"):
            inference_endpoints_of(**asked(service_name="anything-at-all"))  # type: ignore[arg-type]

    def test_a_tree_it_cannot_load_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--config", str(tmp_path), "--service", "any"]) == 1
        assert "refused" in capsys.readouterr().err

    def test_a_copy_told_nothing_about_itself_is_refused(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--config", str(TREE), "--service", ""]) == 1
        assert "declares no" in capsys.readouterr().err


CPU_ONLY = ("CPUExecutionProvider",)
GPU_FIRST = ("CUDAExecutionProvider", "CPUExecutionProvider")


class TestTheProvidersItAsksFor:
    @pytest.mark.parametrize(
        ("service_name", "chain"),
        [
            ("infer-b1-pedestrian-bt3", CPU_ONLY),
            ("infer-b3-motorcycle-pi4", CPU_ONLY),
            ("infer-b8-general-agx-orin", GPU_FIRST),
            ("infer-b7-general-p5820", GPU_FIRST),
        ],
    )
    def test_a_service_gets_the_chain_its_device_can_run(
        self, service_name: str, chain: tuple[str, ...]
    ) -> None:
        assert providers_for(load_inference_config(TREE), service_name) == chain

    @pytest.mark.parametrize(
        ("service_name", "chain"),
        [
            ("infer-b1-pedestrian-bt3", CPU_ONLY),
            ("infer-b8-general-agx-orin", GPU_FIRST),
        ],
    )
    def test_a_service_of_the_pool_loads_its_model_on_that_chain(
        self, service_name: str, chain: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked_for: list[tuple[str, ...]] = []

        def loaded(*_: object, providers: tuple[str, ...], **__: object) -> object:
            asked_for.append(tuple(providers))
            return object()

        monkeypatch.setattr("eemls_inference_service.main.load", loaded)

        serving = inference_endpoints_of(**asked(service_name=service_name))  # type: ignore[arg-type]

        assert asked_for == [chain]
        services = load_inference_config(TREE).models.inference_services.services
        assert serving.device == services[service_name].device_type


class TestWhatItBuildsFromTheTree:
    def test_the_measuring_surface_is_the_one_the_tree_describes(self) -> None:
        config = load_inference_config(TREE)

        held = monitoring_of(config)

        assert held.path == config.services.monitoring.metrics_path
        assert held.enabled is config.services.monitoring.enabled


def edited(tree: Path, name: str, key: str, value: object) -> None:
    path = tree / name
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body[key] = value
    path.write_text(yaml.safe_dump(body), encoding="utf-8")


class TestHowACopyAnnouncesItself:
    def copy(self, tree: Path = TREE) -> Instance:
        return instance_of(
            load_inference_config(tree),
            service_name="infer-b1-general-pi4",
            instance_id="infer-b1-general-pi4-abc",
            node="pi-1",
            replica=1,
            provider="CPUExecutionProvider",
            image="docker.io/someone/eemls-inference-service:cpu-arm64-latest",
            started_at=AT,
        )

    def test_its_record_names_what_it_runs_from_the_tree(self) -> None:
        held = self.copy()
        config = load_inference_config(TREE)

        assert (held.model, held.device) == ("b1_general", "raspberry_pi4")
        assert held.number_format == config.services.inference_service.number_format
        assert held.image_tag == "docker.io/someone/eemls-inference-service:cpu-arm64-latest"
        assert held.runtime_version
        assert held.started_at == AT
        assert held.stopped_at is None

    def test_the_same_tree_gives_the_same_settings_digest(self) -> None:
        assert self.copy().config_hash == self.copy().config_hash

    def test_it_announces_on_the_topic_of_its_run_and_its_name(self) -> None:
        client = FakeMqttClient()

        config = load_inference_config(TREE)
        announcer_of(config, self.copy(), run_id="r1", factory=lambda _: client)

        assert client.will is not None
        assert client.will.topic == "eemls/instances/r1/infer-b1-general-pi4-abc"

    def test_a_control_plane_edit_leaves_the_settings_digest(self, tmp_path: Path) -> None:
        before = committed_tree(tmp_path / "before")
        tree = committed_tree(tmp_path / "config")
        edited(tree, "services/control_plane.yaml", "member_timeout_ms", 900)

        assert self.copy(tree).config_hash == self.copy(before).config_hash

    def test_an_edit_of_its_own_settings_changes_the_digest(self, tmp_path: Path) -> None:
        before = committed_tree(tmp_path / "before")
        tree = committed_tree(tmp_path / "config")
        edited(tree, "services/inference_service.yaml", "warmup_requests", 9)

        assert self.copy(tree).config_hash != self.copy(before).config_hash
