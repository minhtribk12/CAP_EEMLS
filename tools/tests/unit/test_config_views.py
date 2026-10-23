from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from eemls_consumer.config import load_consumer_config
from eemls_control_plane.config import load_control_plane_config
from eemls_data_hub.config import load_data_hub_config
from eemls_inference_service.config import load_inference_config
from eemls_observation_service.config import load_observation_service_config
from pydantic import BaseModel

from eemls.testing.trees import committed_tree
from eemls_tools.config_tree import FILES_READ_BY, ConfigError, load_config
from eemls_tools.services import load_images_config
from support.repository import ROOT

TREE = ROOT / "config"

Loader = Callable[[Path], BaseModel]
LOADERS: dict[str, Loader] = {
    "inference": load_inference_config,
    "data_hub": load_data_hub_config,
    "consumer": load_consumer_config,
    "control_plane": load_control_plane_config,
    "observation_service": load_observation_service_config,
}
OWN_FILE = {
    "inference": "services/inference_service.yaml",
    "data_hub": "services/data_hub.yaml",
    "consumer": "services/consumer.yaml",
    "control_plane": "services/control_plane.yaml",
    "observation_service": "services/observation_service.yaml",
}
ANOTHER_SERVICE_FILE = {
    "inference": "services/control_plane.yaml",
    "data_hub": "services/control_plane.yaml",
    "consumer": "services/control_plane.yaml",
    "control_plane": "services/data_hub.yaml",
    "observation_service": "services/data_hub.yaml",
}
FOREIGN_FILES = [
    ("inference", "cost/prices.yaml"),
    ("control_plane", "services/inference_service.yaml"),
    ("control_plane", "datasets/sources.yaml"),
]


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    return committed_tree(tmp_path / "config")


def unknown_key_in(tree: Path, name: str) -> None:
    path = tree / name
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body["a_key_no_service_knows"] = 1
    path.write_text(yaml.safe_dump(body), encoding="utf-8")


def only_the_files_of(service: str, target: Path) -> Path:
    for name in FILES_READ_BY[service]:
        source = TREE / name
        if source.is_dir():
            shutil.copytree(source, target / name)
        else:
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target / name)
    return target


@pytest.mark.parametrize("service", LOADERS)
class TestEachServiceChecksOnlyItsOwnFiles:
    def test_the_committed_tree_loads(self, service: str) -> None:
        LOADERS[service](TREE)

    def test_a_tree_of_only_the_files_it_reads_loads(self, service: str, tmp_path: Path) -> None:
        LOADERS[service](only_the_files_of(service, tmp_path / "config"))

    def test_every_file_it_reads_is_in_the_committed_tree(self, service: str) -> None:
        missing = [name for name in FILES_READ_BY[service] if not (TREE / name).exists()]

        assert not missing

    def test_an_unknown_key_in_another_service_file_is_not_its_concern(
        self, service: str, tree: Path
    ) -> None:
        unknown_key_in(tree, ANOTHER_SERVICE_FILE[service])

        LOADERS[service](tree)
        with pytest.raises(ConfigError, match="a_key_no_service_knows"):
            load_config(tree)

    def test_an_unknown_key_in_its_own_file_is_refused(self, service: str, tree: Path) -> None:
        unknown_key_in(tree, OWN_FILE[service])

        with pytest.raises(ConfigError, match="a_key_no_service_knows"):
            LOADERS[service](tree)
        with pytest.raises(ConfigError, match="a_key_no_service_knows"):
            load_config(tree)

    def test_its_own_file_is_one_it_owns(self, service: str) -> None:
        assert OWN_FILE[service] in FILES_READ_BY[service]


@pytest.mark.parametrize(("service", "name"), FOREIGN_FILES)
def test_an_unknown_key_in_a_file_read_as_foreign_is_not_its_concern(
    service: str, name: str, tree: Path
) -> None:
    unknown_key_in(tree, name)

    assert name in FILES_READ_BY[service]
    LOADERS[service](tree)
    with pytest.raises(ConfigError, match="a_key_no_service_knows"):
        load_config(tree)


class TestWhatEachViewHolds:
    def test_the_inference_view_holds_what_the_full_tree_holds_for_it(self) -> None:
        full, held = load_config(TREE), load_inference_config(TREE)

        assert held.services.inference_service == full.services.inference_service
        assert held.services.broker == full.services.broker
        assert held.models.model_repository == full.models.model_repository
        assert held.models.inference_services == full.models.inference_services
        assert {name: one.execution_provider for name, one in held.cost.devices.items()} == {
            name: one.execution_provider for name, one in full.cost.devices.items()
        }

    def test_the_client_view_holds_each_consumer_contract(self) -> None:
        full, held = load_config(TREE), load_consumer_config(TREE)

        assert held.services.consumer == full.services.consumer
        assert dict(held.contracts) == {name: one.contract for name, one in full.consumers.items()}

    def test_the_data_hub_view_holds_its_settings(self) -> None:
        assert load_data_hub_config(TREE).services.data_hub == load_config(TREE).services.data_hub

    def test_the_control_plane_view_holds_what_the_full_tree_holds_for_it(self) -> None:
        full, held = load_config(TREE), load_control_plane_config(TREE)

        assert held.services.control_plane == full.services.control_plane
        assert held.services.inference_service.port == full.services.inference_service.port
        assert dict(held.consumers) == dict(full.consumers)
        assert dict(held.approaches) == dict(full.approaches)
        assert held.cost == full.cost
        assert held.datasets.classes == full.datasets.classes
        assert held.datasets.dominant_classes == full.datasets.dominant_classes

    def test_the_observation_service_view_holds_what_the_full_tree_holds_for_it(self) -> None:
        full, held = load_config(TREE), load_observation_service_config(TREE)

        assert held.services.observation_service == full.services.observation_service
        assert held.services.metrics_database == full.services.metrics_database
        assert held.services.broker == full.services.broker
        assert held.cost == full.cost


class TestTheChecksAcrossTheFilesOfOneService:
    def test_the_control_plane_refuses_a_policy_with_no_file(self, tree: Path) -> None:
        path = tree / "services" / "control_plane.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["policy"] = "nowhere"
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="policy nowhere has no file"):
            load_control_plane_config(tree)

    def test_the_inference_service_refuses_a_device_type_with_no_price(self, tree: Path) -> None:
        path = tree / "cost" / "prices.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["devices"].pop("raspberry_pi4")
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(
            ConfigError, match="raspberry_pi4, which cost/prices.yaml does not hold"
        ):
            load_inference_config(tree)

    def test_the_client_refuses_a_contract_in_the_wrong_folder(self, tree: Path) -> None:
        path = tree / "consumers" / "c1_navigation" / "contract.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["consumer_id"] = "c2_safety"
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="does not match the folder c1_navigation"):
            load_consumer_config(tree)

    def test_the_client_refuses_a_tree_with_no_consumers_folder(self, tree: Path) -> None:
        shutil.rmtree(tree / "consumers")

        with pytest.raises(ConfigError, match="consumers: directory is missing"):
            load_consumer_config(tree)

    def test_the_client_refuses_a_consumers_folder_with_no_consumer(self, tree: Path) -> None:
        shutil.rmtree(tree / "consumers")
        (tree / "consumers").mkdir()

        with pytest.raises(ConfigError, match="consumers: no consumer folder found"):
            load_consumer_config(tree)


class TestTheImageNames:
    def test_only_the_local_image_file_is_read(self, tmp_path: Path) -> None:
        (tmp_path / "local").mkdir()
        (tmp_path / "local" / "images.yaml").write_text(
            "registry: registry.example:5000\nversion: v1\n", encoding="utf-8"
        )

        held = load_images_config(tmp_path)

        assert (held.registry, held.version) == ("registry.example:5000", "v1")

    def test_a_tree_with_no_local_image_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="local/images.yaml: file is missing"):
            load_images_config(tmp_path)


def test_the_prefix_the_control_plane_publishes_under_is_the_one_the_client_reads() -> None:
    config = load_config(TREE)

    assert (
        config.services.control_plane.result_topic_prefix
        == config.services.consumer.reply_topic_prefix
    )
