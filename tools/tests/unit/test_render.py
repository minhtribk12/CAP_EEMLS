from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from eemls_tools.config_tree import ConfigError, load_config
from eemls_tools.render import rendered
from support.repository import ROOT

TREE = ROOT / "config"
ONE_HOST = ROOT / "deploy" / "one_host" / "overrides.yaml"


def overrides(root: Path, body: Mapping[str, object]) -> Path:
    path = root / "overrides.yaml"
    path.write_text(yaml.safe_dump(dict(body)), encoding="utf-8")
    return path


def test_a_replaced_key_holds_its_new_value(tmp_path: Path) -> None:
    out = tmp_path / "config"
    rendered(
        TREE,
        overrides(tmp_path, {"services/control_plane.yaml": {"member_timeout_ms": 900}}),
        out,
    )

    assert load_config(out).services.control_plane.member_timeout_ms == 900


def test_a_file_no_override_names_is_copied_byte_for_byte(tmp_path: Path) -> None:
    out = tmp_path / "config"
    rendered(
        TREE,
        overrides(tmp_path, {"services/control_plane.yaml": {"member_timeout_ms": 900}}),
        out,
    )

    for source in TREE.rglob("*.yaml"):
        relative = source.relative_to(TREE)
        if relative != Path("services/control_plane.yaml"):
            assert (out / relative).read_bytes() == source.read_bytes(), relative


def test_a_file_the_tree_does_not_hold_is_refused_by_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="services/nowhere.yaml"):
        rendered(TREE, overrides(tmp_path, {"services/nowhere.yaml": {"x": 1}}), tmp_path / "c")


def test_a_key_the_file_does_not_hold_is_refused_by_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="services/control_plane.yaml: decision_intervals"):
        rendered(
            TREE,
            overrides(tmp_path, {"services/control_plane.yaml": {"decision_intervals": 5}}),
            tmp_path / "c",
        )


def test_a_tree_the_overrides_break_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        rendered(
            TREE,
            overrides(tmp_path, {"services/control_plane.yaml": {"periods_of_the_day": 5}}),
            tmp_path / "c",
        )


def test_the_same_overrides_render_the_same_tree(tmp_path: Path) -> None:
    body = {"services/control_plane.yaml": {"member_timeout_ms": 900}}
    rendered(TREE, overrides(tmp_path, body), tmp_path / "one")
    rendered(TREE, overrides(tmp_path, body), tmp_path / "two")

    for path in (tmp_path / "one").rglob("*.yaml"):
        assert (
            path.read_bytes()
            == (tmp_path / "two" / path.relative_to(tmp_path / "one")).read_bytes()
        )


def test_it_refuses_to_write_over_a_tree_that_is_already_there(tmp_path: Path) -> None:
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "held.yaml").write_text("kept: true\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="already holds"):
        rendered(TREE, overrides(tmp_path, {}), tmp_path / "c")


class TestTheOneHostRun:
    def test_it_serves_a_small_pool_of_fast_services(self, tmp_path: Path) -> None:
        config = load_config(rendered(TREE, ONE_HOST, tmp_path / "config"))

        assert len(config.models.inference_services.services) == 4

    def test_every_consumer_starts_on_services_of_that_pool(self, tmp_path: Path) -> None:
        config = load_config(rendered(TREE, ONE_HOST, tmp_path / "config"))
        pool = set(config.models.inference_services.services)

        for consumer in config.consumers.values():
            assert set(consumer.initial_ensemble) <= pool

    def test_the_consumers_ask_at_the_rates_of_the_article(self, tmp_path: Path) -> None:
        config = load_config(rendered(TREE, ONE_HOST, tmp_path / "config"))

        assert {one.contract.request_rate_per_s for one in config.consumers.values()} == {20.0}

    def test_every_device_class_runs_the_image_of_this_host(self, tmp_path: Path) -> None:
        config = load_config(rendered(TREE, ONE_HOST, tmp_path / "config"))

        assert set(config.services.deployment.image_tags.values()) == {"cpu-amd64"}

    def test_the_one_host_tree_does_not_autoscale(self, tmp_path: Path) -> None:
        config = load_config(rendered(TREE, ONE_HOST, tmp_path / "config"))

        assert not config.models.inference_services.autoscale.enabled
        assert load_config(TREE).models.inference_services.autoscale.enabled


EXAMPLES = sorted((ROOT / "examples" / "overrides").glob("*.yaml"))


def test_examples_of_overrides_are_given() -> None:
    assert {path.name for path in EXAMPLES} >= {"small_pool.yaml", "cluster_gpu.yaml"}


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.name)
def test_each_example_of_overrides_renders_a_tree_that_loads(tmp_path: Path, example: Path) -> None:
    out = rendered(TREE, example, tmp_path / "config")

    assert load_config(out).models.inference_services.services


class TestTheQuickStart:
    QUICK_START = ROOT / "examples" / "quick_start" / "overrides.yaml"

    def tree_with_the_example_local_files(self, root: Path) -> Path:
        tree = root / "source"
        shutil.copytree(TREE, tree, ignore=shutil.ignore_patterns("local"))
        shutil.copytree(ROOT / "examples" / "local", tree / "local")
        return tree

    def test_it_renders_a_tree_that_loads(self, tmp_path: Path) -> None:
        tree = self.tree_with_the_example_local_files(tmp_path)

        config = load_config(rendered(tree, self.QUICK_START, tmp_path / "config"))

        assert set(config.models.inference_services.services) == {
            "infer-b1-imagenet-pi4",
            "infer-b2-imagenet-bt3",
        }

    def test_it_serves_only_backbones_whose_source_the_example_gives(self, tmp_path: Path) -> None:
        tree = self.tree_with_the_example_local_files(tmp_path)
        backbones = yaml.safe_load((ROOT / "examples" / "local" / "models.yaml").read_text())

        config = load_config(rendered(tree, self.QUICK_START, tmp_path / "config"))

        for entry in config.models.model_repository.models.values():
            assert entry.variant_of is None
            assert entry.sha256 == backbones["base_models"][entry.base_model]["sha256"]
            assert len(entry.classes) == 1000

    def test_every_consumer_starts_on_both_services(self, tmp_path: Path) -> None:
        tree = self.tree_with_the_example_local_files(tmp_path)

        config = load_config(rendered(tree, self.QUICK_START, tmp_path / "config"))
        pool = set(config.models.inference_services.services)

        for consumer in config.consumers.values():
            assert set(consumer.initial_ensemble) == pool


REPOSITORY = "models/model_repository.yaml"


MAPPED = "b1_general"


def mapped_models(class_mapping: str, **keys: object) -> dict[str, object]:
    held = yaml.safe_load((TREE / REPOSITORY).read_text(encoding="utf-8"))["models"]
    entry = {key: value for key, value in held[MAPPED].items() if key != "classes"}
    return {
        REPOSITORY: {"models": held | {MAPPED: entry | {"class_mapping": class_mapping} | keys}}
    }


def mapping(root: Path, body: Mapping[str, object]) -> Path:
    path = root / "mapping.yaml"
    path.write_text(yaml.safe_dump(dict(body)), encoding="utf-8")
    return path


class TestAClassMapping:
    def rendered_classes(self, root: Path, body: Mapping[str, object]) -> tuple[str, ...]:
        mapping(root, body)
        out = rendered(
            TREE,
            overrides(root, mapped_models("mapping.yaml")),
            root / "config",
        )
        return load_config(out).models.model_repository.models[MAPPED].classes

    def test_it_names_each_output_by_the_class_it_maps_to(self, tmp_path: Path) -> None:
        classes = self.rendered_classes(
            tmp_path, {"outputs": 6, "classes": {"bird": [[1, 2]], "dog": [0, 4]}}
        )

        assert classes == ("dog", "bird", "bird", "output_3", "dog", "output_5")

    def test_its_file_is_read_next_to_the_overrides(self, tmp_path: Path) -> None:
        (tmp_path / "maps").mkdir()
        mapping(tmp_path / "maps", {"outputs": 2, "classes": {"bird": [1]}})
        out = rendered(
            TREE,
            overrides(tmp_path, mapped_models("maps/mapping.yaml")),
            tmp_path / "config",
        )

        assert load_config(out).models.model_repository.models[MAPPED].classes == (
            "output_0",
            "bird",
        )

    def test_a_missing_file_is_refused_by_path(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="nowhere.yaml: no such file"):
            rendered(
                TREE,
                overrides(tmp_path, mapped_models("nowhere.yaml")),
                tmp_path / "config",
            )

    def test_an_output_past_the_count_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="mapping.yaml: output 3 is past the 3 outputs"):
            self.rendered_classes(tmp_path, {"outputs": 3, "classes": {"bird": [[1, 3]]}})

    def test_an_output_mapped_twice_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="mapping.yaml: output 1 maps to more than one class"):
            self.rendered_classes(tmp_path, {"outputs": 3, "classes": {"bird": [1], "dog": [1]}})

    def test_an_entry_with_both_classes_and_a_mapping_is_refused(self, tmp_path: Path) -> None:
        mapping(tmp_path, {"outputs": 2, "classes": {"bird": [1]}})
        both = mapped_models("mapping.yaml", classes=["bird", "dog"])

        with pytest.raises(ConfigError, match=f"{MAPPED}: both classes and a class mapping"):
            rendered(TREE, overrides(tmp_path, both), tmp_path / "config")


QUICK_START_CLASSES = {
    "bird": [*range(7, 25), *range(80, 101), *range(127, 147)],
    "dog": list(range(151, 269)),
    "cat": list(range(281, 286)),
    "car": [407, 436, 468, 511, 609, 627, 656, 734, 751, 817],
    "truck": [555, 569, 675, 717, 864, 867],
    "bus": [654, 779, 874],
    "motorcycle": [665, 670],
    "cyclist": [444, 671],
    "train": [466, 547, 565, 705, 820, 829],
    "boat": [403, 472, 484, 510, 554, 576, 625, 628, 724, 780, 814, 871, 914],
    "airplane": [404, 895],
    "traffic_sign": [919],
    "traffic_light": [920],
}


def test_the_quick_start_models_name_our_classes_at_their_imagenet_outputs(
    tmp_path: Path,
) -> None:
    tree = TestTheQuickStart().tree_with_the_example_local_files(tmp_path)
    config = load_config(rendered(tree, TestTheQuickStart.QUICK_START, tmp_path / "config"))
    ours = {index: name for name, indices in QUICK_START_CLASSES.items() for index in indices}

    for entry in config.models.model_repository.models.values():
        assert entry.classes == tuple(ours.get(i, f"output_{i}") for i in range(1000))
