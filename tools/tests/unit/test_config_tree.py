from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from types import MappingProxyType

import pytest
import yaml

from eemls.domain.contract import ConsumerContract, Objective
from eemls.ensemble_optimization.approaches.baselines.axiomvision import AxiomVisionSettings
from eemls.ensemble_optimization.approaches.baselines.cocktail import CocktailSettings
from eemls.ensemble_optimization.approaches.baselines.frugalml import FrugalMlSettings
from eemls.ensemble_optimization.approaches.cap.approach import CapSettings
from eemls.evaluation_engine.contract_scoring import ScoreParts, score
from eemls.testing.trees import committed_tree
from eemls_tools.config_tree import ConfigError, load_config
from eemls_tools.model_config import VariationRecipeConfig
from support.repository import ROOT

TREE = ROOT / "config"


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    return committed_tree(tmp_path / "config")


def write(path: Path, body: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(body), encoding="utf-8")


def services(tree: Path, *names: str) -> None:
    path = tree / "models" / "inference_services.yaml"
    held = yaml.safe_load(path.read_text(encoding="utf-8"))
    held["services"].update(
        {name: {"model": "b3", "device_type": "raspberry_pi5", "replicas": 1} for name in names}
    )
    repository = tree / "models" / "model_repository.yaml"
    body = yaml.safe_load(repository.read_text(encoding="utf-8"))
    body["models"].update(
        {
            "b3": {
                "hugging_face_id": "onnx-community/b3_1.0_224",
                "base_model": "b3",
                "onnx_file": "model.onnx",
                "sha256": "a" * 64,
                "input_height": 224,
                "input_width": 224,
                "normalization_mean": [0.5, 0.5, 0.5],
                "normalization_std": [0.5, 0.5, 0.5],
                "classes": ["car"],
                "memory_mb": 14.0,
                "variant_of": None,
            }
        }
    )
    write(repository, body)
    write(path, held)


def starting(tree: Path, *names: str) -> None:
    for folder in sorted((tree / "consumers").iterdir()):
        write(folder / "initial_ensemble.yaml", {"members": list(names)})


def edit(path: Path, change: dict[str, object]) -> None:
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body.update(change)
    path.write_text(yaml.safe_dump(body), encoding="utf-8")


def test_the_copy_the_mutation_tests_start_from_loads(tree: Path) -> None:
    config = load_config(tree)

    assert config.services.control_plane.policy == "cap"


def test_the_committed_tree_loads() -> None:
    config = load_config(TREE)

    assert set(config.consumers) == {"c1_navigation", "c2_safety", "c3_surveillance"}
    assert set(config.approaches) == {
        "cap",
        "scale_reduction",
        "sneakpeek",
        "axiomvision",
        "cocktail",
        "frugalml",
        "keep_the_ensemble",
    }


def test_the_loader_reads_the_directory_the_environment_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EEMLS_CONFIG_DIR", str(TREE))

    assert load_config().services.control_plane.max_ensemble_size == 3


def test_a_missing_config_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="nowhere"):
        load_config(tmp_path / "nowhere")


def test_an_unknown_key_names_its_file_and_its_key(tree: Path) -> None:
    edit(tree / "approaches" / "cap.yaml", {"evaluaton_window": 500})

    with pytest.raises(ConfigError) as refusal:
        load_config(tree)

    assert "approaches/cap.yaml" in str(refusal.value)
    assert "evaluaton_window" in str(refusal.value)


def test_a_missing_key_names_its_file_and_its_key(tree: Path) -> None:
    path = tree / "approaches" / "cap.yaml"
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    del body["evaluation_window"]
    path.write_text(yaml.safe_dump(body), encoding="utf-8")

    with pytest.raises(ConfigError) as refusal:
        load_config(tree)

    assert "approaches/cap.yaml" in str(refusal.value)
    assert "evaluation_window" in str(refusal.value)


def test_a_url_that_is_not_a_url_is_refused(tree: Path) -> None:
    edit(tree / "services" / "control_plane.yaml", {"data_hub_url": "data-hub:8080"})

    with pytest.raises(ConfigError, match="data_hub_url"):
        load_config(tree)


def test_a_negative_window_is_refused(tree: Path) -> None:
    edit(tree / "approaches" / "cap.yaml", {"evaluation_window": -500})

    with pytest.raises(ConfigError, match="evaluation_window"):
        load_config(tree)


def test_a_weight_outside_its_range_is_refused(tree: Path) -> None:
    path = tree / "consumers" / "c1_navigation" / "contract.yaml"
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body["cost_weight"] = -1.0
    path.write_text(yaml.safe_dump(body), encoding="utf-8")

    with pytest.raises(ConfigError, match="cost_weight"):
        load_config(tree)


def test_a_contract_may_only_name_a_cdm_its_consumer_declares(tree: Path) -> None:
    path = tree / "consumers" / "c1_navigation" / "contract.yaml"
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body["objectives"] = {"accuracy_on_unicorns": body["objectives"]["accuracy_car"]}
    path.write_text(yaml.safe_dump(body), encoding="utf-8")

    with pytest.raises(ConfigError, match="accuracy_on_unicorns"):
        load_config(tree)


def test_a_consumer_folder_and_its_contract_must_agree(tree: Path) -> None:
    edit(tree / "consumers" / "c2_safety" / "contract.yaml", {"consumer_id": "c9_other"})

    with pytest.raises(ConfigError, match="c9_other"):
        load_config(tree)


def test_a_policy_may_not_exceed_the_ensemble_size_of_the_control_plane(tree: Path) -> None:
    edit(tree / "approaches" / "cap.yaml", {"max_ensemble_size": 4})

    with pytest.raises(ConfigError, match="max_ensemble_size"):
        load_config(tree)


def test_the_periods_of_the_day_are_the_dominant_classes_of_the_dataset(tree: Path) -> None:
    edit(tree / "services" / "control_plane.yaml", {"periods_of_the_day": 5})

    with pytest.raises(ConfigError, match="periods_of_the_day"):
        load_config(tree)


def test_an_approach_may_serve_a_smaller_ensemble_than_the_control_plane_allows(
    tree: Path,
) -> None:
    edit(tree / "approaches" / "cap.yaml", {"max_ensemble_size": 2})
    services(tree, "infer-a", "infer-b")
    starting(tree, "infer-a", "infer-b")

    held = load_config(tree)

    assert held.approaches["cap"].max_ensemble_size == 2
    assert held.services.control_plane.max_ensemble_size == 3


def test_a_starting_ensemble_above_the_size_of_the_running_approach_is_refused(
    tree: Path,
) -> None:
    edit(tree / "approaches" / "cap.yaml", {"max_ensemble_size": 1})
    write(
        tree / "consumers" / "c1_navigation" / "initial_ensemble.yaml",
        {"members": ["infer-a", "infer-b"]},
    )
    services(tree, "infer-a", "infer-b")

    with pytest.raises(ConfigError, match="max_ensemble_size"):
        load_config(tree)


def test_a_starting_ensemble_within_the_running_approach_is_served(tree: Path) -> None:
    edit(tree / "approaches" / "cap.yaml", {"max_ensemble_size": 2})
    services(tree, "infer-a", "infer-b")
    starting(tree, "infer-a", "infer-b")

    held = load_config(tree).consumers["c1_navigation"]

    assert held.initial_ensemble == ("infer-a", "infer-b")


def test_a_loaded_setting_cannot_be_changed() -> None:
    config = load_config(TREE)

    with pytest.raises(ValueError, match="frozen|immutable"):
        config.services.control_plane.port = 9999  # type: ignore[misc]


ABOUT_THE_STUDY = ("paper", "article", "approach", "cap", "figure", "experiment", "reports")


def test_no_description_talks_about_the_study_instead_of_the_thing() -> None:
    for path in TREE.rglob("*.yaml"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped.startswith("description:"):
                continue
            words = re.findall(r"[a-z]+", stripped.lower())
            named = sorted(set(words) & set(ABOUT_THE_STUDY))
            assert not named, f"{path}:{number} describes the study, not the thing: {named}"


CLASS_WEIGHTS = {
    "car": 4.0,
    "truck": 3.0,
    "bus": 3.0,
    "pedestrian": 1.0,
    "cyclist": 1.0,
    "motorcycle": 1.0,
}


def declare_weighted(tree: Path, consumer: str, *, extra_class: str | None = None) -> None:
    weights = dict(CLASS_WEIGHTS)
    if extra_class:
        weights[extra_class] = 2.0
    path = tree / "consumers" / consumer / "cdm.yaml"
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body["cdms"]["accuracy_weighted_by_class"] = {
        "measured": "per_window",
        "window": 500,
        "unit": "ratio",
        "description": "Accuracy over the classes, weighted inside the metric",
        "specification": {
            "function": "ratio",
            "inputs": [
                {
                    "function": "weighted_count",
                    "group_by": "true_label",
                    "weights": weights,
                    "scopes": [{"attribute": "predicted_label", "same_as": "true_label"}],
                },
                {
                    "function": "weighted_count",
                    "group_by": "true_label",
                    "weights": weights,
                    "scopes": [{"attribute": "true_label", "observed": True}],
                },
            ],
        },
    }
    path.write_text(yaml.safe_dump(body), encoding="utf-8")


def test_one_cdm_id_never_means_two_different_metrics() -> None:
    declared: dict[str, dict[str, str]] = {}
    for path in sorted(TREE.glob("consumers/*/cdm.yaml")):
        body = yaml.safe_load(path.read_text(encoding="utf-8"))["cdms"]
        for name, definition in body.items():
            declared.setdefault(name, {})[path.parent.name] = json.dumps(definition, sort_keys=True)
    for name, by_consumer in declared.items():
        distinct = set(by_consumer.values())
        assert len(distinct) == 1, (
            f"{name} means {len(distinct)} different metrics, declared by {sorted(by_consumer)}"
        )


def test_no_configuration_file_carries_a_credential() -> None:
    for path in TREE.rglob("*.yaml"):
        body = path.read_text(encoding="utf-8").lower()
        assert "password:" not in body, path
        assert "secret:" not in body, path
        assert "token:" not in body, path


class TestThePaperValues:
    def test_cap_parameters(self) -> None:
        cap = load_config(TREE).approaches["cap"]

        assert isinstance(cap, CapSettings)
        assert cap.evaluation_window == 500
        assert cap.max_ensemble_size == 3
        assert cap.initial_switching_margin == 0.25
        assert cap.contributions_before_margin == 30
        assert cap.prior_precision == 1.0
        assert cap.noise_variance == 1.0

    def test_the_window_an_approach_reads_comes_from_its_own_file(self) -> None:
        held = load_config(TREE)

        assert held.approaches["cap"].evaluation_window == 500

    def test_a_fixed_ensemble_reads_no_recent_predictions(self) -> None:
        held = load_config(TREE)

        assert held.approaches["keep_the_ensemble"].evaluation_window == 0

    def test_a_policy_stating_an_objective_of_its_own_is_refused(self, tree: Path) -> None:
        edit(tree / "approaches" / "cap.yaml", {"objective": "general_accuracy"})

        with pytest.raises(ConfigError, match="objective"):
            load_config(tree)

    def test_the_baseline_knobs_match_the_runs_that_produced_the_paper(self) -> None:
        found = load_config(TREE).approaches
        frugalml, cocktail, axiomvision = (
            found["frugalml"],
            found["cocktail"],
            found["axiomvision"],
        )

        assert isinstance(frugalml, FrugalMlSettings)
        assert isinstance(cocktail, CocktailSettings)
        assert isinstance(axiomvision, AxiomVisionSettings)
        assert frugalml.budget_per_1k == 20.0
        assert frugalml.quality_level == 0.8
        assert frugalml.failing_intervals == 3
        assert cocktail.accuracy_target == 0.85
        assert cocktail.accuracy_target_range == 0.05
        assert axiomvision.exploration_alpha == 0.25
        assert axiomvision.prior_zeta == 1.0
        assert axiomvision.accuracy_target == 0.85

    def test_control_plane_decides_every_five_queries(self) -> None:
        control_plane = load_config(TREE).services.control_plane

        assert control_plane.decision_interval_queries == 5
        assert control_plane.max_ensemble_size == 3
        assert control_plane.periods_of_the_day == 6

    def test_every_consumer_returns_half_its_labels_after_the_same_delay(self) -> None:
        consumers = load_config(TREE).consumers

        for consumer in consumers.values():
            assert consumer.contract.feedback.fraction == 0.5
            assert consumer.contract.feedback.delay_intervals == 15

    def test_the_prioritized_classes_follow_the_paper(self) -> None:
        consumers = load_config(TREE).consumers

        assert consumers["c1_navigation"].contract.prioritized_classes == ("car", "truck", "bus")
        assert consumers["c2_safety"].contract.prioritized_classes == (
            "pedestrian",
            "cyclist",
            "motorcycle",
        )
        assert consumers["c3_surveillance"].contract.prioritized_classes == ()

    def test_the_safety_consumer_scores_its_classes_and_its_miss_rate(self) -> None:
        consumer = load_config(TREE).consumers["c2_safety"]

        assert set(consumer.contract.objectives) == {
            "accuracy_pedestrian",
            "accuracy_cyclist",
            "accuracy_motorcycle",
            "accuracy_car",
            "accuracy_truck",
            "accuracy_bus",
            "miss_rate_cyclist_as_motorcycle",
        }

    def test_the_miss_rate_approach_weighs_the_miss_rate_ten_times_the_accuracy(self) -> None:
        accuracy = (
            load_config(TREE).consumers["c2_safety"].contract.objectives["accuracy_pedestrian"]
        )
        miss_rate = Objective(
            min_value=0.0,
            max_value=0.2,
            weight=20.0,
            direction="minimize",
            mapping="linear",
            mapping_base=None,
        )
        slope = (miss_rate.weight / miss_rate.max_value) / (accuracy.weight / accuracy.max_value)

        assert slope == pytest.approx(10.0)

    def test_the_six_dominant_classes_of_the_dataset(self) -> None:
        datasets = load_config(TREE).datasets

        assert datasets.dominant_classes == (
            "pedestrian",
            "cyclist",
            "motorcycle",
            "car",
            "truck",
            "bus",
        )
        assert set(datasets.sources) == {"imagenet", "open_images", "multispectral"}

    def test_the_tail_classes_are_the_ones_a_street_also_holds(self) -> None:
        datasets = load_config(TREE).datasets

        assert datasets.tail_classes
        assert not set(datasets.tail_classes) & set(datasets.dominant_classes)

    def test_every_class_is_a_dominant_one_or_a_tail_one(self) -> None:
        datasets = load_config(TREE).datasets

        assert datasets.classes == datasets.dominant_classes + datasets.tail_classes

    def test_a_tail_class_is_drawn_less_than_a_dominant_one(self) -> None:
        datasets = load_config(TREE).datasets

        assert datasets.images_per_tail_class < datasets.images_per_dominant_class

    def test_the_two_conditions_of_the_article_are_declared(self) -> None:
        datasets = load_config(TREE).datasets

        assert datasets.conditions == ("day", "night")

    def test_every_condition_is_supplied_by_a_source(self) -> None:
        datasets = load_config(TREE).datasets

        supplied = {one for source in datasets.sources.values() for one in source.conditions}
        assert supplied == set(datasets.conditions)

    def test_the_night_source_covers_three_of_the_six_dominant_classes(self) -> None:
        datasets = load_config(TREE).datasets

        at_night = {
            name
            for source in datasets.sources.values()
            if "night" in source.conditions
            for name in source.contributes
        }
        assert at_night & set(datasets.dominant_classes) == {"pedestrian", "cyclist", "car"}

    def test_the_tail_a_street_also_holds_is_sourced_at_both_conditions(self) -> None:
        datasets = load_config(TREE).datasets

        at_night = {
            name
            for source in datasets.sources.values()
            if "night" in source.conditions
            for name in source.contributes
        }
        assert at_night & set(datasets.tail_classes) == {
            "traffic_cone",
            "road_bump",
            "road_curve",
            "car_stop_sign",
            "guardrail",
        }

    def test_the_night_images_come_from_the_source_the_article_cites_for_them(self) -> None:
        datasets = load_config(TREE).datasets

        at_night = sorted(
            name for name, source in datasets.sources.items() if "night" in source.conditions
        )
        assert at_night == ["multispectral"]

    def test_a_class_a_dominant_and_a_tail_list_both_name_is_refused(self, tree: Path) -> None:
        path = tree / "datasets" / "sources.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["tail_classes"] = [*body["tail_classes"], "car"]
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="car"):
            load_config(tree)

    def test_a_source_supplying_a_condition_the_dataset_does_not_declare_is_refused(
        self, tree: Path
    ) -> None:
        path = tree / "datasets" / "sources.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["sources"]["imagenet"]["conditions"] = ["dusk"]
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="dusk"):
            load_config(tree)

    def test_a_condition_no_source_supplies_is_refused(self, tree: Path) -> None:
        path = tree / "datasets" / "sources.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        for source in body["sources"].values():
            source["conditions"] = ["day"]
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="night"):
            load_config(tree)

    def test_the_cost_model_prices_electricity_and_amortizes_the_devices(self) -> None:
        cost = load_config(TREE).cost

        assert cost.energy_price_per_kwh == 0.20
        assert cost.device_lifetime_years == 3
        assert cost.eur_per_cost_unit == 0.00001
        assert cost.devices
        for device in cost.devices.values():
            assert device.hardware_price_eur > 0
            assert device.idle_power_watts < device.peak_power_watts


class TestEverySchemaRefusal:
    def test_an_unknown_onnx_provider_is_refused(self, tree: Path) -> None:
        edit(
            tree / "services" / "inference_service.yaml",
            {"providers": {"jetson": ["FastProvider"]}},
        )

        with pytest.raises(ConfigError, match="unknown providers"):
            load_config(tree)

    def test_an_empty_provider_chain_is_refused(self, tree: Path) -> None:
        edit(tree / "services" / "inference_service.yaml", {"providers": {"jetson": []}})

        with pytest.raises(ConfigError, match="names no provider"):
            load_config(tree)

    def test_a_reconnect_window_in_the_wrong_order_is_refused(self, tree: Path) -> None:
        edit(tree / "services" / "broker.yaml", {"reconnect_min_s": 9.0})

        with pytest.raises(ConfigError, match="reconnect_min_s"):
            load_config(tree)

    def test_a_device_consuming_more_power_idle_than_at_peak_is_refused(self, tree: Path) -> None:
        path = tree / "cost" / "prices.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["devices"]["jetson_nano"]["idle_power_watts"] = 99.0
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="idle power"):
            load_config(tree)

    def test_a_device_number_without_a_source_is_refused(self, tree: Path) -> None:
        path = tree / "cost" / "prices.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["devices"]["jetson_nano"]["source"] = ""
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="source"):
            load_config(tree)

    def test_a_charge_unit_other_than_the_reported_one_is_refused(self, tree: Path) -> None:
        edit(tree / "cost" / "prices.yaml", {"charge_unit": "per_query"})

        with pytest.raises(ConfigError, match="per_1k_queries"):
            load_config(tree)

    def test_a_class_no_source_contributes_is_refused(self, tree: Path) -> None:
        path = tree / "datasets" / "sources.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["tail_classes"].append("tram")
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="tram"):
            load_config(tree)

    def test_a_source_contributing_an_unknown_class_is_refused(self, tree: Path) -> None:
        path = tree / "datasets" / "sources.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["sources"]["imagenet"]["contributes"].append("tram")
        body["sources"]["imagenet"]["label_map"]["n00000000"] = "tram"
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="tram"):
            load_config(tree)

    def test_a_source_whose_label_map_misses_a_class_it_claims_is_refused(self, tree: Path) -> None:
        path = tree / "datasets" / "sources.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["sources"]["imagenet"]["label_map"] = {"n02958343": "car"}
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="maps no original label"):
            load_config(tree)

    def test_a_repeated_class_is_refused(self, tree: Path) -> None:
        path = tree / "datasets" / "sources.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["dominant_classes"].append("car")
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="repeats a name"):
            load_config(tree)

    def test_a_model_without_a_real_checksum_is_refused(self, tree: Path) -> None:
        edit(
            tree / "models" / "model_repository.yaml",
            {
                "models": {
                    "b3": {
                        "hugging_face_id": "owner/b3",
                        "base_model": "b3",
                        "onnx_file": "b3.onnx",
                        "sha256": "deadbeef",
                        "input_height": 224,
                        "input_width": 224,
                        "normalization_mean": [0.485, 0.456, 0.406],
                        "normalization_std": [0.229, 0.224, 0.225],
                        "classes": ["car"],
                        "memory_mb": 14.0,
                        "variant_of": None,
                    }
                }
            },
        )

        with pytest.raises(ConfigError, match="should match pattern"):
            load_config(tree)

    def test_an_objective_range_without_width_is_refused(self, tree: Path) -> None:
        path = tree / "consumers" / "c1_navigation" / "contract.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["objectives"]["accuracy_car"]["max_value"] = 0.0
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="not below max_value"):
            load_config(tree)

    def test_the_same_key_written_twice_in_a_file_is_refused(self, tree: Path) -> None:
        (tree / "approaches" / "keep_the_ensemble.yaml").write_text(
            "kind: keep_the_ensemble\nmax_ensemble_size: 3\nmax_ensemble_size: 2\n",
            encoding="utf-8",
        )

        with pytest.raises(ConfigError, match="appears twice"):
            load_config(tree)

    def test_a_linear_objective_carrying_a_scale_base_is_refused(self, tree: Path) -> None:
        path = tree / "consumers" / "c1_navigation" / "contract.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["objectives"]["accuracy_car"]["mapping_base"] = 2.0
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="mapping_base belongs to a log mapping"):
            load_config(tree)

    def test_a_log_objective_without_its_base_is_refused(self, tree: Path) -> None:
        path = tree / "consumers" / "c1_navigation" / "contract.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["objectives"]["accuracy_car"]["mapping"] = "log"
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="log mapping needs its mapping_base"):
            load_config(tree)

    def test_a_contract_without_an_objective_is_refused(self, tree: Path) -> None:
        edit(tree / "consumers" / "c1_navigation" / "contract.yaml", {"objectives": {}})

        with pytest.raises(ConfigError, match="objectives is empty"):
            load_config(tree)

    def test_a_period_metric_without_a_window_is_refused(self, tree: Path) -> None:
        path = tree / "consumers" / "c3_surveillance" / "cdm.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["cdms"]["general_accuracy"]["window"] = None
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="window is missing"):
            load_config(tree)

    def test_an_expression_without_a_function_is_refused(self, tree: Path) -> None:
        path = tree / "consumers" / "c3_surveillance" / "cdm.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["cdms"]["general_accuracy"]["specification"] = {"attribute": "correct"}
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="specification.function: Field required"):
            load_config(tree)

    def test_a_consumer_declaring_no_cdm_is_refused(self, tree: Path) -> None:
        (tree / "consumers" / "c3_surveillance" / "cdm.yaml").write_text(
            "cdms: {}\n", encoding="utf-8"
        )

        with pytest.raises(ConfigError, match="definitions is empty"):
            load_config(tree)

    def test_a_cdm_file_without_its_cdms_key_is_refused(self, tree: Path) -> None:
        (tree / "consumers" / "c3_surveillance" / "cdm.yaml").write_text(
            "general_accuracy: {}\n", encoding="utf-8"
        )

        with pytest.raises(ConfigError, match="cdms: Field required"):
            load_config(tree)

    def test_a_device_naming_an_execution_provider_that_does_not_exist_is_refused(
        self, tree: Path
    ) -> None:
        path = tree / "cost" / "prices.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["devices"]["jetson_nano"]["execution_provider"] = "quantum"
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="execution provider quantum"):
            load_config(tree)


class TestEveryFileRefusal:
    def test_a_missing_file_names_itself(self, tree: Path) -> None:
        (tree / "datasets" / "sources.yaml").unlink()

        with pytest.raises(ConfigError, match="datasets/sources.yaml: file is missing"):
            load_config(tree)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file whatever its mode")
    def test_a_folder_it_may_not_open_names_its_file(self, tree: Path) -> None:
        (tree / "datasets").chmod(0o000)
        try:
            with pytest.raises(ConfigError, match="datasets/sources.yaml: could not be read"):
                load_config(tree)
        finally:
            (tree / "datasets").chmod(0o755)

    def test_a_file_that_is_not_text_names_itself(self, tree: Path) -> None:
        (tree / "datasets" / "sources.yaml").write_bytes(b"\xff\xfe\x00broken")

        with pytest.raises(ConfigError, match="datasets/sources.yaml: could not be read"):
            load_config(tree)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file whatever its mode")
    def test_a_file_it_may_not_read_names_itself(self, tree: Path) -> None:
        (tree / "datasets" / "sources.yaml").chmod(0o000)

        with pytest.raises(ConfigError, match="datasets/sources.yaml: could not be read"):
            load_config(tree)

    def test_a_deleted_policy_file_is_reported_through_the_service_that_names_it(
        self, tree: Path
    ) -> None:
        (tree / "approaches" / "cap.yaml").unlink()

        with pytest.raises(ConfigError, match="policy cap has no file in approaches"):
            load_config(tree)

    def test_an_empty_file_is_refused(self, tree: Path) -> None:
        (tree / "services" / "broker.yaml").write_text("", encoding="utf-8")

        with pytest.raises(ConfigError, match="file is empty"):
            load_config(tree)

    def test_a_file_that_is_not_a_mapping_is_refused(self, tree: Path) -> None:
        (tree / "services" / "broker.yaml").write_text("- info\n", encoding="utf-8")

        with pytest.raises(ConfigError, match="not a mapping"):
            load_config(tree)

    def test_a_file_that_is_not_yaml_is_refused(self, tree: Path) -> None:
        (tree / "services" / "broker.yaml").write_text("level: [info\n", encoding="utf-8")

        with pytest.raises(ConfigError, match="services/broker.yaml"):
            load_config(tree)

    def test_an_unknown_policy_kind_is_refused(self, tree: Path) -> None:
        edit(tree / "approaches" / "keep_the_ensemble.yaml", {"kind": "magic"})

        with pytest.raises(ConfigError, match="not one of"):
            load_config(tree)

    def test_two_approaches_may_share_one_kind(self, tree: Path) -> None:
        (tree / "approaches" / "cap_probe.yaml").write_text(
            (tree / "approaches" / "cap.yaml").read_text(encoding="utf-8"), encoding="utf-8"
        )

        config = load_config(tree)

        assert config.approaches["cap_probe"].kind == "cap"

    def test_a_control_plane_naming_a_policy_with_no_file_is_refused(self, tree: Path) -> None:
        edit(tree / "services" / "control_plane.yaml", {"policy": "oracle"})

        with pytest.raises(ConfigError, match="has no file in approaches"):
            load_config(tree)

    def test_a_service_naming_a_model_outside_the_catalogue_is_refused(self, tree: Path) -> None:
        edit(
            tree / "models" / "inference_services.yaml",
            {
                "services": {
                    "infer-ghost": {
                        "model": "ghost_net",
                        "device_type": "raspberry_pi5",
                        "replicas": 1,
                    }
                }
            },
        )

        starting(tree, "infer-ghost")

        with pytest.raises(ConfigError, match="model_repository does not hold"):
            load_config(tree)

    def test_a_consumer_folder_without_files_is_refused(self, tree: Path) -> None:
        (tree / "consumers" / "c4_empty").mkdir()

        with pytest.raises(ConfigError, match="c4_empty"):
            load_config(tree)

    def test_a_tree_without_approaches_is_refused(self, tree: Path) -> None:
        for path in (tree / "approaches").glob("*.yaml"):
            path.unlink()

        with pytest.raises(ConfigError, match="no approach file found"):
            load_config(tree)


class TestWhatCannotBeQuietlyCorrected:
    @pytest.mark.parametrize(
        ("key", "value", "refusal"),
        [
            ("evaluation_window", "500", "valid integer"),
            ("max_ensemble_size", 3.0, "valid integer"),
            ("initial_switching_margin", "0.25", "valid number"),
            ("noise_variance", float("inf"), "finite number"),
        ],
    )
    def test_a_quietly_corrected_number_is_refused(
        self, tree: Path, key: str, value: object, refusal: str
    ) -> None:
        edit(tree / "approaches" / "cap.yaml", {key: value})

        with pytest.raises(ConfigError, match=refusal):
            load_config(tree)

    def test_a_quoted_price_is_refused(self, tree: Path) -> None:
        edit(tree / "cost" / "prices.yaml", {"energy_price_per_kwh": "0.20"})

        with pytest.raises(ConfigError, match="valid number"):
            load_config(tree)

    def test_a_whole_number_is_accepted_where_a_fraction_belongs(self, tree: Path) -> None:
        path = tree / "consumers" / "c1_navigation" / "contract.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["objectives"]["accuracy_car"]["weight"] = 2
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        assert load_config(tree).consumers["c1_navigation"].contract.objectives[
            "accuracy_car"
        ].weight == pytest.approx(2.0)

    def test_an_objective_naming_a_violation_penalty_is_refused(self, tree: Path) -> None:
        path = tree / "consumers" / "c1_navigation" / "contract.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["objectives"]["accuracy_car"]["violation_penalty"] = 1.0
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="violation_penalty"):
            load_config(tree)

    def test_an_extra_key_in_a_cdm_file_is_refused(self, tree: Path) -> None:
        path = tree / "consumers" / "c3_surveillance" / "cdm.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["window"] = 500
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="window"):
            load_config(tree)

    @pytest.mark.parametrize(
        "stray",
        [
            "services/orchestrator.yaml",
            "services/logging.yaml",
            "services/control_plane.yml",
            "cost/prices_v2.yaml",
            "consumers/c4.yaml",
            "consumers/c2_safety/contract.yaml.bak.yaml",
        ],
    )
    def test_a_file_nothing_reads_is_refused(self, tree: Path, stray: str) -> None:
        (tree / stray).write_text("port: 1\n", encoding="utf-8")

        with pytest.raises(ConfigError, match="nothing reads"):
            load_config(tree)

    def test_the_hidden_copies_of_a_config_map_folder_are_not_read(self, tree: Path) -> None:
        hidden = tree / "..2026_10_18_19_14_29.2800882840" / "services"
        hidden.mkdir(parents=True)
        (hidden / "control_plane.yaml").write_text("port: 1\n", encoding="utf-8")

        assert load_config(tree).services.control_plane.port != 1

    def test_a_specification_cannot_be_changed_after_loading(self, tree: Path) -> None:
        declare_weighted(tree, "c1_navigation")

        specification = (
            load_config(tree)
            .consumers["c1_navigation"]
            .cdms["accuracy_weighted_by_class"]
            .specification
        )
        weights = specification.inputs[0].weights

        assert isinstance(weights, MappingProxyType)

    def test_a_loaded_tree_can_be_written_back_out(self) -> None:
        body = load_config(TREE).cost.model_dump()

        assert body["devices"]["jetson_nano"]["idle_power_watts"] == 2.0


class TestTheRefusalsNothingCovered:
    def test_a_prioritized_class_the_dataset_does_not_hold_is_refused(self, tree: Path) -> None:
        edit(
            tree / "consumers" / "c1_navigation" / "contract.yaml",
            {"prioritized_classes": ["tram"]},
        )

        with pytest.raises(ConfigError, match="dataset does not hold"):
            load_config(tree)

    def test_a_tree_with_no_devices_is_refused(self, tree: Path) -> None:
        edit(tree / "cost" / "prices.yaml", {"devices": {}})

        with pytest.raises(ConfigError, match="devices is empty"):
            load_config(tree)

    def test_a_source_contributing_nothing_is_refused(self, tree: Path) -> None:
        path = tree / "datasets" / "sources.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["sources"]["imagenet"]["contributes"] = []
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="contributes no class"):
            load_config(tree)

    def test_no_directory_and_no_environment_variable_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="EEMLS_CONFIG_DIR is not set"):
            load_config()

    def test_a_consumer_folder_without_a_contract_is_refused(self, tree: Path) -> None:
        (tree / "consumers" / "c4_empty").mkdir()

        with pytest.raises(ConfigError, match="consumers/c4_empty/contract.yaml: file is missing"):
            load_config(tree)


class TestTheRefusalsOfTheWholeTree:
    def test_a_requirement_naming_a_cdm_the_consumer_does_not_declare_is_refused(
        self, tree: Path
    ) -> None:
        path = tree / "consumers" / "c2_safety" / "contract.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["requirements"] = [
            {
                "cdm": "accuracy_on_unicorns",
                "operator": ">=",
                "target_value": 0.8,
            }
        ]
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="accuracy_on_unicorns"):
            load_config(tree)

    def test_a_metric_weighing_a_class_the_dataset_does_not_hold_is_refused(
        self, tree: Path
    ) -> None:
        declare_weighted(tree, "c1_navigation", extra_class="tram")

        with pytest.raises(ConfigError, match="tram"):
            load_config(tree)

    def test_a_tree_without_a_consumers_directory_is_refused(self, tree: Path) -> None:
        import shutil

        shutil.rmtree(tree / "consumers")

        with pytest.raises(ConfigError, match="consumers: directory is missing"):
            load_config(tree)

    def test_a_tree_without_an_approaches_directory_is_refused(self, tree: Path) -> None:
        import shutil

        shutil.rmtree(tree / "approaches")

        with pytest.raises(ConfigError, match="approaches: directory is missing"):
            load_config(tree)

    def test_a_consumers_directory_with_no_consumer_is_refused(self, tree: Path) -> None:
        for folder in (tree / "consumers").iterdir():
            for path in folder.iterdir():
                path.unlink()
            folder.rmdir()

        with pytest.raises(ConfigError, match="no consumer folder found"):
            load_config(tree)

    def test_a_service_naming_a_device_type_prices_do_not_hold_is_refused(self, tree: Path) -> None:
        edit(
            tree / "models" / "model_repository.yaml",
            {
                "models": {
                    "b3": {
                        "hugging_face_id": "owner/b3",
                        "base_model": "b3",
                        "onnx_file": "b3.onnx",
                        "sha256": "a" * 64,
                        "input_height": 224,
                        "input_width": 224,
                        "normalization_mean": [0.485, 0.456, 0.406],
                        "normalization_std": [0.229, 0.224, 0.225],
                        "classes": ["car"],
                        "memory_mb": 14.0,
                        "variant_of": None,
                    }
                }
            },
        )
        edit(
            tree / "models" / "inference_services.yaml",
            {
                "services": {
                    "infer-b3-moon": {
                        "model": "b3",
                        "device_type": "moon_lander",
                        "replicas": 1,
                    }
                }
            },
        )

        starting(tree, "infer-b3-moon")

        with pytest.raises(ConfigError, match="moon_lander"):
            load_config(tree)


class TestWhatAContractMayNotSay:
    def test_an_objective_weighing_nothing_is_refused(self, tree: Path) -> None:
        path = tree / "consumers" / "c1_navigation" / "contract.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["objectives"]["accuracy_car"]["weight"] = 0.0
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="weight"):
            load_config(tree)


class TestWhatTheLoaderMustCatchBeforeARunStarts:
    def test_a_scored_cdm_reading_a_column_no_table_holds_is_refused(self, tree: Path) -> None:
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

        with pytest.raises(ConfigError, match="unicorn_latency_ms"):
            load_config(tree)

    def test_an_objective_on_a_grouped_cdm_is_refused(self, tree: Path) -> None:
        path = tree / "consumers" / "c3_surveillance" / "contract.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["objectives"]["per_class_accuracy"] = dict(body["objectives"]["general_accuracy"])
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="per_class_accuracy"):
            load_config(tree)

    def test_one_cdm_id_with_two_definitions_across_consumers_is_refused(self, tree: Path) -> None:
        path = tree / "consumers" / "c3_surveillance" / "cdm.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["cdms"]["general_accuracy"]["window"] = 100
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="general_accuracy"):
            load_config(tree)

    def test_an_objective_states_its_weight_and_nothing_it_need_not(self, tree: Path) -> None:
        path = tree / "consumers" / "c3_surveillance" / "contract.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["objectives"] = {"general_accuracy": {"weight": 1.0}}
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        asked = load_config(tree).consumers["c3_surveillance"].contract.objectives
        held = asked["general_accuracy"]

        assert (held.min_value, held.max_value) == (0.0, 1.0)
        assert (held.direction, held.mapping, held.mapping_base) == ("maximize", "linear", None)


class TestTheSafetyConsumerScoresItsMissRate:
    def test_the_miss_rate_is_one_of_the_scored_objectives(self) -> None:
        contract = load_config(TREE).consumers["c2_safety"].contract
        miss_rate = contract.objectives["miss_rate_cyclist_as_motorcycle"]

        assert miss_rate.direction == "minimize"
        assert (miss_rate.min_value, miss_rate.max_value) == (0.0, 0.2)
        assert miss_rate.weight > 0.0

    def test_the_cost_still_weighs_against_the_whole_quality_term(self) -> None:
        contract = load_config(TREE).consumers["c2_safety"].contract
        total = sum(asked.weight for asked in contract.objectives.values())

        assert contract.cost_weight == pytest.approx(0.15 * total)

    def test_a_rising_miss_rate_lowers_the_score(self) -> None:
        contract = load_config(TREE).consumers["c2_safety"].contract
        answered: dict[str, object] = dict.fromkeys(contract.objectives, 0.9)
        low = score(contract, answered | {"miss_rate_cyclist_as_motorcycle": 0.01}, cost_per_1k=0.0)
        high = score(
            contract, answered | {"miss_rate_cyclist_as_motorcycle": 0.05}, cost_per_1k=0.0
        )

        assert isinstance(low, ScoreParts)
        assert isinstance(high, ScoreParts)
        assert low.total > high.total


class TestAContractVariant:
    def test_the_safety_consumer_carries_a_contract_without_its_miss_rate(self) -> None:
        consumer = load_config(TREE).consumers["c2_safety"]
        variant = consumer.contract_variants["without_miss_rate"]

        assert "miss_rate_cyclist_as_motorcycle" in consumer.contract.objectives
        assert "miss_rate_cyclist_as_motorcycle" not in variant.objectives

    def test_the_variant_weighs_the_classes_exactly_as_the_contract_does(self) -> None:
        consumer = load_config(TREE).consumers["c2_safety"]
        variant = consumer.contract_variants["without_miss_rate"]

        held = {name: asked.weight for name, asked in consumer.contract.objectives.items()}
        assert {name: asked.weight for name, asked in variant.objectives.items()} == {
            name: weight for name, weight in held.items() if name.startswith("accuracy_")
        }

    def test_the_variant_keeps_the_cost_to_quality_ratio_of_the_contract(self) -> None:
        consumer = load_config(TREE).consumers["c2_safety"]
        variant = consumer.contract_variants["without_miss_rate"]

        def ratio(contract: ConsumerContract) -> float:
            return contract.cost_weight / sum(one.weight for one in contract.objectives.values())

        assert ratio(variant) == pytest.approx(ratio(consumer.contract))

    def test_the_variant_differs_from_the_contract_only_in_the_miss_rate(self) -> None:
        consumer = load_config(TREE).consumers["c2_safety"]
        variant = consumer.contract_variants["without_miss_rate"]
        miss_rate = "miss_rate_cyclist_as_motorcycle"

        expected = consumer.contract.model_copy(
            update={
                "objectives": {
                    name: asked
                    for name, asked in consumer.contract.objectives.items()
                    if name != miss_rate
                },
                "requirements": tuple(
                    level for level in consumer.contract.requirements if level.cdm != miss_rate
                ),
                "cost_weight": variant.cost_weight,
            }
        )
        assert variant.model_dump() == expected.model_dump()

    def test_a_consumer_whose_folder_states_no_variant_has_none(self, tree: Path) -> None:
        assert set(load_config(tree).consumers["c1_navigation"].contract_variants) == {
            "only_general_accuracy"
        }

        for path in (tree / "consumers" / "c1_navigation").glob("contract_*.yaml"):
            path.unlink()

        assert load_config(tree).consumers["c1_navigation"].contract_variants == {}

    def test_every_consumer_carries_a_contract_declaring_the_general_accuracy_alone(self) -> None:
        for consumer_id, consumer in load_config(TREE).consumers.items():
            variant = consumer.scored_under("only_general_accuracy")

            assert set(variant.objectives) == {"general_accuracy"}, consumer_id
            assert [level.cdm for level in variant.requirements] == ["general_accuracy"], (
                consumer_id
            )

    def test_every_consumer_trades_the_general_accuracy_against_the_cost_alike(self) -> None:
        config = load_config(TREE)
        surveillance = config.consumers["c3_surveillance"].contract

        for consumer_id, consumer in config.consumers.items():
            variant = consumer.scored_under("only_general_accuracy")

            assert variant.objectives == surveillance.objectives, consumer_id
            assert variant.requirements == surveillance.requirements, consumer_id
            assert variant.cost_weight == surveillance.cost_weight, consumer_id
            assert variant.cost_range_max_per_1k == surveillance.cost_range_max_per_1k, consumer_id

    def test_the_surveillance_consumer_declares_the_same_under_cap_and_cap_star(self) -> None:
        consumer = load_config(TREE).consumers["c3_surveillance"]

        assert consumer.scored_under("only_general_accuracy") == consumer.contract

    def test_the_variant_keeps_the_classes_the_exhibits_slice_on(self) -> None:
        for consumer_id, consumer in load_config(TREE).consumers.items():
            variant = consumer.scored_under("only_general_accuracy")

            assert variant.prioritized_classes == consumer.contract.prioritized_classes, consumer_id
            assert variant.request_rate_per_s == consumer.contract.request_rate_per_s, consumer_id
            assert variant.feedback == consumer.contract.feedback, consumer_id

    def test_a_variant_scoring_a_cdm_nobody_declares_is_refused(self, tree: Path) -> None:
        path = tree / "consumers" / "c2_safety" / "contract_without_miss_rate.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["objectives"] = {"accuracy_on_unicorns": {"weight": 1.0}}
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="accuracy_on_unicorns"):
            load_config(tree)

    def test_a_variant_scoring_a_grouped_cdm_is_refused(self, tree: Path) -> None:
        path = tree / "consumers" / "c1_navigation" / "contract_only_general_accuracy.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["objectives"] = {"per_class_accuracy": {"weight": 1.0}}
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="contract_only_general_accuracy.yaml"):
            load_config(tree)

    def test_a_variant_prioritizing_a_class_the_dataset_does_not_hold_is_refused(
        self, tree: Path
    ) -> None:
        path = tree / "consumers" / "c1_navigation" / "contract_only_general_accuracy.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["prioritized_classes"] = ["unicorn"]
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="unicorn"):
            load_config(tree)

    def test_a_variant_naming_another_consumer_is_refused(self, tree: Path) -> None:
        path = tree / "consumers" / "c2_safety" / "contract_without_miss_rate.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["consumer_id"] = "c1_navigation"
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="c1_navigation"):
            load_config(tree)


class TestTheEnsembleARunStartsFrom:
    def test_every_consumer_starts_on_the_same_ensemble(self) -> None:
        held = {one.initial_ensemble for one in load_config(TREE).consumers.values()}

        assert len(held) == 1
        assert len(held.pop()) == 3

    def test_the_ensemble_it_starts_on_is_served_by_the_pool(self) -> None:
        config = load_config(TREE)
        pool = set(config.models.inference_services.services)

        for consumer_id, consumer in config.consumers.items():
            assert set(consumer.initial_ensemble) <= pool, consumer_id

    def test_the_ensemble_a_consumer_states_is_read(self, tree: Path) -> None:
        write(
            tree / "consumers" / "c1_navigation" / "initial_ensemble.yaml",
            {"members": ["infer-b5-pi5", "infer-b6-orin"]},
        )
        services(tree, "infer-b5-pi5", "infer-b6-orin")

        held = load_config(tree).consumers["c1_navigation"]

        assert held.initial_ensemble == ("infer-b5-pi5", "infer-b6-orin")

    def test_a_starting_member_the_pool_does_not_hold_is_refused(self, tree: Path) -> None:
        write(
            tree / "consumers" / "c1_navigation" / "initial_ensemble.yaml",
            {"members": ["infer-nobody"]},
        )

        with pytest.raises(ConfigError, match="infer-nobody"):
            load_config(tree)

    def test_an_ensemble_above_the_limit_of_the_control_plane_is_refused(self, tree: Path) -> None:
        names = [f"infer-{index}" for index in range(4)]
        write(tree / "consumers" / "c1_navigation" / "initial_ensemble.yaml", {"members": names})
        services(tree, *names)

        with pytest.raises(ConfigError, match="max_ensemble_size"):
            load_config(tree)

    def test_an_ensemble_naming_one_service_twice_is_refused(self, tree: Path) -> None:
        write(
            tree / "consumers" / "c1_navigation" / "initial_ensemble.yaml",
            {"members": ["infer-a", "infer-a"]},
        )
        services(tree, "infer-a")

        with pytest.raises(ConfigError, match="infer-a"):
            load_config(tree)

    def test_an_ensemble_of_no_member_is_refused(self, tree: Path) -> None:
        write(tree / "consumers" / "c1_navigation" / "initial_ensemble.yaml", {"members": []})

        with pytest.raises(ConfigError, match="members"):
            load_config(tree)


class TestWhichContractARunScores:
    def test_the_committed_tree_scores_every_consumer_on_its_own_contract(self) -> None:
        config = load_config(TREE)

        assert config.services.control_plane.contract is None
        for consumer in config.consumers.values():
            assert consumer.scored_under(None) == consumer.contract

    def test_naming_the_cap_star_contract_scores_every_consumer_under_it(self, tree: Path) -> None:
        edit(tree / "services" / "control_plane.yaml", {"contract": "only_general_accuracy"})

        config = load_config(tree)

        assert config.services.control_plane.contract == "only_general_accuracy"
        for consumer_id, consumer in config.consumers.items():
            scored = consumer.scored_under("only_general_accuracy")
            assert set(scored.objectives) == {"general_accuracy"}, consumer_id

    def test_an_approach_of_one_consumer_leaves_the_others_on_their_own_contract(
        self, tree: Path
    ) -> None:
        edit(tree / "services" / "control_plane.yaml", {"contract": "without_miss_rate"})

        consumers = load_config(tree).consumers

        safety = consumers["c2_safety"].scored_under("without_miss_rate")
        assert "miss_rate_cyclist_as_motorcycle" not in safety.objectives
        for consumer_id in ("c1_navigation", "c3_surveillance"):
            consumer = consumers[consumer_id]
            assert consumer.scored_under("without_miss_rate") == consumer.contract

    def test_a_contract_no_consumer_states_is_refused(self, tree: Path) -> None:
        edit(tree / "services" / "control_plane.yaml", {"contract": "only_the_cheapest"})

        with pytest.raises(ConfigError, match="only_the_cheapest"):
            load_config(tree)


class TestTheBackbonesThePoolIsBuiltFrom:
    def recipe(self) -> VariationRecipeConfig:
        return load_config(TREE).models.variations

    def test_the_recipe_carries_the_numbers_the_probe_measured(self) -> None:
        held = self.recipe()

        assert held.train_images_per_class == 250
        assert held.specialist_class_weight == 1000.0
        assert held.variations_per_base_model == 2
        assert held.frozen_base_model is True

    def test_the_pool_is_the_sixteen_services_the_article_reports(self) -> None:
        held = self.recipe()

        assert len(held.base_models) * held.variations_per_base_model == 16

    def test_every_backbone_names_the_repository_and_the_file_it_is_served_as(self) -> None:
        if not (TREE / "local" / "models.yaml").is_file():
            pytest.skip("config/local/models.yaml is not here; run: tools/new_local.py")
        for name, base in load_config(TREE).models.variations.base_models.items():
            assert base.hugging_face_id is not None, name
            assert base.onnx_file is not None, name
            assert base.hugging_face_id.count("/") == 1, name
            assert base.onnx_file.endswith(".onnx"), name

    def test_every_backbone_carries_the_preprocessing_it_was_trained_with(self) -> None:
        held = self.recipe().base_models

        assert held["b6"].normalization_mean == (0.485, 0.456, 0.406)
        assert held["b6"].normalization_std == (0.229, 0.224, 0.225)
        assert held["b7"].normalization_mean == (0.5, 0.5, 0.5)
        assert held["b7"].normalization_std == (0.5, 0.5, 0.5)

    def test_every_backbone_takes_the_same_image_size(self) -> None:
        for name, base in self.recipe().base_models.items():
            assert (base.input_height, base.input_width) == (224, 224), name

    def test_no_two_backbones_are_the_same_file(self) -> None:
        if not (TREE / "local" / "models.yaml").is_file():
            pytest.skip("config/local/models.yaml is not here; run: tools/new_local.py")
        held = self.recipe().base_models

        assert len({one.sha256 for one in held.values()}) == len(held)

    def test_the_backbones_span_the_sizes_an_edge_cluster_has(self) -> None:
        held = self.recipe().base_models

        assert len(held) == 8
        assert min(one.weights_mb for one in held.values()) < 15.0
        assert max(one.weights_mb for one in held.values()) > 250.0

    def test_one_backbone_is_not_a_convolution(self) -> None:
        held = self.recipe().base_models

        assert any(one.family == "transformer" for one in held.values())


class TestTheRuntimeItServesOn:
    def test_only_cuda_and_the_cpu_may_be_named(self) -> None:
        from eemls.config.services import PROVIDERS

        assert PROVIDERS == ("CUDAExecutionProvider", "CPUExecutionProvider")

    def test_every_device_type_falls_back_to_the_cpu(self) -> None:
        for device_type, chain in load_config(TREE).services.inference_service.providers.items():
            assert chain[-1] == "CPUExecutionProvider", device_type

    def test_a_gpu_device_tries_cuda_before_the_cpu(self) -> None:
        chains = load_config(TREE).services.inference_service.providers

        assert chains["x86_gpu"] == ("CUDAExecutionProvider", "CPUExecutionProvider")
        assert chains["jetson"] == ("CUDAExecutionProvider", "CPUExecutionProvider")

    def test_a_cpu_device_names_the_cpu_alone(self) -> None:
        chains = load_config(TREE).services.inference_service.providers

        assert chains["x86_cpu"] == ("CPUExecutionProvider",)
        assert chains["arm_cpu"] == ("CPUExecutionProvider",)

    def test_a_provider_this_cluster_does_not_have_is_refused(self, tree: Path) -> None:
        edit(
            tree / "services" / "inference_service.yaml",
            {"providers": {"x86_gpu": ["TensorrtExecutionProvider", "CPUExecutionProvider"]}},
        )

        with pytest.raises(ConfigError, match="Tensorrt"):
            load_config(tree)


TABLE_1_MEMORY_GB = {
    "precision_5820": 32,
    "hp_z240": 16,
    "jetson_agx_orin": 64,
    "jetson_agx_xavier": 16,
    "jetson_orin_nano": 8,
    "jetson_nano": 4,
    "raspberry_pi5": 8,
    "raspberry_pi4": 4,
    "beelink_ser8": 32,
    "beelink_bt3": 4,
}


class TestWhatTheDeploymentReads:
    def test_every_device_has_the_memory_table_1_of_the_article_gives(self) -> None:
        devices = load_config(TREE).cost.devices

        assert {name: device.memory_gb for name, device in devices.items()} == (TABLE_1_MEMORY_GB)

    def test_every_execution_provider_has_an_image(self) -> None:
        config = load_config(TREE)
        providers = {device.execution_provider for device in config.cost.devices.values()}

        assert providers <= set(config.services.deployment.image_tags)

    def test_a_device_class_with_no_image_is_refused_by_name(self, tree: Path) -> None:
        edit(tree / "services" / "deployment.yaml", {"image_tags": {"x86_cpu": "cpu-amd64"}})

        with pytest.raises(ConfigError, match="precision_5820 .* x86_gpu, .* gives no image"):
            load_config(tree)

    @pytest.mark.parametrize("provider", ["x86_cpu", "x86_gpu", "arm_cpu", "jetson"])
    def test_a_copy_needs_its_runtime_and_its_model_in_memory(self, provider: str) -> None:
        deployment = load_config(TREE).services.deployment

        assert deployment.copy_memory_mb(
            model_memory_mb=100.0, execution_provider=provider
        ) == pytest.approx(
            deployment.runtime_memory_mb[provider] + deployment.model_memory_factor * 100.0
        )

    def test_a_gpu_runtime_needs_more_memory_than_a_cpu_runtime(self) -> None:
        runtime = load_config(TREE).services.deployment.runtime_memory_mb

        assert min(runtime["jetson"], runtime["x86_gpu"]) > max(
            runtime["arm_cpu"], runtime["x86_cpu"]
        )

    def test_a_provider_with_an_image_and_no_runtime_memory_is_refused(self, tree: Path) -> None:
        edit(tree / "services" / "deployment.yaml", {"runtime_memory_mb": {"x86_cpu": 256}})

        with pytest.raises(ConfigError, match="runtime_memory_mb .*arm_cpu"):
            load_config(tree)

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("runtime_memory_mb", {"x86_cpu": 0, "x86_gpu": 1024, "arm_cpu": 256, "jetson": 1024}),
            ("model_memory_factor", 0),
        ],
    )
    def test_a_memory_figure_that_is_not_positive_is_refused(
        self, tree: Path, key: str, value: object
    ) -> None:
        edit(tree / "services" / "deployment.yaml", {key: value})

        with pytest.raises(ConfigError, match=key):
            load_config(tree)


class TestALabelLandsInsideTheWindowThatScoresIt:
    def test_the_committed_delay_fits_inside_every_window(self) -> None:
        config = load_config(TREE)
        interval = config.services.control_plane.decision_interval_queries
        approach_windows = [approach.evaluation_window for approach in config.approaches.values()]

        for consumer in config.consumers.values():
            windows = [one.window for one in consumer.cdms.values() if one.window]
            narrowest = min(windows + [window for window in approach_windows if window])
            assert consumer.contract.feedback.delay_intervals * interval < narrowest

    def test_a_delay_longer_than_the_window_is_refused_by_consumer(self, tree: Path) -> None:
        edit(tree / "services" / "control_plane.yaml", {"decision_interval_queries": 100})
        edit(tree / "services" / "consumer.yaml", {"decision_interval_queries": 100})

        with pytest.raises(
            ConfigError, match=r"c1_navigation: a label returns 1500 queries late .* holds 500"
        ):
            load_config(tree)


class TestTheCommittedRepositoryFollowsTheRecipe:
    def test_every_model_is_a_variation_of_a_recipe_backbone(self) -> None:
        models = load_config(TREE).models
        backbones = set(models.variations.base_models)

        assert {one.base_model for one in models.model_repository.models.values()} == backbones

    def test_each_backbone_has_the_variations_the_recipe_asks(self) -> None:
        models = load_config(TREE).models
        held = Counter(one.base_model for one in models.model_repository.models.values())

        assert set(held.values()) == {models.variations.variations_per_base_model}

    def test_the_variation_profiles_record_every_model_of_the_repository(self) -> None:
        models = load_config(TREE).models

        assert set(models.variation_profiles.variations) == set(models.model_repository.models)


class TestWhatTwoServicesStateTheSame:
    @pytest.mark.parametrize(
        ("own", "other", "key"),
        [
            ("consumer", "control_plane", "data_hub_url"),
            ("consumer", "control_plane", "decision_interval_queries"),
            ("inference_service", "data_hub", "max_image_bytes"),
        ],
    )
    def test_the_committed_tree_states_one_value(self, own: str, other: str, key: str) -> None:
        services = load_config(TREE).services

        assert getattr(getattr(services, own), key) == getattr(getattr(services, other), key)

    @pytest.mark.parametrize(
        ("own", "key", "value"),
        [
            ("consumer", "data_hub_url", "http://elsewhere:8081"),
            ("consumer", "decision_interval_queries", 7),
            ("inference_service", "max_image_bytes", 1024),
        ],
    )
    def test_a_value_one_service_states_differently_is_refused(
        self, tree: Path, own: str, key: str, value: object
    ) -> None:
        edit(tree / "services" / f"{own}.yaml", {key: value})

        with pytest.raises(ConfigError, match=f"{own} {key} .* is not the"):
            load_config(tree)


class TestTheObservationServiceAddress:
    def test_the_committed_tree_names_the_port_it_serves_on(self) -> None:
        held = load_config(TREE).services.observation_service

        assert held.url.port == held.port

    def test_an_address_on_another_port_is_refused(self, tree: Path) -> None:
        edit(
            tree / "services" / "observation_service.yaml",
            {"url": "http://observation-service:9999"},
        )

        with pytest.raises(ConfigError, match="observation_service url .* is not on port 8083"):
            load_config(tree)

    def test_a_deployment_with_no_memory_for_it_is_refused(self, tree: Path) -> None:
        path = tree / "services" / "deployment.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["service_memory_mb"].pop("observation_service")
        write(path, body)

        with pytest.raises(ConfigError, match="names no memory for \\['observation_service'\\]"):
            load_config(tree)
