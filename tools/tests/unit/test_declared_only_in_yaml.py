from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from eemls.ensemble_optimization.approaches.cap.approach import CapSettings
from eemls.evaluation_engine.engine import EvaluationEngine
from eemls.evaluation_engine.window import Window
from eemls.metrics_database.schema import check_coverage
from eemls.testing.trees import committed_tree
from eemls_tools.config_tree import ConfigError, load_config
from support.repository import ROOT, sources

ROWS: tuple[dict[str, Any], ...] = (
    {
        "true_label": "cyclist",
        "predicted_label": "cyclist",
        "response_time_ms": 150.0,
        "cost_per_1k": 10.5,
        "throughput_rps": 18.0,
    },
    {
        "true_label": "cyclist",
        "predicted_label": "motorcycle",
        "response_time_ms": 400.0,
        "cost_per_1k": 10.5,
        "throughput_rps": 18.0,
    },
    {
        "true_label": "car",
        "predicted_label": "car",
        "response_time_ms": 120.0,
        "cost_per_1k": 10.5,
        "throughput_rps": 24.0,
    },
    {
        "true_label": "bus",
        "predicted_label": "car",
        "response_time_ms": 500.0,
        "cost_per_1k": 10.5,
        "throughput_rps": 24.0,
    },
)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    return committed_tree(tmp_path / "config")


def declare(
    tree: Path, consumer: str, name: str, specification: dict[str, Any], **extra: Any
) -> None:
    path = tree / "consumers" / consumer / "cdm.yaml"
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body["cdms"][name] = {
        "measured": "per_window",
        "window": 500,
        "unit": "ratio",
        "description": f"The metric {name}, declared for this study",
        "specification": specification,
    } | extra
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")


def value_of(tree: Path, consumer: str, name: str) -> object:
    declared = load_config(tree).consumers[consumer].cdms
    from eemls.domain.consumer_defined_metrics import CdmSet

    return EvaluationEngine(CdmSet.model_validate({"definitions": dict(declared)})).of(
        name, Window(ROWS)
    )


class TestANewMetricNeedsNoCode:
    def test_a_plain_count_with_two_filters(self, tree: Path) -> None:
        declare(
            tree,
            "c2_safety",
            "cyclists_read_as_motorcycles",
            {
                "function": "count",
                "scopes": [
                    {"attribute": "true_label", "value": "cyclist"},
                    {"attribute": "predicted_label", "value": "motorcycle"},
                ],
            },
            unit="count",
        )

        assert value_of(tree, "c2_safety", "cyclists_read_as_motorcycles") == 1

    def test_a_ratio_the_study_invents(self, tree: Path) -> None:
        declare(
            tree,
            "c1_navigation",
            "slow_query_share",
            {
                "function": "ratio",
                "inputs": [
                    {
                        "function": "count",
                        "scopes": [{"attribute": "response_time_ms", "observed": True}],
                    },
                    {"function": "count"},
                ],
            },
        )

        assert value_of(tree, "c1_navigation", "slow_query_share") == pytest.approx(1.0)

    def test_a_metric_built_from_another_declared_metric(self, tree: Path) -> None:
        declare(
            tree,
            "c3_surveillance",
            "accuracy_again",
            {"function": "reference", "cdm": "general_accuracy"},
        )

        both = (
            value_of(tree, "c3_surveillance", "accuracy_again"),
            value_of(tree, "c3_surveillance", "general_accuracy"),
        )

        assert both[0] == both[1] == pytest.approx(0.5)

    def test_a_reference_narrowed_by_its_own_filter(self, tree: Path) -> None:
        declare(
            tree,
            "c3_surveillance",
            "accuracy_on_cyclists",
            {
                "function": "reference",
                "cdm": "general_accuracy",
                "scopes": [{"attribute": "true_label", "value": "cyclist"}],
            },
        )

        assert value_of(tree, "c3_surveillance", "accuracy_on_cyclists") == pytest.approx(0.5)

    def test_a_metric_grouped_by_a_column(self, tree: Path) -> None:
        declare(
            tree,
            "c1_navigation",
            "mean_time_per_class",
            {"function": "average", "attribute": "response_time_ms", "group_by": "true_label"},
            unit="milliseconds",
        )

        assert value_of(tree, "c1_navigation", "mean_time_per_class") == {
            "bus": pytest.approx(500.0),
            "car": pytest.approx(120.0),
            "cyclist": pytest.approx(275.0),
        }

    def test_a_metric_with_weights_a_consumer_chooses(self, tree: Path) -> None:
        declare(
            tree,
            "c2_safety",
            "my_own_weighting",
            {
                "function": "weighted_count",
                "group_by": "true_label",
                "weights": {"cyclist": 5.0, "car": 1.0, "bus": 1.0},
            },
            unit="count",
        )

        assert value_of(tree, "c2_safety", "my_own_weighting") == pytest.approx(5.0 * 2 + 1 + 1)

    def test_a_percentile_the_study_asks_for(self, tree: Path) -> None:
        declare(
            tree,
            "c1_navigation",
            "response_time_p99_ms",
            {"function": "percentile", "attribute": "response_time_ms", "percentile_value": 99.0},
            unit="milliseconds",
        )

        assert value_of(tree, "c1_navigation", "response_time_p99_ms") == pytest.approx(497.0)

    def test_a_metric_the_whole_tree_still_validates_with(self, tree: Path) -> None:
        declare(
            tree,
            "c1_navigation",
            "queries_seen",
            {"function": "count"},
            unit="count",
        )

        assert load_config(tree).consumers["c1_navigation"].cdms["queries_seen"].unit == "count"


class TestAWrongDeclarationStopsTheRun:
    def test_a_function_the_engine_does_not_have_is_refused(self, tree: Path) -> None:
        declare(tree, "c1_navigation", "invented", {"function": "sorcery"})

        with pytest.raises(ConfigError, match="sorcery"):
            load_config(tree)

    def test_a_metric_reading_a_column_no_table_holds_is_refused(self, tree: Path) -> None:
        declare(
            tree, "c1_navigation", "unicorns", {"function": "average", "attribute": "unicorn_count"}
        )

        with pytest.raises(Exception, match="unicorn_count"):
            check_coverage(
                __import__(
                    "eemls.domain.consumer_defined_metrics", fromlist=["CdmSet"]
                ).CdmSet.model_validate(
                    {"definitions": dict(load_config(tree).consumers["c1_navigation"].cdms)}
                )
            )

    def test_a_reference_to_a_metric_nothing_declares_is_refused(self, tree: Path) -> None:
        declare(tree, "c1_navigation", "borrowed", {"function": "reference", "cdm": "nothing"})

        with pytest.raises(ConfigError, match="nothing"):
            load_config(tree)

    def test_a_metric_a_contract_scores_but_nobody_declares_is_refused(self, tree: Path) -> None:
        path = tree / "consumers" / "c1_navigation" / "contract.yaml"
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        body["objectives"]["not_declared"] = body["objectives"]["accuracy_car"]
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        with pytest.raises(ConfigError, match="not_declared"):
            load_config(tree)


class TestAnArmOfTheArticleNeedsNoCode:
    def test_one_policy_of_the_cap_kind_serves_every_cap_approach(self) -> None:
        config = load_config(ROOT / "config")

        assert [name for name, one in config.approaches.items() if one.kind == "cap"] == ["cap"]

    def test_the_cap_policy_states_no_objective_of_its_own(self) -> None:
        assert "objective" not in CapSettings.model_fields

    def test_each_approach_is_one_contract_beside_the_one_it_varies(self) -> None:
        consumers = load_config(ROOT / "config").consumers

        assert set(consumers["c1_navigation"].contract_variants) == {"only_general_accuracy"}
        assert set(consumers["c2_safety"].contract_variants) == {
            "only_general_accuracy",
            "without_miss_rate",
        }
        assert not consumers["c3_surveillance"].contract_variants

    def test_no_module_names_an_approach(self) -> None:
        approaches = ("cap_star", "cap_minus", "only_general_accuracy", "without_miss_rate")
        named = sorted(
            f"{path.relative_to(ROOT)}:{number}"
            for source in sources()
            for path in source.rglob("*.py")
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            for approach in approaches
            if approach in line
        )

        assert named == []


class TestTheWholeEnsembleActionBelongsToTheBaselines:
    def test_no_module_of_cap_takes_it(self) -> None:
        named = sorted(
            f"{path.relative_to(ROOT)}:{number}"
            for source in sources()
            for path in (source / "ensemble_optimization" / "approaches" / "cap").rglob("*.py")
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if 'kind="select"' in line
        )

        assert named == []

    def test_the_four_kinds_cap_takes_are_the_ones_it_names(self) -> None:
        taken = {
            line.split('kind="')[1].split('"')[0]
            for source in sources()
            for path in (source / "ensemble_optimization" / "approaches" / "cap").rglob("*.py")
            for line in path.read_text(encoding="utf-8").splitlines()
            if 'kind="' in line
        }

        assert taken == {"keep", "add", "remove", "replace"}
