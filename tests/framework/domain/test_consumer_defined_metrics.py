from __future__ import annotations

from datetime import UTC, datetime

import pytest
import yaml
from pydantic import ValidationError

from eemls.domain.consumer_defined_metrics import CdmDefinition, CdmSet, MetricSpecification, Scope
from eemls.domain.contract import ConsumerContract, Objective, Requirement
from eemls.testing.trees import TREE

CONSUMERS = TREE / "consumers"
NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)


def count(**changes: object) -> MetricSpecification:
    return MetricSpecification.model_validate({"function": "count"} | changes)


class TestEveryCommittedCdmParses:
    @pytest.mark.parametrize("folder", sorted(path.name for path in CONSUMERS.iterdir()))
    def test_every_declared_cdm_of_every_consumer_builds(self, folder: str) -> None:
        body = yaml.safe_load((CONSUMERS / folder / "cdm.yaml").read_text(encoding="utf-8"))

        declared = CdmSet.model_validate({"definitions": body["cdms"]})

        assert declared.definitions
        for name, definition in declared.definitions.items():
            assert definition.name == name

    def test_the_fields_every_consumer_reads_are_collected(self) -> None:
        body = yaml.safe_load((CONSUMERS / "c2_safety" / "cdm.yaml").read_text(encoding="utf-8"))

        declared = CdmSet.model_validate({"definitions": body["cdms"]})

        assert declared.attributes() == {
            "cost_per_1k",
            "predicted_label",
            "response_time_ms",
            "throughput_rps",
            "true_label",
        }


class TestAnAggregate:
    @pytest.mark.parametrize("function", ["sum", "average", "min", "max"])
    def test_it_needs_the_field_it_reads(self, function: str) -> None:
        with pytest.raises(ValidationError, match="needs an attribute"):
            MetricSpecification.model_validate({"function": function})

    def test_counting_needs_no_field(self) -> None:
        assert count().attribute is None

    def test_it_may_not_carry_operands(self) -> None:
        with pytest.raises(ValidationError, match="takes no inputs"):
            MetricSpecification(function="average", attribute="response_time_ms", inputs=(count(),))

    def test_an_unknown_function_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="function"):
            MetricSpecification.model_validate({"function": "sorcery"})


class TestAPercentile:
    def test_it_needs_which_percentile(self) -> None:
        with pytest.raises(ValidationError, match="percentile_value"):
            MetricSpecification(function="percentile", attribute="response_time_ms")

    @pytest.mark.parametrize("value", [-1.0, 0.0, 100.5])
    def test_a_percentile_outside_the_range_is_refused(self, value: float) -> None:
        with pytest.raises(ValidationError):
            MetricSpecification(
                function="percentile", attribute="response_time_ms", percentile_value=value
            )

    def test_only_a_percentile_takes_a_percentile_value(self) -> None:
        with pytest.raises(ValidationError, match="percentile_value"):
            MetricSpecification(
                function="average", attribute="response_time_ms", percentile_value=95.0
            )


class TestARatio:
    def test_it_needs_exactly_two_operands(self) -> None:
        with pytest.raises(ValidationError, match="exactly two inputs"):
            MetricSpecification(function="ratio", inputs=(count(),))

    def test_three_operands_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="exactly two inputs"):
            MetricSpecification(function="ratio", inputs=(count(), count(), count()))

    def test_it_reads_no_field_of_its_own(self) -> None:
        with pytest.raises(ValidationError, match="reads no attribute"):
            MetricSpecification(
                function="ratio", attribute="response_time_ms", inputs=(count(), count())
            )

    def test_two_counts_make_an_accuracy(self) -> None:
        specification = MetricSpecification(
            function="ratio",
            inputs=(
                count(scopes=({"attribute": "predicted_label", "same_as": "true_label"},)),
                count(scopes=({"attribute": "true_label", "observed": True},)),
            ),
        )

        assert len(specification.inputs) == 2


class TestAWeightedCount:
    def test_it_needs_weights_and_the_field_to_group_by(self) -> None:
        with pytest.raises(ValidationError, match="weights"):
            MetricSpecification(function="weighted_count", group_by="true_label")

    def test_it_needs_the_field_to_group_by(self) -> None:
        with pytest.raises(ValidationError, match="group_by"):
            MetricSpecification(function="weighted_count", weights={"car": 4.0})

    def test_a_negative_weight_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            MetricSpecification(
                function="weighted_count", group_by="true_label", weights={"car": -1.0}
            )

    def test_only_a_weighted_count_takes_weights(self) -> None:
        with pytest.raises(ValidationError, match="weights"):
            count(weights={"car": 1.0})

    def test_it_answers_one_number_though_it_groups(self) -> None:
        specification = MetricSpecification(
            function="weighted_count", group_by="true_label", weights={"car": 4.0}
        )

        assert not specification.answers_one_value_per_group()


class TestAGroupedMetric:
    def test_a_grouped_count_answers_one_value_per_group(self) -> None:
        assert count(group_by="true_label").answers_one_value_per_group()

    def test_an_ungrouped_count_answers_one_number(self) -> None:
        assert not count().answers_one_value_per_group()


class TestAReference:
    def test_a_reference_must_name_a_cdm(self) -> None:
        with pytest.raises(ValidationError, match="names no cdm"):
            MetricSpecification(function="reference")

    def test_a_reference_that_names_nothing_declared_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="accuracy_on_unicorns"):
            CdmSet.model_validate(
                {
                    "definitions": {
                        "borrowed": {
                            "name": "borrowed",
                            "measured": "per_window",
                            "window": 500,
                            "unit": "ratio",
                            "description": "Built from one that does not exist",
                            "specification": {
                                "function": "reference",
                                "cdm": "accuracy_on_unicorns",
                            },
                        }
                    }
                }
            )

    def test_a_reference_to_a_declared_cdm_is_accepted(self) -> None:
        declared = CdmSet.model_validate(
            {
                "definitions": {
                    "general_accuracy": {
                        "name": "general_accuracy",
                        "measured": "per_window",
                        "window": 500,
                        "unit": "ratio",
                        "description": "Share of labeled queries the result matched",
                        "specification": {
                            "function": "ratio",
                            "inputs": [
                                {
                                    "function": "count",
                                    "scopes": [
                                        {"attribute": "predicted_label", "same_as": "true_label"}
                                    ],
                                },
                                {
                                    "function": "count",
                                    "scopes": [{"attribute": "true_label", "observed": True}],
                                },
                            ],
                        },
                    },
                    "borrowed": {
                        "name": "borrowed",
                        "measured": "per_window",
                        "window": 500,
                        "unit": "ratio",
                        "description": "The same accuracy under another name",
                        "specification": {"function": "reference", "cdm": "general_accuracy"},
                    },
                }
            }
        )

        assert declared.definitions["borrowed"].specification.cdm == "general_accuracy"


class TestAContext:
    def test_it_needs_a_statement(self) -> None:
        with pytest.raises(ValidationError, match="states nothing"):
            Scope(attribute="true_label")

    def test_two_statements_at_once_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="states more than one"):
            Scope(attribute="true_label", value="cyclist", observed=True)

    def test_an_empty_set_of_values_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="no value"):
            Scope(attribute="true_label", one_of=())

    def test_it_names_the_field_it_reads(self) -> None:
        assert Scope(attribute="true_label", value="cyclist").attribute == "true_label"

    def test_comparing_two_fields_names_both(self) -> None:
        assert Scope(attribute="predicted_label", same_as="true_label").same_as == ("true_label")


class TestADefinition:
    def definition(self, **changes: object) -> CdmDefinition:
        attributes: dict[str, object] = {
            "name": "general_accuracy",
            "measured": "per_window",
            "window": 500,
            "unit": "ratio",
            "description": "Share of labeled queries the result matched",
            "specification": {"function": "count"},
        }
        return CdmDefinition.model_validate(attributes | changes)

    def test_a_period_metric_without_a_window_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="window"):
            self.definition(window=None)

    def test_a_request_metric_needs_no_window(self) -> None:
        assert self.definition(measured="per_query", window=None).window is None

    def test_a_negative_window_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            self.definition(window=-1)

    def test_a_definition_without_a_description_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            self.definition(description="")

    def test_it_collects_every_field_its_expression_reads(self) -> None:
        definition = self.definition(
            specification={
                "function": "ratio",
                "inputs": [
                    {"function": "average", "attribute": "response_time_ms"},
                    {
                        "function": "count",
                        "scopes": [{"attribute": "true_label", "observed": True}],
                    },
                ],
            }
        )

        assert definition.attributes() == {"response_time_ms", "true_label"}


class TestAnObjective:
    def objective(self, **changes: object) -> Objective:
        attributes: dict[str, object] = {
            "min_value": 0.0,
            "max_value": 1.0,
            "weight": 1.0,
            "direction": "maximize",
            "mapping": "linear",
            "mapping_base": None,
        }
        return Objective.model_validate(attributes | changes)

    def test_a_range_of_no_width_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="not below max_value"):
            self.objective(max_value=0.0)

    def test_a_linear_objective_carrying_a_base_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="log mapping"):
            self.objective(mapping_base=2.0)

    def test_a_log_objective_without_a_base_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="needs its mapping_base"):
            self.objective(mapping="log")

    def test_a_negative_weight_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            self.objective(weight=-1.0)


class TestARequirement:
    def requirement(self, **changes: object) -> Requirement:
        attributes: dict[str, object] = {
            "cdm": "accuracy_car_truck_bus",
            "operator": ">=",
            "target_value": 0.8,
        }
        return Requirement.model_validate(attributes | changes)

    def test_it_states_the_cdm_the_operator_and_the_target(self) -> None:
        level = self.requirement()

        assert (level.cdm, level.operator, level.target_value) == (
            "accuracy_car_truck_bus",
            ">=",
            0.8,
        )

    def test_an_unknown_operator_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            self.requirement(operator="=~")

    def test_a_share_above_one_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            self.requirement(min_share_of_blocks=1.5)


class TestAContract:
    def contract(self, **changes: object) -> ConsumerContract:
        attributes: dict[str, object] = {
            "consumer_id": "c1_navigation",
            "description": "Counts and classifies vehicles",
            "prioritized_classes": ("car", "truck", "bus"),
            "request_rate_per_s": 20.0,
            "objectives": {
                "accuracy_car": {
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "weight": 1.0,
                    "direction": "maximize",
                    "mapping": "linear",
                    "mapping_base": None,
                }
            },
            "cost_weight": 0.15,
            "cost_range_max_per_1k": 30.0,
            "feedback": {"fraction": 0.5, "delay_intervals": 15},
        }
        return ConsumerContract.model_validate(attributes | changes)

    def test_it_holds_from_the_start_of_the_run_by_default(self) -> None:
        contract = self.contract()

        assert contract.valid_from is None
        assert contract.valid_until is None

    def test_a_contract_that_ends_before_it_starts_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="before"):
            self.contract(valid_from=NOW, valid_until=NOW.replace(hour=11))

    def test_a_contract_with_no_objective_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="objectives is empty"):
            self.contract(objectives={})

    def test_the_prioritized_classes_cannot_repeat(self) -> None:
        with pytest.raises(ValidationError, match="more than once"):
            self.contract(prioritized_classes=("car", "car"))

    def test_it_names_every_cdm_it_scores(self) -> None:
        assert self.contract().scored() == {"accuracy_car"}


class TestTheLastGuards:
    def test_a_definition_whose_name_disagrees_with_its_key_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="calls itself"):
            CdmSet.model_validate(
                {
                    "definitions": {
                        "general_accuracy": {
                            "name": "other_accuracy",
                            "measured": "per_query",
                            "window": None,
                            "unit": "ratio",
                            "description": "Mislabeled",
                            "specification": {"function": "count"},
                        }
                    }
                }
            )

    def test_a_set_given_something_that_is_not_a_mapping_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            CdmSet.model_validate({"definitions": "general_accuracy"})

    def test_an_empty_set_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="definitions is empty"):
            CdmSet.model_validate({"definitions": {}})

    def test_a_nested_reference_is_collected_from_under_a_ratio(self) -> None:
        specification = MetricSpecification.model_validate(
            {
                "function": "ratio",
                "inputs": [
                    {"function": "reference", "cdm": "general_accuracy"},
                    {"function": "count"},
                ],
            }
        )

        assert specification.references() == {"general_accuracy"}

    def test_a_set_reports_every_field_of_every_definition(self) -> None:
        declared = CdmSet.model_validate(
            {
                "definitions": {
                    "mean_time": {
                        "measured": "per_query",
                        "window": None,
                        "unit": "milliseconds",
                        "description": "Mean response time",
                        "specification": {"function": "average", "attribute": "response_time_ms"},
                    },
                    "slowest": {
                        "measured": "per_window",
                        "window": 500,
                        "unit": "queries_per_second",
                        "description": "Throughput of the slowest member",
                        "specification": {"function": "min", "attribute": "throughput_rps"},
                    },
                }
            }
        )

        assert declared.attributes() == {"response_time_ms", "throughput_rps"}

    def test_an_aggregate_that_names_a_cdm_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="names no cdm"):
            MetricSpecification(function="count", cdm="general_accuracy")

    def test_a_filter_comparing_a_field_with_itself_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="with itself"):
            Scope(attribute="true_label", same_as="true_label")
