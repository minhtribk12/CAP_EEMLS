from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import yaml

from eemls.config.consumers import load_consumers
from eemls.config.reading import Reader
from eemls.domain.consumer_defined_metrics import (
    READS_AN_ATTRIBUTE,
    CdmDefinition,
    CdmSet,
    MetricSpecification,
)
from eemls.evaluation_engine.aggregation_functions import OF_NUMBERS, CdmError, Undefined, aggregate
from eemls.evaluation_engine.contract_scoring import ScoreParts, score
from eemls.evaluation_engine.engine import EvaluationEngine
from eemls.evaluation_engine.window import Window
from eemls.testing.trees import TREE

CONSUMERS = TREE / "consumers"

ROWS: tuple[dict[str, Any], ...] = (
    {
        "true_label": "car",
        "predicted_label": "car",
        "response_time_ms": 150.0,
        "cost_per_1k": 10.5,
        "throughput_rps": 24.0,
    },
    {
        "true_label": "car",
        "predicted_label": "truck",
        "response_time_ms": 170.0,
        "cost_per_1k": 10.5,
        "throughput_rps": 24.0,
    },
    {
        "true_label": "cyclist",
        "predicted_label": "motorcycle",
        "response_time_ms": 200.0,
        "cost_per_1k": 10.5,
        "throughput_rps": 18.0,
    },
    {
        "true_label": "cyclist",
        "predicted_label": "cyclist",
        "response_time_ms": 190.0,
        "cost_per_1k": 10.5,
        "throughput_rps": 18.0,
    },
    {
        "true_label": "bus",
        "predicted_label": "bus",
        "response_time_ms": 160.0,
        "cost_per_1k": 10.5,
        "throughput_rps": 30.0,
    },
    {
        "true_label": None,
        "predicted_label": "truck",
        "response_time_ms": 180.0,
        "cost_per_1k": 10.5,
        "throughput_rps": 24.0,
    },
)


QUERIES_ONLY = CdmSet.model_validate(
    {
        "definitions": {
            "queries": {
                "measured": "per_window",
                "window": 500,
                "unit": "count",
                "description": "How many queries the window holds",
                "specification": {"function": "count"},
            }
        }
    }
)


def evaluate(specification: dict[str, Any], rows: tuple[dict[str, Any], ...] = ROWS) -> object:
    probe = CdmDefinition.model_validate(
        {
            "name": "probe",
            "measured": "per_query",
            "window": None,
            "unit": "count",
            "description": "The metric under test",
            "specification": specification,
        }
    )
    declared = CdmSet(definitions={**QUERIES_ONLY.definitions, "probe": probe})
    return EvaluationEngine(declared).of("probe", Window(rows))


class TestEveryFunction:
    def test_every_function_reading_an_attribute_has_one_implementation(self) -> None:
        assert set(OF_NUMBERS) == set(READS_AN_ATTRIBUTE)

    def test_count_counts_the_rows_that_pass_its_filters(self) -> None:
        assert evaluate({"function": "count"}) == 6
        assert (
            evaluate({"function": "count", "scopes": [{"attribute": "true_label", "value": "car"}]})
            == 2
        )

    def test_count_counts_only_the_rows_that_hold_the_field_when_asked(self) -> None:
        assert (
            evaluate(
                {"function": "count", "scopes": [{"attribute": "true_label", "observed": True}]}
            )
            == 5
        )

    def test_sum_adds_the_field_it_reads(self) -> None:
        assert evaluate({"function": "sum", "attribute": "response_time_ms"}) == pytest.approx(
            1050.0
        )

    def test_avg_is_the_mean_of_the_field(self) -> None:
        assert evaluate({"function": "average", "attribute": "response_time_ms"}) == pytest.approx(
            175.0
        )

    def test_min_is_the_smallest(self) -> None:
        assert evaluate({"function": "min", "attribute": "throughput_rps"}) == pytest.approx(18.0)

    def test_max_is_the_largest(self) -> None:
        assert evaluate({"function": "max", "attribute": "response_time_ms"}) == pytest.approx(
            200.0
        )

    def test_a_percentile_interpolates_between_the_two_nearest(self) -> None:
        assert evaluate(
            {"function": "percentile", "attribute": "response_time_ms", "percentile_value": 50.0}
        ) == pytest.approx(175.0)
        assert evaluate(
            {"function": "percentile", "attribute": "response_time_ms", "percentile_value": 95.0}
        ) == pytest.approx(197.5)

    def test_a_ratio_divides_its_first_operand_by_its_second(self) -> None:
        value = evaluate(
            {
                "function": "ratio",
                "inputs": [
                    {
                        "function": "count",
                        "scopes": [{"attribute": "true_label", "value": "car"}],
                    },
                    {"function": "count"},
                ],
            }
        )

        assert value == pytest.approx(2 / 6)

    def test_a_weighted_count_weighs_each_class_by_what_the_consumer_asked(self) -> None:
        value = evaluate(
            {
                "function": "weighted_count",
                "group_by": "true_label",
                "weights": {"car": 4.0, "cyclist": 3.0, "bus": 1.0},
            }
        )

        assert value == pytest.approx(4.0 * 2 + 3.0 * 2 + 1.0 * 1)

    def test_a_weighted_count_ignores_a_class_it_was_given_no_weight_for(self) -> None:
        value = evaluate(
            {"function": "weighted_count", "group_by": "true_label", "weights": {"car": 1.0}}
        )

        assert value == pytest.approx(2.0)


class TestRatiosFiltersAndBadRecords:
    def test_a_ratio_applies_the_filters_on_its_own_node(self) -> None:
        value = evaluate(
            {
                "function": "ratio",
                "scopes": [{"attribute": "true_label", "value": "cyclist"}],
                "inputs": [
                    {
                        "function": "count",
                        "scopes": [{"attribute": "predicted_label", "value": "motorcycle"}],
                    },
                    {"function": "count"},
                ],
            }
        )

        assert value == pytest.approx(0.5)

    def test_a_reference_applies_the_filters_on_its_own_node(self) -> None:
        declared = CdmSet.model_validate(
            {
                "definitions": {
                    "queries": {
                        "measured": "per_window",
                        "window": 500,
                        "unit": "count",
                        "description": "How many queries the window holds",
                        "specification": {"function": "count"},
                    },
                    "cyclist_queries": {
                        "measured": "per_window",
                        "window": 500,
                        "unit": "count",
                        "description": "The same count, over the cyclist queries only",
                        "specification": {
                            "function": "reference",
                            "cdm": "queries",
                            "scopes": [{"attribute": "true_label", "value": "cyclist"}],
                        },
                    },
                }
            }
        )
        evaluator = EvaluationEngine(declared)

        assert evaluator.of("queries", Window(ROWS)) == 6
        assert evaluator.of("cyclist_queries", Window(ROWS)) == 2

    def test_a_division_by_zero_is_a_typed_result_and_never_a_zero(self) -> None:
        value = evaluate(
            {
                "function": "ratio",
                "inputs": [
                    {
                        "function": "count",
                        "scopes": [{"attribute": "true_label", "value": "tram"}],
                    },
                    {
                        "function": "count",
                        "scopes": [{"attribute": "true_label", "value": "tram"}],
                    },
                ],
            }
        )

        assert isinstance(value, Undefined)
        assert value != 0.0
        assert "no row" in value.reason

    def test_a_non_numeric_record_is_reported_and_named(self) -> None:
        rows = ({"response_time_ms": "fast"},)

        with pytest.raises(CdmError, match="response_time_ms"):
            evaluate({"function": "average", "attribute": "response_time_ms"}, rows)


class TestAnEmptyWindow:
    def test_a_count_over_no_row_is_zero(self) -> None:
        assert evaluate({"function": "count"}, ()) == 0

    @pytest.mark.parametrize("function", ["sum", "average", "min", "max"])
    def test_an_aggregate_over_no_row_is_undefined(self, function: str) -> None:
        value = evaluate({"function": function, "attribute": "response_time_ms"}, ())

        assert isinstance(value, Undefined)
        assert "no row" in value.reason

    def test_a_percentile_over_no_row_is_undefined(self) -> None:
        value = evaluate(
            {"function": "percentile", "attribute": "response_time_ms", "percentile_value": 95.0},
            (),
        )

        assert isinstance(value, Undefined)


class TestAReferenceAndACycle:
    def declared(self, **expressions: dict[str, Any]) -> EvaluationEngine:
        definitions = {
            name: {
                "measured": "per_window",
                "window": 500,
                "unit": "ratio",
                "description": f"The metric {name}",
                "specification": specification,
            }
            for name, specification in expressions.items()
        }
        return EvaluationEngine(CdmSet.model_validate({"definitions": definitions}))

    def test_a_reference_reads_the_metric_it_names(self) -> None:
        evaluator = self.declared(
            total={"function": "count"},
            borrowed={"function": "reference", "cdm": "total"},
        )

        assert evaluator.of("borrowed", Window(ROWS)) == 6

    def test_a_reference_nested_two_deep_resolves(self) -> None:
        evaluator = self.declared(
            total={"function": "count"},
            once={"function": "reference", "cdm": "total"},
            twice={"function": "reference", "cdm": "once"},
        )

        assert evaluator.of("twice", Window(ROWS)) == 6

    def test_a_reference_inside_a_ratio_resolves(self) -> None:
        evaluator = self.declared(
            total={"function": "count"},
            share={
                "function": "ratio",
                "inputs": [
                    {
                        "function": "count",
                        "scopes": [{"attribute": "true_label", "value": "car"}],
                    },
                    {"function": "reference", "cdm": "total"},
                ],
            },
        )

        assert evaluator.of("share", Window(ROWS)) == pytest.approx(2 / 6)

    def test_a_metric_that_refers_to_itself_is_refused_by_name(self) -> None:
        evaluator = self.declared(loop={"function": "reference", "cdm": "loop"})

        with pytest.raises(CdmError, match="loop"):
            evaluator.of("loop", Window(ROWS))

    def test_a_cycle_through_two_metrics_is_refused_and_names_the_path(self) -> None:
        evaluator = self.declared(
            first={"function": "reference", "cdm": "second"},
            second={"function": "reference", "cdm": "first"},
        )

        with pytest.raises(CdmError, match="first -> second -> first"):
            evaluator.of("first", Window(ROWS))

    def test_asking_for_a_metric_nothing_declares_is_refused_by_name(self) -> None:
        with pytest.raises(CdmError, match="accuracy_on_unicorns"):
            EvaluationEngine(QUERIES_ONLY).of("accuracy_on_unicorns", Window(ROWS))


class TestFiltersApplyEverywhere:
    @pytest.mark.parametrize(
        ("function", "extra", "over_the_car_rows"),
        [
            ("count", {}, 2),
            ("sum", {"attribute": "response_time_ms"}, 320.0),
            ("average", {"attribute": "response_time_ms"}, 160.0),
            ("min", {"attribute": "response_time_ms"}, 150.0),
            ("max", {"attribute": "response_time_ms"}, 170.0),
            ("percentile", {"attribute": "response_time_ms", "percentile_value": 50.0}, 160.0),
        ],
    )
    def test_every_aggregate_reads_only_the_rows_its_filters_pass(
        self, function: str, extra: dict[str, Any], over_the_car_rows: float
    ) -> None:
        value = evaluate(
            {"function": function, "scopes": [{"attribute": "true_label", "value": "car"}]} | extra
        )

        assert value == pytest.approx(over_the_car_rows)

    def test_a_weighted_count_honours_its_filters(self) -> None:
        value = evaluate(
            {
                "function": "weighted_count",
                "group_by": "true_label",
                "weights": {"car": 4.0, "cyclist": 3.0},
                "scopes": [{"attribute": "predicted_label", "same_as": "true_label"}],
            }
        )

        assert value == pytest.approx(4.0 * 1 + 3.0 * 1)

    def test_two_filters_both_apply(self) -> None:
        value = evaluate(
            {
                "function": "count",
                "scopes": [
                    {"attribute": "true_label", "value": "cyclist"},
                    {"attribute": "predicted_label", "value": "motorcycle"},
                ],
            }
        )

        assert value == 1

    def test_a_filter_on_a_set_of_values_applies(self) -> None:
        value = evaluate(
            {
                "function": "count",
                "scopes": [{"attribute": "true_label", "one_of": ["car", "bus"]}],
            }
        )

        assert value == 3

    def test_a_filter_on_a_field_no_row_holds_passes_nothing(self) -> None:
        assert evaluate({"function": "count", "scopes": [{"attribute": "tram", "value": "x"}]}) == 0


EVERY_CLASS: tuple[dict[str, Any], ...] = ROWS + tuple(
    {
        "true_label": label,
        "predicted_label": label,
        "response_time_ms": 160.0,
        "cost_per_1k": 10.5,
        "throughput_rps": 24.0,
    }
    for label in ("pedestrian", "truck", "motorcycle")
)


class TestEveryCdmOfThePaper:
    def declared(self, folder: str) -> EvaluationEngine:
        body = yaml.safe_load((CONSUMERS / folder / "cdm.yaml").read_text(encoding="utf-8"))
        return EvaluationEngine(CdmSet.model_validate({"definitions": body["cdms"]}))

    def test_general_accuracy(self) -> None:
        value = self.declared("c3_surveillance").of("general_accuracy", Window(ROWS))

        assert value == pytest.approx(3 / 5)

    def test_accuracy_on_the_prioritized_classes(self) -> None:
        value = self.declared("c1_navigation").of("accuracy_car_truck_bus", Window(ROWS))

        assert value == pytest.approx(2 / 3)

    def test_accuracy_per_class(self) -> None:
        value = self.declared("c3_surveillance").of("per_class_accuracy", Window(ROWS))

        assert value == {"car": pytest.approx(0.5), "cyclist": pytest.approx(0.5), "bus": 1.0}

    def test_a_per_class_cdm_means_the_same_thing_for_every_consumer(self) -> None:
        for name in ("accuracy_car", "accuracy_cyclist", "accuracy_bus"):
            assert self.declared("c1_navigation").of(name, Window(ROWS)) == self.declared(
                "c2_safety"
            ).of(name, Window(ROWS)), name
        assert self.declared("c1_navigation").of("accuracy_car", Window(ROWS)) == pytest.approx(0.5)
        assert self.declared("c1_navigation").of("accuracy_bus", Window(ROWS)) == 1.0

    def test_the_cyclist_read_as_motorcycle_miss_rate(self) -> None:
        value = self.declared("c2_safety").of("miss_rate_cyclist_as_motorcycle", Window(ROWS))

        assert value == pytest.approx(0.5)

    def test_the_response_time_and_its_percentile(self) -> None:
        declared = self.declared("c1_navigation")

        assert declared.of("response_time_ms", Window(ROWS)) == pytest.approx(175.0)
        assert declared.of("response_time_p95_ms", Window(ROWS)) == pytest.approx(197.5)

    def test_the_throughput_is_the_mean_over_the_window(self) -> None:
        value = self.declared("c1_navigation").of("throughput_per_s", Window(ROWS))
        rates = [row["throughput_rps"] for row in ROWS if "throughput_rps" in row]

        assert value == pytest.approx(sum(rates) / len(rates))

    def test_the_serving_cost_per_1k_queries(self) -> None:
        value = self.declared("c1_navigation").of("cost_per_1k_queries", Window(ROWS))

        assert value == pytest.approx(10.5)

    def test_every_declared_cdm_of_every_consumer_answers_a_number(self) -> None:
        for folder in sorted(path.name for path in CONSUMERS.iterdir()):
            declared = self.declared(folder)
            for name in declared.declared:
                value = declared.of(name, Window(EVERY_CLASS))
                if isinstance(value, Mapping):
                    assert value, f"{folder}: {name} answered no group"
                    assert all(isinstance(held, float) for held in value.values())
                    continue
                assert isinstance(value, int | float), f"{folder}: {name} answered {value!r}"

    def test_a_cdm_over_an_empty_window_answers_undefined_rather_than_a_number(self) -> None:
        declared = self.declared("c1_navigation")

        for name in ("accuracy_car", "response_time_ms"):
            assert isinstance(declared.of(name, Window(())), Undefined)


class TestTheWindow:
    def test_it_keeps_only_the_most_recent_rows(self) -> None:
        window = Window(ROWS).last(2)

        assert len(window) == 2
        assert window.rows[-1]["response_time_ms"] == 180.0

    def test_asking_for_more_rows_than_it_holds_keeps_them_all(self) -> None:
        assert len(Window(ROWS).last(99)) == 6

    def test_it_can_keep_only_the_rows_whose_label_arrived(self) -> None:
        assert len(Window(ROWS).labeled("true_label")) == 5

    def test_an_empty_window_knows_it_is_empty(self) -> None:
        assert not Window(())


class TestTheRemainingBranches:
    def test_a_grouped_aggregate_without_a_ratio_gives_one_value_per_group(self) -> None:
        value = evaluate(
            {"function": "average", "attribute": "response_time_ms", "group_by": "true_label"}
        )

        assert value == {
            "bus": pytest.approx(160.0),
            "car": pytest.approx(160.0),
            "cyclist": pytest.approx(195.0),
        }

    def test_a_group_the_rows_cannot_answer_is_left_out(self) -> None:
        rows = (
            {"true_label": "car", "response_time_ms": 150.0},
            {"true_label": "bus", "response_time_ms": None},
        )

        value = evaluate(
            {"function": "average", "attribute": "response_time_ms", "group_by": "true_label"}, rows
        )

        assert value == {"car": pytest.approx(150.0)}

    def test_a_ratio_of_a_grouped_numerator_is_refused(self) -> None:
        with pytest.raises(CdmError, match="grouped numerator"):
            evaluate(
                {
                    "function": "ratio",
                    "inputs": [
                        {
                            "function": "average",
                            "attribute": "response_time_ms",
                            "group_by": "true_label",
                        },
                        {"function": "count"},
                    ],
                }
            )

    def test_a_ratio_by_a_grouped_denominator_is_refused(self) -> None:
        with pytest.raises(CdmError, match="grouped denominator"):
            evaluate(
                {
                    "function": "ratio",
                    "inputs": [
                        {"function": "count"},
                        {
                            "function": "average",
                            "attribute": "response_time_ms",
                            "group_by": "true_label",
                        },
                    ],
                }
            )

    def test_grouping_a_metric_that_is_already_grouped_is_refused(self) -> None:
        declared = CdmSet.model_validate(
            {
                "definitions": {
                    "per_class": {
                        "measured": "per_window",
                        "window": 500,
                        "unit": "milliseconds",
                        "description": "Mean response time per class",
                        "specification": {
                            "function": "average",
                            "attribute": "response_time_ms",
                            "group_by": "true_label",
                        },
                    },
                    "grouped_again": {
                        "measured": "per_window",
                        "window": 500,
                        "unit": "milliseconds",
                        "description": "The same, grouped a second time",
                        "specification": {
                            "function": "reference",
                            "cdm": "per_class",
                            "group_by": "predicted_label",
                        },
                    },
                }
            }
        )

        with pytest.raises(CdmError, match="already grouped"):
            EvaluationEngine(declared).of("grouped_again", Window(ROWS))

    def test_an_undefined_numerator_stays_undefined(self) -> None:
        value = evaluate(
            {
                "function": "ratio",
                "inputs": [
                    {"function": "average", "attribute": "tram_ms"},
                    {"function": "count"},
                ],
            }
        )

        assert isinstance(value, Undefined)

    def test_an_undefined_value_reads_back_as_itself(self) -> None:
        first = Undefined("no row")

        assert repr(first) == "Undefined('no row')"
        assert first == Undefined("no row")
        assert first != Undefined("something else")
        assert len({first, Undefined("no row")}) == 1

    def test_a_window_can_be_indexed_like_the_sequence_it_is(self) -> None:
        window = Window(ROWS)

        assert window[0]["true_label"] == "car"
        assert list(window)[1]["predicted_label"] == "truck"

    def test_a_grouped_reference_gives_one_value_per_group(self) -> None:
        declared = CdmSet.model_validate(
            {
                "definitions": {
                    "mean_time": {
                        "measured": "per_window",
                        "window": 500,
                        "unit": "milliseconds",
                        "description": "Mean response time",
                        "specification": {"function": "average", "attribute": "response_time_ms"},
                    },
                    "mean_time_per_class": {
                        "measured": "per_window",
                        "window": 500,
                        "unit": "milliseconds",
                        "description": "The same, one value per class",
                        "specification": {
                            "function": "reference",
                            "cdm": "mean_time",
                            "group_by": "true_label",
                        },
                    },
                }
            }
        )

        value = EvaluationEngine(declared).of("mean_time_per_class", Window(ROWS))

        assert value == {
            "bus": pytest.approx(160.0),
            "car": pytest.approx(160.0),
            "cyclist": pytest.approx(195.0),
        }

    def test_a_ratio_whose_denominator_cannot_be_answered_is_undefined(self) -> None:
        value = evaluate(
            {
                "function": "ratio",
                "inputs": [
                    {"function": "count"},
                    {"function": "average", "attribute": "tram_ms"},
                ],
            }
        )

        assert isinstance(value, Undefined)


class TestTheWeightsOfThePublishedRuns:
    def rows(self) -> tuple[dict[str, Any], ...]:
        held = []
        for label, right, total in (
            ("car", 2, 3),
            ("truck", 1, 2),
            ("bus", 0, 1),
            ("pedestrian", 2, 2),
            ("cyclist", 1, 1),
            ("motorcycle", 1, 1),
        ):
            for index in range(total):
                held.append(
                    {"true_label": label, "predicted_label": label if index < right else "other"}
                )
        return tuple(held)

    def declared(self, folder: str) -> EvaluationEngine:
        body = yaml.safe_load((CONSUMERS / folder / "cdm.yaml").read_text(encoding="utf-8"))
        return EvaluationEngine(CdmSet.model_validate({"definitions": body["cdms"]}))

    def values(self, folder: str) -> dict[str, object]:
        declared = self.declared(folder)
        return {name: declared.of(name, Window(self.rows())) for name in declared.declared}

    def quality(self, folder: str) -> float:
        contract = load_consumers(Reader(TREE))[folder].contract
        parts = score(contract, self.values(folder), cost_per_1k=0.0)

        assert isinstance(parts, ScoreParts)
        return parts.quality

    def test_the_navigation_consumer_weighs_the_vehicle_classes(self) -> None:
        # car 2 of 3, truck 1 of 2, bus 0 of 1, pedestrian 1, cyclist 1, motorcycle 1
        assert self.quality("c1_navigation") == pytest.approx(
            4 * 2 / 3 + 3 * 1 / 2 + 3 * 0 + 1 * 1 + 1 * 1 + 1 * 1
        )

    def test_the_safety_consumer_weighs_by_the_severity_of_the_class(self) -> None:
        # no cyclist is read as a motorcycle in this window
        assert self.quality("c2_safety") == pytest.approx(
            10 * 1 + 10 * 1 + 10 * 1 + 3 * 2 / 3 + 3 * 1 / 2 + 3 * 0 + 50 * 1.0
        )

    def test_the_two_consumers_rank_the_same_window_differently(self) -> None:
        assert self.quality("c2_safety") > self.quality("c1_navigation")

    def test_the_general_accuracy_pools_over_the_window(self) -> None:
        value = self.declared("c3_surveillance").of("general_accuracy", Window(self.rows()))

        assert value == pytest.approx(7 / 10)


class TestADeclaredWindow:
    def declared(self, window: int) -> EvaluationEngine:
        return EvaluationEngine(
            CdmSet.model_validate(
                {
                    "definitions": {
                        "recent": {
                            "measured": "per_window",
                            "window": window,
                            "unit": "count",
                            "description": "How many of the most recent queries there are",
                            "specification": {"function": "count"},
                        }
                    }
                }
            )
        )

    def test_it_reads_only_the_rows_its_window_declares(self) -> None:
        assert self.declared(2).of("recent", Window(ROWS)) == 2

    def test_a_window_larger_than_the_rows_reads_them_all(self) -> None:
        assert self.declared(500).of("recent", Window(ROWS)) == 6

    def test_two_metrics_of_different_windows_read_different_rows(self) -> None:
        narrow = self.declared(2).of("recent", Window(ROWS))
        wide = self.declared(4).of("recent", Window(ROWS))

        assert (narrow, wide) == (2, 4)

    def test_a_metric_measured_per_request_reads_the_whole_window_it_was_given(self) -> None:
        asked = EvaluationEngine(
            CdmSet.model_validate(
                {
                    "definitions": {
                        "mean_time": {
                            "measured": "per_query",
                            "window": None,
                            "unit": "milliseconds",
                            "description": "Mean response time of what it was given",
                            "specification": {
                                "function": "average",
                                "attribute": "response_time_ms",
                            },
                        }
                    }
                }
            )
        )

        assert asked.of("mean_time", Window(ROWS)) == pytest.approx(175.0)

    def test_the_most_recent_rows_are_the_ones_kept(self) -> None:
        value = self.declared(1).of("recent", Window(ROWS))
        latest = Window(ROWS).last(1).rows[0]

        assert value == 1
        assert latest["response_time_ms"] == 180.0


class TestThePercentileMethod:
    def test_the_hundredth_percentile_is_the_largest(self) -> None:
        assert evaluate(
            {"function": "percentile", "attribute": "response_time_ms", "percentile_value": 100.0}
        ) == pytest.approx(200.0)

    def test_a_single_row_is_its_own_percentile(self) -> None:
        assert evaluate(
            {"function": "percentile", "attribute": "response_time_ms", "percentile_value": 95.0},
            ({"response_time_ms": 42.0},),
        ) == pytest.approx(42.0)

    def test_it_interpolates_between_the_two_nearest(self) -> None:
        rows = tuple({"response_time_ms": float(value)} for value in range(1, 101))

        assert evaluate(
            {"function": "percentile", "attribute": "response_time_ms", "percentile_value": 95.0},
            rows,
        ) == pytest.approx(95.05)


class TestGroupsFiltersAndRefusedValues:
    def test_a_grouped_count_leaves_out_a_row_whose_group_has_not_arrived(self) -> None:
        rows = ({"true_label": "car"}, {"true_label": None})

        value = evaluate({"function": "count", "group_by": "true_label"}, rows)

        assert value == {"car": 1}

    def test_the_groups_come_back_in_one_order_so_a_run_repeats(self) -> None:
        rows = tuple(
            {"true_label": label, "response_time_ms": 10.0}
            for label in ("truck", "car", "bus", "cyclist")
        )

        value = evaluate(
            {"function": "average", "attribute": "response_time_ms", "group_by": "true_label"}, rows
        )

        assert isinstance(value, Mapping)
        assert list(value) == sorted(value)
        assert list(value) == ["bus", "car", "cyclist", "truck"]

    def test_a_filter_can_ask_for_the_rows_whose_field_has_not_arrived(self) -> None:
        value = evaluate(
            {"function": "count", "scopes": [{"attribute": "true_label", "observed": False}]}
        )

        assert value == 1

    @pytest.mark.parametrize("held", ["fast", True, False])
    def test_a_record_that_is_not_a_number_is_refused_whatever_it_is(self, held: object) -> None:
        with pytest.raises(CdmError, match="response_time_ms"):
            evaluate(
                {"function": "average", "attribute": "response_time_ms"},
                ({"response_time_ms": held},),
            )

    def test_a_percentile_reaching_the_evaluator_with_no_value_is_refused(self) -> None:
        asked = MetricSpecification.model_validate(
            {"function": "percentile", "attribute": "response_time_ms", "percentile_value": 95.0}
        )
        without = asked.model_construct(**{**asked.__dict__, "percentile_value": None})

        with pytest.raises(CdmError, match="no percentile_value"):
            aggregate(without, Window(ROWS))


class TestTheWindowIsItsOwn:
    def test_a_window_does_not_share_the_list_it_was_built_from(self) -> None:
        held = [{"true_label": "car"}]
        window = Window(held)

        held.append({"true_label": "truck"})

        assert len(window) == 1


class TestTheReductionsMeanWhatTheyName:
    def declared(self, folder: str) -> EvaluationEngine:
        body = yaml.safe_load((CONSUMERS / folder / "cdm.yaml").read_text(encoding="utf-8"))
        return EvaluationEngine(CdmSet.model_validate({"definitions": body["cdms"]}))

    def mixed(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "true_label": "car",
                "predicted_label": "car",
                "response_time_ms": 400.0 if index < 100 else 80.0,
                "cost_per_1k": 4.0 if index < 100 else 20.0,
                "throughput_rps": 5.0 if index < 100 else 40.0,
            }
            for index in range(500)
        )

    def test_the_throughput_blends_the_window_like_the_cost_and_the_response_time(self) -> None:
        declared = self.declared("c1_navigation")
        rows = Window(self.mixed())

        assert declared.of("throughput_per_s", rows) == pytest.approx((100 * 5 + 400 * 40) / 500)
        assert declared.of("cost_per_1k_queries", rows) == pytest.approx((100 * 4 + 400 * 20) / 500)
        assert declared.of("response_time_ms", rows) == pytest.approx((100 * 400 + 400 * 80) / 500)

    def test_one_bad_query_does_not_latch_the_throughput_for_the_whole_window(self) -> None:
        rows = Window(
            tuple(
                {"true_label": "car", "predicted_label": "car", "throughput_rps": rate}
                for rate in (0.5, *([40.0] * 499))
            )
        )

        assert self.declared("c1_navigation").of("throughput_per_s", rows) == pytest.approx(
            (0.5 + 499 * 40.0) / 500
        )

    def test_a_group_label_that_is_not_a_class_name_is_refused(self) -> None:
        rows = Window(
            (
                {"true_label": 1, "predicted_label": 1},
                {"true_label": "car", "predicted_label": "car"},
            )
        )

        with pytest.raises(CdmError, match="true_label"):
            self.declared("c1_navigation").of("per_class_accuracy", rows)
