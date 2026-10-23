from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

from eemls.domain.contract import ConsumerContract, Objective
from eemls.evaluation_engine.aggregation_functions import Undefined
from eemls.evaluation_engine.contract_scoring import (
    NotScored,
    ScoreError,
    ScoreParts,
    map_to_range,
    score,
)


def spec(**changes: Any) -> dict[str, Any]:
    return {
        "min_value": 0.0,
        "max_value": 1.0,
        "weight": 1.0,
        "direction": "maximize",
        "mapping": "linear",
        "mapping_base": None,
    } | changes


def objective(**changes: Any) -> Objective:
    return Objective.model_validate(spec(**changes))


def contract(**changes: Any) -> ConsumerContract:
    fields: dict[str, Any] = {
        "consumer_id": "c1_navigation",
        "description": "Counts and classifies vehicles",
        "prioritized_classes": ("car", "truck", "bus"),
        "request_rate_per_s": 20.0,
        "objectives": {"accuracy_car": objective()},
        "cost_weight": 0.15,
        "cost_range_max_per_1k": 30.0,
        "feedback": {"fraction": 0.5, "delay_intervals": 15},
    }
    return ConsumerContract.model_validate(fields | changes)


def scored(asked: ConsumerContract, values: dict[str, Any], cost_per_1k: float) -> ScoreParts:
    result = score(asked, values, cost_per_1k=cost_per_1k)
    assert isinstance(result, ScoreParts)
    return result


class TestTransformingOneValue:
    def test_a_value_at_the_top_of_its_range_is_one(self) -> None:
        assert map_to_range(1.0, objective()) == pytest.approx(1.0)

    def test_a_value_at_the_bottom_of_its_range_is_zero(self) -> None:
        assert map_to_range(0.0, objective()) == pytest.approx(0.0)

    def test_a_value_inside_a_narrow_range_uses_that_range(self) -> None:
        narrow = objective(min_value=0.6, max_value=0.9)

        assert map_to_range(0.75, narrow) == pytest.approx(0.5)

    @pytest.mark.parametrize(("value", "expected"), [(-1.0, 0.0), (2.0, 1.0)])
    def test_a_value_outside_its_range_is_clamped(self, value: float, expected: float) -> None:
        assert map_to_range(value, objective()) == pytest.approx(expected)

    def test_a_lower_is_better_metric_is_flipped(self) -> None:
        miss_rate = objective(min_value=0.0, max_value=0.2, direction="minimize")

        assert map_to_range(0.0, miss_rate) == pytest.approx(1.0)
        assert map_to_range(0.2, miss_rate) == pytest.approx(0.0)
        assert map_to_range(0.05, miss_rate) == pytest.approx(0.75)

    def test_a_log_scale_rewards_the_first_improvements_most(self) -> None:
        logged = objective(mapping="log", mapping_base=10.0)

        at_half = map_to_range(0.5, logged)

        assert at_half > 0.5
        assert at_half == pytest.approx(math.log(1.0 + 9.0 * 0.5, 10.0))

    def test_a_log_scale_keeps_the_ends_of_the_range(self) -> None:
        logged = objective(mapping="log", mapping_base=10.0)

        assert map_to_range(0.0, logged) == pytest.approx(0.0)
        assert map_to_range(1.0, logged) == pytest.approx(1.0)

    def test_a_log_scale_can_also_be_flipped(self) -> None:
        logged = objective(mapping="log", mapping_base=10.0, direction="minimize")

        assert map_to_range(0.0, logged) == pytest.approx(1.0)
        assert map_to_range(1.0, logged) == pytest.approx(0.0)

    def test_an_unknown_scale_name_is_refused_where_the_objective_is_built(self) -> None:
        with pytest.raises(ValidationError, match="sigmoid"):
            objective(mapping="sigmoid")

    def test_a_range_of_no_width_is_refused_where_the_objective_is_built(self) -> None:
        with pytest.raises(ValidationError, match="not below max_value"):
            objective(min_value=0.5, max_value=0.5)

    def test_a_log_base_at_or_below_one_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="not above one"):
            objective(mapping="log", mapping_base=1.0)

    def test_a_log_objective_reaching_the_scorer_with_no_base_is_refused(self) -> None:
        built = objective(mapping="log", mapping_base=10.0)
        without = built.model_construct(**{**built.__dict__, "mapping_base": None})

        with pytest.raises(ScoreError, match="no base"):
            map_to_range(0.5, without)


class TestScoringACandidate:
    def test_the_score_is_the_weighted_quality_minus_the_weighted_cost(self) -> None:
        result = scored(contract(), {"accuracy_car": 0.86}, 10.5)

        assert result.quality == pytest.approx(0.86)
        assert result.cost == pytest.approx(0.005 * 10.5)
        assert result.total == pytest.approx(0.86 - 0.0525)

    def test_each_objective_contributes_by_its_weight(self) -> None:
        two = contract(
            objectives={
                "accuracy_car": spec(weight=1.0),
                "miss_rate_cyclist_as_motorcycle": spec(
                    max_value=0.2, weight=2.0, direction="minimize"
                ),
            }
        )

        result = scored(
            two,
            {"accuracy_car": 0.8, "miss_rate_cyclist_as_motorcycle": 0.05},
            0.0,
        )

        assert result.by_objective["accuracy_car"] == pytest.approx(0.8)
        assert result.by_objective["miss_rate_cyclist_as_motorcycle"] == pytest.approx(1.5)
        assert result.total == pytest.approx(2.3)

    def test_a_weight_over_a_range_sets_how_much_a_metric_moves_the_score(self) -> None:
        two = contract(
            objectives={
                "accuracy_car": spec(weight=1.0),
                "miss_rate_cyclist_as_motorcycle": spec(
                    max_value=0.2, weight=2.0, direction="minimize"
                ),
            }
        )
        values = {"accuracy_car": 0.8, "miss_rate_cyclist_as_motorcycle": 0.05}

        base = scored(two, values, 0.0).total
        better_accuracy = scored(two, values | {"accuracy_car": 0.81}, 0.0).total
        worse_miss_rate = scored(two, values | {"miss_rate_cyclist_as_motorcycle": 0.06}, 0.0).total

        assert better_accuracy - base == pytest.approx(0.01)
        assert base - worse_miss_rate == pytest.approx(0.1)
        assert (base - worse_miss_rate) / (better_accuracy - base) == pytest.approx(10.0)

    def test_a_value_the_rows_could_not_answer_makes_the_candidate_unscorable(self) -> None:
        result = score(
            contract(),
            {"accuracy_car": Undefined("no labeled row")},
            cost_per_1k=10.5,
        )

        assert isinstance(result, NotScored)
        assert "accuracy_car" in result.missing

    def test_a_missing_value_makes_the_candidate_unscorable_and_names_it(self) -> None:
        result = score(contract(), {}, cost_per_1k=10.5)

        assert isinstance(result, NotScored)
        assert result.missing == ("accuracy_car",)

    def test_a_value_for_a_cdm_the_contract_does_not_score_is_ignored(self) -> None:
        result = scored(contract(), {"accuracy_car": 0.86, "response_time_ms": 174.0}, 0.0)

        assert result.total == pytest.approx(0.86)

    def test_a_grouped_value_cannot_be_scored_and_says_so(self) -> None:
        with pytest.raises(ScoreError, match="per class"):
            score(contract(), {"accuracy_car": {"car": 0.9}}, cost_per_1k=0.0)

    def test_a_negative_cost_is_refused(self) -> None:
        with pytest.raises(ScoreError, match="cost"):
            score(contract(), {"accuracy_car": 0.86}, cost_per_1k=-1.0)

    def test_a_candidate_that_costs_nothing_scores_its_quality(self) -> None:
        result = scored(contract(), {"accuracy_car": 0.9}, 0.0)

        assert result.total == pytest.approx(0.9)


class TestTheCostTermAtItsBoundary:
    def test_a_cost_at_the_normalizer_costs_its_whole_weight(self) -> None:
        result = scored(contract(), {"accuracy_car": 1.0}, 30.0)

        assert result.cost == pytest.approx(0.15)

    def test_a_cost_above_the_normalizer_costs_no_more(self) -> None:
        at_the_norm = scored(contract(), {"accuracy_car": 1.0}, 30.0)
        far_above = scored(contract(), {"accuracy_car": 1.0}, 300.0)

        assert far_above.cost == pytest.approx(at_the_norm.cost)
        assert far_above.total == pytest.approx(0.85)

    def test_a_cost_just_below_the_normalizer_is_still_linear(self) -> None:
        result = scored(contract(), {"accuracy_car": 1.0}, 15.0)

        assert result.cost == pytest.approx(0.075)

    @pytest.mark.parametrize("held", [True, False])
    def test_a_boolean_is_not_a_cdm_value(self, held: bool) -> None:
        result = score(contract(), {"accuracy_car": held}, cost_per_1k=0.0)

        assert isinstance(result, NotScored)
        assert result.missing == ("accuracy_car",)


CLASSES = ("pedestrian", "cyclist", "motorcycle", "car", "truck", "bus")


def per_class_contract(weights: dict[str, float]) -> ConsumerContract:
    return ConsumerContract.model_validate(
        {
            "consumer_id": "c2_safety",
            "description": "Weighs each class on its own",
            "prioritized_classes": ("pedestrian", "cyclist", "motorcycle"),
            "request_rate_per_s": 20.0,
            "objectives": {
                f"accuracy_{name}": {
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "weight": weight,
                    "direction": "maximize",
                    "mapping": "linear",
                    "mapping_base": None,
                }
                for name, weight in weights.items()
            },
            "cost_weight": 0.15,
            "cost_range_max_per_1k": 30.0,
            "feedback": {"fraction": 0.5, "delay_intervals": 15},
        }
    )


class TestClassWeightsLiveInTheContract:
    def test_the_score_is_the_weighted_sum_over_the_classes_the_window_answered(self) -> None:
        asked = per_class_contract({"pedestrian": 10.0, "car": 3.0})
        parts = score(asked, {"accuracy_pedestrian": 0.9, "accuracy_car": 0.5}, cost_per_1k=0.0)

        assert isinstance(parts, ScoreParts)
        assert parts.quality == pytest.approx(10.0 * 0.9 + 3.0 * 0.5)

    def test_a_class_the_window_never_saw_drops_out_with_its_weight(self) -> None:
        asked = per_class_contract({"pedestrian": 10.0, "car": 3.0})
        parts = score(
            asked,
            {"accuracy_pedestrian": 0.9, "accuracy_car": Undefined("no car in the window")},
            cost_per_1k=0.0,
        )

        assert isinstance(parts, ScoreParts)
        assert parts.quality == pytest.approx(10.0 * 0.9)

    def test_the_cost_is_subtracted_from_the_weighted_sum(self) -> None:
        asked = per_class_contract({"pedestrian": 10.0, "car": 3.0})
        parts = score(asked, {"accuracy_pedestrian": 1.0, "accuracy_car": 1.0}, cost_per_1k=30.0)

        assert isinstance(parts, ScoreParts)
        assert parts.quality == pytest.approx(13.0)
        assert parts.cost == pytest.approx(0.15)
        assert parts.total == pytest.approx(12.85)

    def test_a_window_that_answered_no_class_at_all_is_not_scored(self) -> None:
        asked = per_class_contract({"pedestrian": 10.0, "car": 3.0})
        parts = score(
            asked,
            {
                "accuracy_pedestrian": Undefined("no pedestrian"),
                "accuracy_car": Undefined("no car"),
            },
            cost_per_1k=0.0,
        )

        assert isinstance(parts, NotScored)
        assert set(parts.missing) == {"accuracy_pedestrian", "accuracy_car"}
