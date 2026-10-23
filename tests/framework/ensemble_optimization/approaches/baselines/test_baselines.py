from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest

from eemls.domain.records import InferenceServiceProfile
from eemls.domain.types import Row, RuntimeContext
from eemls.ensemble_optimization.approach import ConsumerSpecification, Observation
from eemls.ensemble_optimization.approaches.baselines.axiomvision import AxiomVision
from eemls.ensemble_optimization.approaches.baselines.cocktail import Cocktail
from eemls.ensemble_optimization.approaches.baselines.frugalml import FrugalMl
from eemls.ensemble_optimization.approaches.baselines.scale_reduction import ScaleReduction
from eemls.ensemble_optimization.approaches.baselines.sneakpeek import SneakPeek
from eemls.ensemble_optimization.testing.samples import (
    POOL,
    answered,
    observed,
    profiles,
    replace_id,
)
from eemls.evaluation_engine.window import Window
from eemls.testing.records import context, profile


def voted(
    votes: Mapping[str, tuple[str, ...]],
    *,
    held: dict[str, InferenceServiceProfile] | None = None,
) -> Observation:
    queries = max(len(labels) for labels in votes.values())
    members = tuple(
        {
            "query_id": f"q-{index}",
            "service_name": name,
            "predicted_label": label,
            "confidence": 0.8,
            "true_label": "car",
            "total_ms": 20.0,
        }
        for name, labels in votes.items()
        for index, label in enumerate(labels)
    )
    base = observed(serving=tuple(votes), right=queries, held=held)
    return replace(base, member_window=Window(members))


class TestCocktailAgainstAnAccuracyTarget:
    def built(self) -> Cocktail:
        return Cocktail(
            max_ensemble_size=3,
            evaluation_window=500,
            accuracy_target=0.85,
            accuracy_target_range=0.05,
        )

    def test_a_missed_target_grows_the_ensemble(self) -> None:
        held = self.built().decide(observed(serving=("infer-a",), right=1, wrong=1))

        assert held.action.kind == "add"

    def test_the_service_it_adds_is_the_most_accurate_one_outside(self) -> None:
        held = self.built().decide(
            observed(
                serving=("infer-a",),
                right=1,
                wrong=1,
                held=profiles(**{"infer-c": {"general_accuracy": 0.95}}),
            )
        )

        assert held.action.member_in == "infer-c"

    def test_an_accuracy_above_the_target_and_its_range_shrinks_the_ensemble(self) -> None:
        held = self.built().decide(observed(serving=POOL, right=10, wrong=0))

        assert held.action.kind == "remove"

    def test_the_member_it_removes_is_the_least_accurate_one(self) -> None:
        held = self.built().decide(
            observed(
                serving=POOL,
                right=10,
                held=profiles(**{"infer-b": {"general_accuracy": 0.4}}),
            )
        )

        assert held.action.member_out == "infer-b"

    def test_an_accuracy_inside_the_target_range_changes_nothing(self) -> None:
        held = self.built().decide(observed(serving=("infer-a", "infer-b"), right=9, wrong=1))

        assert held.action.kind == "keep"

    def test_an_accuracy_just_past_the_range_does_shrink_it(self) -> None:
        held = self.built().decide(observed(serving=POOL, right=19, wrong=1))

        assert held.action.kind == "remove"

    def test_a_full_ensemble_that_misses_the_target_cannot_grow(self) -> None:
        held = self.built().decide(observed(serving=POOL, right=1, wrong=1))

        assert held.action.kind == "keep"

    def test_the_last_member_is_never_removed(self) -> None:
        held = self.built().decide(observed(serving=("infer-a",), right=10))

        assert held.action.kind == "keep"

    def test_a_window_with_no_label_leaves_the_ensemble_alone(self) -> None:
        held = self.built().decide(observed(serving=("infer-a",), right=0, wrong=0))

        assert held.action.kind == "keep"

    def test_two_members_that_agree_are_never_pruned(self) -> None:
        held = self.built().decide(observed(serving=("infer-a", "infer-b"), right=10))

        assert held.action.kind == "keep"

    def test_members_that_mostly_split_their_votes_are_not_pruned(self) -> None:
        held = self.built().decide(
            voted(
                {
                    "infer-a": ("car",) * 10,
                    "infer-b": ("car",) * 10,
                    "infer-c": ("bus",) * 6 + ("car",) * 4,
                }
            )
        )

        assert held.action.kind == "keep"

    def test_members_that_mostly_agree_are_pruned_to_a_bare_majority(self) -> None:
        held = self.built().decide(
            voted(
                {
                    "infer-a": ("car",) * 10,
                    "infer-b": ("car",) * 10,
                    "infer-c": ("bus",) * 4 + ("car",) * 6,
                }
            )
        )

        assert held.action.kind == "remove"
        assert held.action.member_out == "infer-c"

    def test_the_member_it_removes_is_the_least_accurate_on_the_recent_labels(self) -> None:
        held = self.built().decide(
            voted(
                {
                    "infer-a": ("car",) * 10,
                    "infer-b": ("bus",) + ("car",) * 9,
                    "infer-c": ("car",) * 10,
                },
                held=profiles(**{"infer-a": {"general_accuracy": 0.4}}),
            )
        )

        assert held.action.member_out == "infer-b"

    def test_a_just_added_member_without_labels_is_judged_by_its_profile(self) -> None:
        labeled_answers = tuple(
            {
                "query_id": f"q-{index}",
                "service_name": name,
                "predicted_label": "bus" if name == "infer-b" and index == 0 else "car",
                "confidence": 0.8,
                "true_label": "car",
                "total_ms": 20.0,
            }
            for name in ("infer-a", "infer-b")
            for index in range(10)
        )
        unlabeled_answers = tuple(
            {
                "query_id": f"q-new-{index}",
                "service_name": name,
                "predicted_label": "car",
                "confidence": 0.8,
                "true_label": None,
                "total_ms": 20.0,
            }
            for name in POOL
            for index in range(11)
        )
        base = observed(
            serving=POOL, right=10, held=profiles(**{"infer-c": {"general_accuracy": 0.95}})
        )

        unlabeled = tuple(answered(f"q-new-{index}", "car", None) for index in range(11))

        held = self.built().decide(
            replace(
                base,
                window=Window((*base.window, *unlabeled)),
                member_window=Window(labeled_answers + unlabeled_answers),
            )
        )

        assert held.action.kind == "remove"
        assert held.action.member_out == "infer-b"

    def test_the_cdm_values_of_the_consumer_do_not_change_its_choice(self) -> None:
        seen = observed(serving=("infer-a",), right=1, wrong=1)

        held = self.built().decide(replace(seen, cdm_values={"general_accuracy": 1.0}))

        assert held.action == self.built().decide(seen).action
        assert held.action.kind == "add"


class TestFrugalMlUnderABudget:
    def built(self, budget_per_1k: float = 20.0, failing_intervals: int = 2) -> FrugalMl:
        return FrugalMl(
            max_ensemble_size=3,
            evaluation_window=500,
            budget_per_1k=budget_per_1k,
            quality_level=0.8,
            failing_intervals=failing_intervals,
        )

    def priced(self) -> dict[str, InferenceServiceProfile]:
        return profiles(
            **{
                "infer-a": {"general_accuracy": 0.6, "cost_per_1k": 1.0},
                "infer-b": {"general_accuracy": 0.85, "cost_per_1k": 4.0},
                "infer-c": {"general_accuracy": 0.95, "cost_per_1k": 15.0},
            }
        )

    def test_one_failing_interval_below_the_threshold_changes_nothing(self) -> None:
        held = self.built().decide(observed(serving=("infer-a",), right=1, wrong=1))

        assert held.action.kind == "keep"

    def test_enough_failing_intervals_add_a_service(self) -> None:
        held = self.built()

        held.decide(observed(serving=("infer-a",), right=1, wrong=1, held=self.priced()))
        decided = held.decide(observed(serving=("infer-a",), right=1, wrong=1, held=self.priced()))

        assert decided.action.kind == "add"

    def test_the_service_it_adds_is_the_most_accurate_one_within_the_budget(self) -> None:
        decided = self.built(failing_intervals=1).decide(
            observed(serving=("infer-a",), right=1, wrong=1, held=self.priced())
        )

        assert decided.action.member_in == "infer-c"

    def test_it_will_not_add_a_service_that_takes_the_ensemble_past_the_budget(self) -> None:
        decided = self.built(budget_per_1k=10.0, failing_intervals=1).decide(
            observed(serving=("infer-a",), right=1, wrong=1, held=self.priced())
        )

        assert decided.action.member_in == "infer-b"

    def test_a_full_ensemble_swaps_its_least_accurate_member_for_a_more_accurate_one(
        self,
    ) -> None:
        pool = self.priced() | {
            "infer-d": profile(
                service_name="infer-d", instance_id="i-d", general_accuracy=0.9, cost_per_1k=2.0
            )
        }

        decided = FrugalMl(
            max_ensemble_size=3,
            evaluation_window=500,
            budget_per_1k=25.0,
            quality_level=0.8,
            failing_intervals=1,
        ).decide(observed(serving=POOL, right=1, wrong=1, held=pool))

        assert decided.action.kind == "replace"
        assert decided.action.member_out == "infer-a"
        assert decided.action.member_in == "infer-d"

    def test_nothing_within_the_budget_keeps_the_ensemble(self) -> None:
        decided = self.built(budget_per_1k=2.0, failing_intervals=1).decide(
            observed(serving=("infer-a",), right=1, wrong=1, held=self.priced())
        )

        assert decided.action.kind == "keep"

    def test_an_ensemble_over_the_budget_switches_to_the_cheapest_service(self) -> None:
        decided = self.built(budget_per_1k=10.0).decide(
            observed(serving=("infer-b", "infer-c"), right=10, held=self.priced())
        )

        assert decided.action.kind == "select"
        assert decided.action.members == ("infer-a",)

    def test_an_ensemble_at_the_quality_level_is_never_reduced(self) -> None:
        held = self.built(failing_intervals=1)

        decided = [
            held.decide(observed(serving=("infer-a", "infer-c"), right=10, held=self.priced()))
            for _ in range(3)
        ]

        assert {one.action.kind for one in decided} == {"keep"}

    def test_the_count_starts_again_after_the_ensemble_changes(self) -> None:
        held = self.built()

        held.decide(observed(serving=("infer-a",), right=1, wrong=1, held=self.priced()))
        held.decide(observed(serving=("infer-a",), right=1, wrong=1, held=self.priced()))
        grown = observed(
            serving=("infer-a", "infer-c"),
            given=tuple(
                {**answered(f"q-{index}", "bus", "car"), "deployment_id": "d-2"}
                for index in range(4)
            ),
            held=self.priced(),
        )
        after = held.decide(replace(grown, deployment=replace_id(grown.deployment, "d-2")))

        assert after.action.kind == "keep"

    def test_labels_of_an_earlier_ensemble_do_not_count(self) -> None:
        earlier = tuple(
            {**answered(f"q-{index}", "bus", "car"), "deployment_id": "d-0"} for index in range(4)
        )

        decided = self.built(failing_intervals=1).decide(
            observed(serving=("infer-a",), given=earlier, held=self.priced())
        )

        assert decided.action.kind == "keep"

    def test_a_good_interval_starts_the_count_again(self) -> None:
        held = self.built()

        held.decide(observed(serving=("infer-a",), right=1, wrong=1, held=self.priced()))
        held.decide(observed(serving=("infer-a",), right=10, held=self.priced()))
        decided = held.decide(observed(serving=("infer-a",), right=1, wrong=1, held=self.priced()))

        assert decided.action.kind == "keep"

    def test_a_window_with_no_label_keeps_the_count(self) -> None:
        held = self.built()

        held.decide(observed(serving=("infer-a",), right=1, wrong=1, held=self.priced()))
        held.decide(observed(serving=("infer-a",), right=0, wrong=0, held=self.priced()))
        decided = held.decide(observed(serving=("infer-a",), right=1, wrong=1, held=self.priced()))

        assert decided.action.kind == "add"

    def test_two_consumers_taking_turns_each_escalate_after_their_own_count(self) -> None:
        held = self.built()
        failing = observed(serving=("infer-a",), right=1, wrong=1, held=self.priced())
        first, second = of_consumer(failing, "c1", "d-1"), of_consumer(failing, "c2", "d-2")

        kinds = [held.decide(one).action.kind for one in (first, second, first, second)]

        assert kinds == ["keep", "keep", "add", "add"]

    def test_a_window_with_no_label_is_not_counted(self) -> None:
        held = self.built()

        held.decide(observed(serving=("infer-a",), right=0, wrong=0, held=self.priced()))
        decided = held.decide(observed(serving=("infer-a",), right=1, wrong=1, held=self.priced()))

        assert decided.action.kind == "keep"


class TestSneakPeekUnderTheObservedClassDistribution:
    def built(self) -> SneakPeek:
        return SneakPeek(classes=CLASSES, max_ensemble_size=2, evaluation_window=500)

    def by_class(self, **held: Mapping[str, float]) -> dict[str, InferenceServiceProfile]:
        return profiles(
            **{
                name: {"accuracy_by_class": mine, "confidence_by_class": mine}
                for name, mine in held.items()
            }
        )

    def test_the_service_that_is_best_on_the_observed_classes_serves(self) -> None:
        held = self.built().decide(
            observed(
                serving=("infer-a",),
                right=4,
                held=self.by_class(
                    **{
                        "infer-a": {"car": 0.5, "bus": 0.9},
                        "infer-b": {"car": 0.95, "bus": 0.1},
                        "infer-c": {"car": 0.2, "bus": 0.2},
                    }
                ),
            )
        )

        assert held.action.kind == "select"
        assert held.action.members[0] == "infer-b"

    def test_a_service_strong_on_a_class_the_window_never_shows_is_not_chosen(self) -> None:
        held = self.built().decide(
            observed(
                serving=("infer-a",),
                right=4,
                held=self.by_class(
                    **{
                        "infer-a": {"car": 0.6, "bus": 0.1},
                        "infer-b": {"car": 0.7, "bus": 0.1},
                        "infer-c": {"car": 0.1, "bus": 0.99},
                    }
                ),
            )
        )

        assert "infer-c" not in held.action.members

    def test_it_takes_as_many_as_the_size_limit_allows(self) -> None:
        held = self.built().decide(observed(serving=("infer-a",), right=4))

        assert len(held.action.members) == 2

    def test_a_window_showing_no_class_keeps_the_ensemble(self) -> None:
        held = self.built().decide(observed(serving=("infer-a",), right=0, wrong=0))

        assert held.action.kind == "keep"

    def test_the_class_distribution_comes_from_the_true_labels(self) -> None:
        held = self.built().decide(
            observed(
                serving=("infer-a",),
                given=tuple(answered(f"q-{index}", "bus", "car") for index in range(4)),
                held=self.by_class(
                    **{
                        "infer-a": {"car": 0.9, "bus": 0.1},
                        "infer-b": {"car": 0.8, "bus": 0.1},
                        "infer-c": {"car": 0.1, "bus": 0.99},
                    }
                ),
            )
        )

        assert held.action.members == ("infer-a", "infer-b")

    def test_queries_without_a_true_label_show_no_class(self) -> None:
        held = self.built().decide(
            observed(
                serving=("infer-a",),
                given=tuple(answered(f"q-{index}", "car", None) for index in range(4)),
            )
        )

        assert held.action.kind == "keep"

    def test_the_cost_of_a_service_is_subtracted_from_its_accuracy(self) -> None:
        held = SneakPeek(classes=CLASSES, max_ensemble_size=1, evaluation_window=500).decide(
            observed(
                serving=("infer-a",),
                right=4,
                cost_weight=0.5,
                held=profiles(
                    **{
                        "infer-a": {"accuracy_by_class": {"car": 0.80}, "cost_per_1k": 30.0},
                        "infer-b": {"accuracy_by_class": {"car": 0.75}, "cost_per_1k": 3.0},
                        "infer-c": {"accuracy_by_class": {"car": 0.10}, "cost_per_1k": 0.0},
                    }
                ),
            )
        )

        assert held.action.members == ("infer-b",)

    def test_the_cost_counts_as_much_against_accuracy_as_the_contract_says(self) -> None:
        held = SneakPeek(classes=CLASSES, max_ensemble_size=1, evaluation_window=500).decide(
            observed(
                serving=("infer-a",),
                right=4,
                cost_weight=0.01,
                held=profiles(
                    **{
                        "infer-a": {"accuracy_by_class": {"car": 0.80}, "cost_per_1k": 30.0},
                        "infer-b": {"accuracy_by_class": {"car": 0.75}, "cost_per_1k": 3.0},
                        "infer-c": {"accuracy_by_class": {"car": 0.10}, "cost_per_1k": 0.0},
                    }
                ),
            )
        )

        assert held.action.members == ("infer-a",)

    def test_every_candidate_it_weighed_is_counted(self) -> None:
        held = self.built().decide(observed(serving=("infer-a",), right=4))

        assert held.candidates == len(POOL)


def labeled(
    serving: tuple[str, ...],
    answers: Mapping[str, tuple[bool, ...]],
    *,
    period: str = "car",
    at: RuntimeContext | None = None,
    in_ensemble: bool = True,
    true_label: str | None = "car",
    held: dict[str, InferenceServiceProfile] | None = None,
) -> Observation:
    queries = max(len(held) for held in answers.values())
    rows = tuple(
        {**answered(f"q-{index}", "car", true_label), "period_name": period}
        for index in range(queries)
    )
    members = tuple(
        {
            "query_id": f"q-{index}",
            "service_name": name,
            "in_ensemble": in_ensemble,
            "predicted_label": "car" if right else "bus",
            "confidence": 0.8,
            "true_label": true_label,
            "total_ms": 20.0,
        }
        for name, held in answers.items()
        for index, right in enumerate(held)
    )
    base = observed(serving=serving, at=at, held=UNPROFILED if held is None else held)
    return replace(base, window=Window(rows), member_window=Window(members))


CLASSES = ("car", "bus", "truck")


def value(held: AxiomVision, at: RuntimeContext, service_name: str) -> float:
    return held._ranked_value(at.period_name, service_name)


UNPROFILED = profiles(**{name: {"general_accuracy": 0.0} for name in POOL})


def of_consumer(
    seen: Observation, consumer_id: str, deployment_id: str | None = None
) -> Observation:
    def moved(row: Row) -> Row:
        changed = {**row, "query_id": f"{consumer_id}-{row['query_id']}"}
        if deployment_id is not None and "deployment_id" in row:
            changed["deployment_id"] = deployment_id
        return changed

    consumer = ConsumerSpecification(
        contract=seen.consumer.contract.model_copy(update={"consumer_id": consumer_id}),
        cdms=seen.consumer.cdms,
    )
    deployment = (
        seen.deployment if deployment_id is None else replace_id(seen.deployment, deployment_id)
    )
    return replace(
        seen,
        consumer=consumer,
        deployment=deployment,
        window=Window(tuple(moved(row) for row in seen.window)),
        member_window=Window(tuple(moved(row) for row in seen.member_window)),
    )


class TestAxiomVisionPerPeriodOfTheDay:
    car = context(period_name="car", period_index=1)
    bus = context(period_name="bus", period_index=2)

    def built(self, *, exploration_alpha: float = 0.0, max_ensemble_size: int = 3) -> AxiomVision:
        return AxiomVision(
            max_ensemble_size=max_ensemble_size,
            evaluation_window=500,
            exploration_alpha=exploration_alpha,
            prior_zeta=1.0,
            accuracy_target=0.85,
        )

    def test_each_service_learns_from_its_own_predictions(self) -> None:
        held = self.built()

        held.decide(
            labeled(
                ("infer-a", "infer-b"),
                {"infer-a": (True,) * 4, "infer-b": (False,) * 4},
                at=self.car,
            )
        )

        assert value(held, self.car, "infer-a") == pytest.approx(0.8)
        assert value(held, self.car, "infer-b") == pytest.approx(0.0)

    def test_a_labeled_prediction_is_counted_once_across_decisions(self) -> None:
        held = self.built()
        seen = labeled(("infer-a",), {"infer-a": (True, True, True, False)}, at=self.car)

        held.decide(seen)
        held.decide(seen)

        assert value(held, self.car, "infer-a") == pytest.approx(3 / 5)

    def test_two_consumers_taking_turns_count_each_labeled_prediction_once(self) -> None:
        held = self.built()
        seen = labeled(("infer-a",), {"infer-a": (True, True, True, False)}, at=self.car)
        first, second = of_consumer(seen, "c1"), of_consumer(seen, "c2")

        for _ in range(3):
            held.decide(first)
            held.decide(second)

        assert value(held, self.car, "infer-a") == pytest.approx(6 / 9)

    def test_an_untried_service_starts_from_its_profile_accuracy(self) -> None:
        held = self.built()
        pool = profiles(**{"infer-c": {"general_accuracy": 0.9}})

        held.decide(labeled(("infer-a",), {"infer-a": (True,) * 4}, at=self.car, held=pool))

        assert value(held, self.car, "infer-c") == pytest.approx(0.9)

    def test_an_untried_service_with_a_high_profile_accuracy_can_be_selected(self) -> None:
        held = self.built(exploration_alpha=0.25)
        pool = profiles(
            **{
                "infer-a": {"general_accuracy": 0.5},
                "infer-b": {"general_accuracy": 0.5},
                "infer-c": {"general_accuracy": 0.95},
            }
        )

        decided = held.decide(
            labeled(("infer-a",), {"infer-a": (True,) * 9}, at=self.car, held=pool)
        )

        assert "infer-c" in decided.action.members

    def test_labels_move_the_estimate_from_the_profile_toward_the_observed_accuracy(
        self,
    ) -> None:
        held = self.built()
        pool = profiles(**{"infer-c": {"general_accuracy": 0.95}})

        held.decide(labeled(("infer-c",), {"infer-c": (False,) * 4}, at=self.car, held=pool))

        assert value(held, self.car, "infer-c") == pytest.approx(0.95 / 5)

    def test_a_label_is_filed_under_the_period_of_its_query(self) -> None:
        held = self.built()

        held.decide(labeled(("infer-a",), {"infer-a": (True,) * 4}, period="car", at=self.bus))

        assert value(held, self.car, "infer-a") == pytest.approx(0.8)
        assert value(held, self.bus, "infer-a") == pytest.approx(0.0)

    def test_a_prediction_without_a_true_label_teaches_it_nothing(self) -> None:
        held = self.built()

        held.decide(labeled(("infer-a",), {"infer-a": (True,) * 4}, true_label=None, at=self.car))

        assert value(held, self.car, "infer-a") == pytest.approx(0.0)

    def test_a_service_it_knows_less_about_carries_a_larger_bonus(self) -> None:
        held = self.built(exploration_alpha=0.25)

        held.decide(labeled(("infer-a",), {"infer-a": (True,) * 4}, at=self.car))

        assert value(held, self.car, "infer-b") == pytest.approx(0.25)
        assert value(held, self.car, "infer-a") == pytest.approx(0.8 + 0.25 * (1 / 5) ** 0.5)

    def test_an_estimate_below_the_target_adds_the_next_ranked_service(self) -> None:
        held = self.built()

        decided = held.decide(
            labeled(
                ("infer-a",),
                {"infer-a": (True,) * 4, "infer-b": (True, True, False, False)},
                at=self.car,
            )
        )

        assert decided.action.kind == "select"
        assert decided.action.members == ("infer-a", "infer-b")

    def test_it_stops_adding_once_its_estimate_reaches_the_target(self) -> None:
        held = self.built()

        decided = held.decide(
            labeled(
                ("infer-a", "infer-b"),
                {"infer-a": (True,) * 4, "infer-b": (True,) * 4},
                at=self.car,
            )
        )

        assert decided.action.members == ("infer-a", "infer-b")

    def test_an_ensemble_whose_top_two_reach_the_target_drops_to_two(self) -> None:
        held = self.built()

        decided = held.decide(
            labeled(
                ("infer-a", "infer-b", "infer-c"),
                {"infer-a": (True,) * 4, "infer-b": (True,) * 4, "infer-c": (False,) * 4},
                at=self.car,
            )
        )

        assert decided.action.kind == "select"
        assert decided.action.members == ("infer-a", "infer-b")

    def test_an_ensemble_below_the_target_grows_in_rank_order(self) -> None:
        held = self.built()
        held.decide(labeled(("infer-a",), {"infer-c": (True, True, False, False)}, at=self.car))

        decided = held.decide(
            labeled(("infer-a",), {"infer-a": (True, False, False, False)}, at=self.car)
        )

        assert decided.action.members == ("infer-c", "infer-a", "infer-b")

    def test_the_selected_ensemble_never_exceeds_its_size_limit(self) -> None:
        held = self.built(max_ensemble_size=2)

        decided = held.decide(
            labeled(("infer-a",), {"infer-a": (True, False, False, False)}, at=self.car)
        )

        assert len(decided.action.members) == 2

    def test_a_full_ensemble_below_the_target_swaps_its_weakest_member(self) -> None:
        held = self.built(max_ensemble_size=2)
        held.decide(labeled(("infer-c",), {"infer-c": (True, True, True, False)}, at=self.car))

        decided = held.decide(
            labeled(
                ("infer-a", "infer-b"),
                {"infer-a": (True, False, False, False), "infer-b": (False,) * 4},
                at=self.car,
            )
        )

        assert decided.action.members == ("infer-c", "infer-a")

    def test_the_same_inputs_select_the_same_ensemble(self) -> None:
        seen = labeled(
            ("infer-a", "infer-b"),
            {"infer-a": (True, False, False, False), "infer-b": (True, True, False, False)},
            at=self.car,
        )

        first = self.built(exploration_alpha=0.25).decide(seen)
        second = self.built(exploration_alpha=0.25).decide(seen)

        assert first.action == second.action

    def test_without_a_label_in_this_period_it_keeps_the_ensemble(self) -> None:
        held = self.built()
        held.decide(labeled(("infer-a",), {"infer-a": (False,) * 4}, period="bus", at=self.car))

        decided = held.decide(
            labeled(("infer-a",), {"infer-a": (True,) * 4}, true_label=None, at=self.car)
        )

        assert decided.action.kind == "keep"

    def test_a_period_learned_before_is_decided_from_it_when_its_members_never_served_there(
        self,
    ) -> None:
        held = self.built()
        held.decide(labeled(("infer-c",), {"infer-c": (True,) * 4}, period="car", at=self.car))

        decided = held.decide(
            labeled(("infer-a",), {"infer-a": (True,) * 4}, true_label=None, at=self.car)
        )

        assert decided.action.kind == "select"
        assert decided.action.members[0] == "infer-c"

    def test_what_it_learns_in_one_period_it_does_not_use_in_another(self) -> None:
        held = self.built()
        held.decide(labeled(("infer-a",), {"infer-a": (False,) * 4}, period="car", at=self.car))
        held.decide(labeled(("infer-a",), {"infer-a": (True,) * 24}, period="bus", at=self.car))

        at_car = held.decide(labeled(("infer-a",), {"infer-a": ()}, at=self.car))
        at_bus = held.decide(labeled(("infer-a",), {"infer-a": ()}, at=self.bus))

        assert at_car.action.members == ("infer-a", "infer-b", "infer-c")
        assert at_bus.action.members == ("infer-a",)


class TestWhenThePoolHoldsNothingToChooseFrom:
    def nothing(self) -> Observation:
        return observed(serving=("infer-a",), right=4, held={})

    def test_cocktail_has_nothing_to_add(self) -> None:
        held = Cocktail(
            max_ensemble_size=3,
            evaluation_window=500,
            accuracy_target=0.85,
            accuracy_target_range=0.05,
        ).decide(observed(serving=("infer-a",), right=1, wrong=1, held={}))

        assert held.action.kind == "keep"

    def test_cocktail_has_nothing_to_remove(self) -> None:
        held = Cocktail(
            max_ensemble_size=3,
            evaluation_window=500,
            accuracy_target=0.85,
            accuracy_target_range=0.05,
        ).decide(observed(serving=("infer-a", "infer-b"), right=10, held={}))

        assert held.action.kind == "keep"

    def test_frugalml_has_nothing_to_price(self) -> None:
        held = FrugalMl(
            max_ensemble_size=3,
            evaluation_window=500,
            budget_per_1k=20.0,
            quality_level=0.8,
            failing_intervals=1,
        ).decide(observed(serving=("infer-a",), right=1, wrong=1, held={}))

        assert held.action.kind == "keep"

    def test_sneakpeek_has_nothing_to_rank(self) -> None:
        held = SneakPeek(classes=CLASSES, max_ensemble_size=2, evaluation_window=500).decide(
            self.nothing()
        )

        assert held.action.kind == "keep"

    def test_axiomvision_has_nothing_to_learn_about(self) -> None:
        held = AxiomVision(
            max_ensemble_size=1,
            evaluation_window=500,
            exploration_alpha=0.25,
            prior_zeta=1.0,
            accuracy_target=0.85,
        ).decide(self.nothing())

        assert held.action.kind == "keep"

    def test_cocktail_with_room_but_no_service_outside_keeps_the_ensemble(self) -> None:
        only = {"infer-a": profile(service_name="infer-a", instance_id="i-a")}

        held = Cocktail(
            max_ensemble_size=3,
            evaluation_window=500,
            accuracy_target=0.85,
            accuracy_target_range=0.05,
        ).decide(observed(serving=("infer-a",), right=1, wrong=1, held=only))

        assert held.action.kind == "keep"


def older_wrong_then_recent(*recent: Row) -> tuple[Row, ...]:
    return tuple(answered(f"q-old-{index}", "bus", "car") for index in range(8)) + recent


class TestEachBaselineReadsOnlyItsEvaluationWindow:
    recent = tuple(answered(f"q-{index}", "car", "car") for index in range(2))

    def test_cocktail_judges_the_accuracy_of_its_last_queries_only(self) -> None:
        held = Cocktail(
            max_ensemble_size=3,
            evaluation_window=2,
            accuracy_target=0.85,
            accuracy_target_range=0.05,
        ).decide(observed(serving=("infer-a",), given=older_wrong_then_recent(*self.recent)))

        assert held.action.kind == "keep"

    def test_cocktail_counts_the_votes_of_its_last_queries_only(self) -> None:
        older = tuple(
            {
                "query_id": f"q-old-{index}",
                "service_name": name,
                "predicted_label": label,
                "confidence": 0.8,
                "true_label": "car",
                "total_ms": 20.0,
            }
            for index in range(8)
            for name, label in zip(POOL, ("car", "bus", "truck"), strict=True)
        )
        base = voted(dict.fromkeys(POOL, ("car",) * 2))
        held = replace(
            base,
            window=Window(
                tuple(answered(f"q-old-{index}", "car", "car") for index in range(8))
                + tuple(base.window)
            ),
            member_window=Window(older + tuple(base.member_window)),
        )

        decided = Cocktail(
            max_ensemble_size=3,
            evaluation_window=2,
            accuracy_target=0.85,
            accuracy_target_range=0.05,
        ).decide(held)

        assert decided.action.kind == "remove"

    def test_frugalml_judges_the_accuracy_of_its_last_queries_only(self) -> None:
        held = FrugalMl(
            max_ensemble_size=3,
            evaluation_window=2,
            budget_per_1k=20.0,
            quality_level=0.8,
            failing_intervals=1,
        ).decide(observed(serving=("infer-a",), given=older_wrong_then_recent(*self.recent)))

        assert held.action.kind == "keep"

    def test_sneakpeek_reads_the_class_distribution_of_its_last_queries_only(self) -> None:
        older = tuple(answered(f"q-old-{index}", "bus", "bus") for index in range(8))
        held = SneakPeek(classes=CLASSES, max_ensemble_size=1, evaluation_window=2).decide(
            observed(
                serving=("infer-a",),
                given=older + self.recent,
                held=profiles(
                    **{
                        "infer-a": {"accuracy_by_class": {"car": 0.5, "bus": 0.5}},
                        "infer-b": {"accuracy_by_class": {"car": 0.9, "bus": 0.1}},
                        "infer-c": {"accuracy_by_class": {"car": 0.1, "bus": 0.9}},
                    }
                ),
            )
        )

        assert held.action.members == ("infer-b",)

    def test_axiomvision_learns_from_its_last_queries_only(self) -> None:
        car = context(period_name="car", period_index=1)
        held = AxiomVision(
            max_ensemble_size=3,
            evaluation_window=4,
            exploration_alpha=0.0,
            prior_zeta=1.0,
            accuracy_target=0.85,
        )

        held.decide(labeled(("infer-a",), {"infer-a": (False,) * 8 + (True,) * 4}, at=car))

        assert value(held, car, "infer-a") == pytest.approx(0.8)

    def test_scale_reduction_scores_the_labels_of_its_last_queries_only(self) -> None:
        queries = [(f"q-old-{index}", "truck") for index in range(8)] + [
            (f"q-{index}", "car") for index in range(2)
        ]
        base = observed(
            serving=("infer-a", "infer-b"),
            given=tuple(answered(query_id, label, label) for query_id, label in queries),
            held={
                name: one
                for name, one in profiles(
                    **{
                        "infer-a": {"accuracy_by_class": {"car": 0.9, "truck": 0.1}},
                        "infer-b": {"accuracy_by_class": {"car": 0.1, "truck": 0.9}},
                    }
                ).items()
                if name != "infer-c"
            },
        )

        decided = ScaleReduction(max_ensemble_size=2, evaluation_window=2).decide(base)

        assert decided.action.members == ("infer-a",)
