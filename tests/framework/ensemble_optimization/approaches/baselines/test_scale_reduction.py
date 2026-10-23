from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace
from datetime import timedelta

import pytest

from eemls.domain.consumer_defined_metrics import CdmSet
from eemls.domain.records import InferenceServiceProfile
from eemls.ensemble_optimization.approach import ConsumerSpecification, Observation
from eemls.ensemble_optimization.approaches.baselines.scale_reduction import (
    ScaleReduction,
    labeled_rows,
    members_of,
    reduced_scales,
    scaled_up,
    scales_for,
    search_space,
)
from eemls.ensemble_optimization.testing.samples import (
    NIGHT_ACCURACY,
    RESPONSE_TIME_ONLY,
    THREE_MEMBER_ANSWERS,
    contract,
    live_window,
    night_contract,
    observed,
)
from eemls.evaluation_engine.window import Window
from eemls.testing.records import AT, profile

SERVICES = {"infer-a": "model-a", "infer-b": "model-b", "infer-c": "model-c"}


def pool(**changes: Mapping[str, object]) -> dict[str, InferenceServiceProfile]:
    return {
        name: profile(
            service_name=name, instance_id=f"i-{name}", model=model, **changes.get(name, {})
        )
        for name, model in SERVICES.items()
    }


def at_rate(held: Observation, rate_per_s: float) -> Observation:
    arrivals = tuple(
        {**row, "t_arrive": AT + timedelta(seconds=index / rate_per_s)}
        for index, row in enumerate(held.window)
    )
    return replace(held, window=Window(arrivals))


def five_models_on_two_device_types(slowest: float, fastest: float) -> dict[str, list[float]]:
    return {
        "m1": [slowest, 10.0],
        "m2": [4.0, 12.0],
        "m3": [8.0, 20.0],
        "m4": [12.0, 30.0],
        "m5": [fastest, fastest],
    }


def two_devices() -> dict[str, InferenceServiceProfile]:
    return {
        "infer-a": profile(
            service_name="infer-a", instance_id="i-a", model="m", throughput_rps=5.0
        ),
        "infer-b": profile(
            service_name="infer-b", instance_id="i-b", model="m", throughput_rps=5.0
        ),
        "infer-c": profile(
            service_name="infer-c", instance_id="i-c", model="n", throughput_rps=40.0
        ),
    }


def one_model_on_two_device_types() -> Observation:
    held = {
        "infer-a": profile(
            service_name="infer-a",
            instance_id="i-a",
            model="m",
            throughput_rps=16.0,
            cost_per_1k=1.0,
        ),
        "infer-b": profile(
            service_name="infer-b",
            instance_id="i-b",
            model="m",
            throughput_rps=5.0,
            cost_per_1k=3.0,
        ),
        "infer-c": profile(
            service_name="infer-c",
            instance_id="i-c",
            model="n",
            throughput_rps=40.0,
            cost_per_1k=1.0,
        ),
    }
    answers = {
        "infer-a": (("car", 0.6), ("bus", 0.6)),
        "infer-b": (("car", 0.9), ("bus", 0.9)),
        "infer-c": (("bus", 0.6), ("car", 0.9)),
    }
    rows = tuple(
        {
            "query_id": f"q-{index}",
            "service_name": name,
            "predicted_label": label,
            "confidence": confidence,
            "true_label": "car",
            "total_ms": 20.0,
        }
        for name, given in answers.items()
        for index, (label, confidence) in enumerate(given)
    )
    base = observed(serving=tuple(answers), right=2, held=held)
    return replace(base, member_window=Window(rows))


class TestScalesOfTheInferenceServices:
    def test_a_service_needs_the_replicas_that_carry_the_request_rate(self) -> None:
        assert scales_for(100.0, (3.1, 35.2)) == (33, 3)

    def test_one_replica_serves_a_rate_below_the_throughput(self) -> None:
        assert scales_for(1.0, (3.1, 35.2)) == (1, 1)

    def test_the_scales_are_divided_by_their_smallest_value(self) -> None:
        assert reduced_scales((15, 20, 25, 22)) == (1, 2, 2, 2)

    def test_the_division_shrinks_the_sum_of_the_scales(self) -> None:
        scales = (15, 20, 25, 22)

        assert (sum(scales), sum(reduced_scales(scales))) == (82, 7)

    def test_the_redundant_replicas_of_the_dearer_device_are_removed(self) -> None:
        held = {
            "infer-a": profile(service_name="infer-a", throughput_rps=10.0, cost_per_1k=1.0),
            "infer-b": profile(service_name="infer-b", throughput_rps=10.0, cost_per_1k=9.0),
        }

        kept = scaled_up({"infer-a": 1, "infer-b": 1}, by=2, profiles=held, rate_per_s=20.0)

        assert kept == {"infer-a": 2}

    def test_the_scaled_up_replicas_still_carry_the_rate(self) -> None:
        held = {
            "infer-a": profile(service_name="infer-a", throughput_rps=7.0, cost_per_1k=1.0),
        }

        kept = scaled_up({"infer-a": 2}, by=2, profiles=held, rate_per_s=20.0)

        assert kept == {"infer-a": 3}


class TestSearchSpaceOfFiveModelsOnTwoDeviceTypes:
    def test_up_to_three_of_five_models_form_25_ensembles(self) -> None:
        held = search_space(
            five_models_on_two_device_types(3.1, 35.2), rate_per_s=1.0, max_ensemble_size=3
        )

        assert held.ensembles == 25

    def test_one_request_per_second_gives_130_deployments(self) -> None:
        held = search_space(
            five_models_on_two_device_types(3.1, 35.2), rate_per_s=1.0, max_ensemble_size=3
        )

        assert (held.naive, held.scale_reduction) == (130, 130)

    def test_each_replica_is_placed_on_one_of_the_device_types(self) -> None:
        held = search_space({"m1": [3.1, 10.0]}, rate_per_s=100.0, max_ensemble_size=1)

        assert held.naive == 2**33

    def test_the_reduced_space_places_only_the_reduced_replicas(self) -> None:
        held = search_space(
            {"m1": [3.1, 10.0], "m5": [35.2, 35.2]}, rate_per_s=100.0, max_ensemble_size=2
        )

        assert held.scale_reduction == 2 + 2 + 2 ** (11 + 1)

    def test_the_reduced_space_grows_slower_than_the_naive_one(self) -> None:
        pool = five_models_on_two_device_types(3.1, 35.2)
        low = search_space(pool, rate_per_s=10.0, max_ensemble_size=3)
        high = search_space(pool, rate_per_s=100.0, max_ensemble_size=3)

        assert high.naive / low.naive > high.scale_reduction / low.scale_reduction > 1

    def test_the_slowest_model_sets_the_scale_of_a_model(self) -> None:
        held = search_space({"m1": [3.1, 35.2]}, rate_per_s=100.0, max_ensemble_size=1)

        assert held.naive == 2**33

    def test_more_device_types_widen_every_placement(self) -> None:
        held = search_space({"m1": [10.0]}, rate_per_s=20.0, max_ensemble_size=1, device_types=3)

        assert held.naive == 3**2


class TestScaleReductionSelection:
    def built(self, max_ensemble_size: int = 2) -> ScaleReduction:
        return ScaleReduction(max_ensemble_size=max_ensemble_size, evaluation_window=500)

    def test_it_selects_a_whole_ensemble(self) -> None:
        held = self.built().decide(observed(right=4, held=pool()))

        assert held.action.kind == "select"

    def test_every_ensemble_up_to_the_size_limit_is_a_candidate(self) -> None:
        held = self.built().decide(observed(right=4, held=pool()))

        assert held.candidates == 6

    def test_a_larger_limit_weighs_more_candidates(self) -> None:
        held = self.built(max_ensemble_size=3).decide(observed(right=4, held=pool()))

        assert held.candidates == 7

    def test_the_ensemble_it_serves_is_no_larger_than_the_limit(self) -> None:
        held = self.built().decide(observed(right=4, held=pool()))

        assert 1 <= len(held.action.members) <= 2

    def test_a_window_it_cannot_score_keeps_the_ensemble(self) -> None:
        held = self.built().decide(observed(right=0, wrong=0, held=pool()))

        assert held.action.kind == "keep"

    def test_a_tie_goes_to_the_smaller_ensemble(self) -> None:
        held = self.built().decide(observed(right=4, cost_weight=1.0, held=pool()))

        assert len(held.action.members) == 1

    def test_the_dearer_of_two_equal_services_is_not_the_one_chosen(self) -> None:
        held = ScaleReduction(max_ensemble_size=1, evaluation_window=500).decide(
            observed(
                right=4,
                held=pool(
                    **{
                        "infer-a": {"cost_per_1k": 1.0},
                        "infer-b": {"cost_per_1k": 20.0},
                        "infer-c": {"cost_per_1k": 25.0},
                    }
                ),
                cost_weight=1.0,
            )
        )

        assert held.action.members == ("infer-a",)

    def test_spare_replicas_make_a_cheaper_service_the_dearer_deployment(self) -> None:
        held = ScaleReduction(max_ensemble_size=1, evaluation_window=500).decide(
            observed(
                serving=tuple(SERVICES),
                right=4,
                held=pool(
                    **{
                        "infer-a": {"cost_per_1k": 10.0, "throughput_rps": 15.0},
                        "infer-b": {"cost_per_1k": 12.0, "throughput_rps": 20.0},
                        "infer-c": {"cost_per_1k": 25.0},
                    }
                ),
                cost_weight=1.0,
            )
        )

        assert held.action.members == ("infer-b",)

    def test_a_model_on_two_devices_places_each_replica_on_either(self) -> None:
        held = ScaleReduction(max_ensemble_size=2, evaluation_window=500).decide(
            observed(right=4, held=two_devices())
        )

        assert held.candidates == 2 + 1 + 2**4

    def test_a_surge_raises_the_scales_and_enlarges_the_search_space(self) -> None:
        held = observed(right=40, held=two_devices())
        policy = ScaleReduction(max_ensemble_size=2, evaluation_window=500)

        calm = policy.decide(at_rate(held, 20.0))
        surge = policy.decide(at_rate(held, 80.0))

        assert surge.candidates > calm.candidates

    def test_every_decision_reads_the_profiles_it_is_given(self) -> None:
        policy = ScaleReduction(max_ensemble_size=1, evaluation_window=500)
        cheap_a = pool(**{"infer-a": {"cost_per_1k": 1.0}, "infer-b": {"cost_per_1k": 20.0}})
        cheap_b = pool(**{"infer-a": {"cost_per_1k": 20.0}, "infer-b": {"cost_per_1k": 1.0}})

        first = policy.decide(observed(right=4, held=cheap_a, cost_weight=1.0))
        second = policy.decide(observed(right=4, held=cheap_b, cost_weight=1.0))

        assert (first.action.members, second.action.members) == (("infer-a",), ("infer-b",))

    def test_it_reports_the_time_of_the_selection(self) -> None:
        held = self.built().decide(observed(right=4, held=pool()))

        assert held.select_ms is not None
        assert held.select_ms >= 0.0


def three_member_answers_at_night() -> Observation:
    names = {"A": "infer-a", "B": "infer-b", "C": "infer-c"}
    rows = tuple(
        {
            "query_id": query_id,
            "service_name": names[name],
            "predicted_label": label,
            "confidence": confidence,
            "true_label": true_label,
            "total_ms": 20.0,
        }
        for query_id, true_label, answers in THREE_MEMBER_ANSWERS
        for name, label, confidence in answers
    )
    night = {
        "infer-a": {"pedestrian": (1.0, 0.6), "cyclist": (1.0, 0.5)},
        "infer-b": {"pedestrian": (1.0, 0.5), "cyclist": (1.0, 0.4)},
        "infer-c": {"pedestrian": (1.0, 0.7), "cyclist": (0.0, 0.95)},
    }
    held = pool(
        **{
            name: {
                "accuracy_by_class": {label: one[0] for label, one in classes.items()},
                "confidence_by_class": {label: one[1] for label, one in classes.items()},
            }
            for name, classes in night.items()
        }
    )
    base = observed(serving=tuple(names.values()), held=held)
    return replace(
        base,
        consumer=ConsumerSpecification(
            contract=night_contract(),
            cdms=CdmSet.model_validate({"definitions": NIGHT_ACCURACY}),
        ),
        window=live_window(("A", "B", "C")),
        member_window=Window(rows),
    )


class TestScaleReductionScoresAScopedMetric:
    def test_a_night_metric_keeps_the_service_wrong_at_night_out(self) -> None:
        decided = ScaleReduction(max_ensemble_size=3, evaluation_window=500).decide(
            three_member_answers_at_night()
        )

        assert decided.action.members == ("infer-a",)


class TestScaleReductionCountsAModelOnce:
    def test_a_model_on_two_device_types_appears_once_in_the_selected_ensemble(self) -> None:
        held = one_model_on_two_device_types()

        decided = ScaleReduction(max_ensemble_size=2, evaluation_window=500).decide(held)

        models = [held.profiles[name].model for name in decided.action.members]
        assert sorted(models) == sorted(set(models))

    def test_the_deployments_split_over_two_device_types_are_scored_too(self) -> None:
        decided = ScaleReduction(max_ensemble_size=2, evaluation_window=500).decide(
            one_model_on_two_device_types()
        )

        assert decided.scored == 5 + 3

    def test_the_count_of_candidates_is_unchanged(self) -> None:
        decided = ScaleReduction(max_ensemble_size=2, evaluation_window=500).decide(
            one_model_on_two_device_types()
        )

        assert decided.candidates == 2 + 1 + 2**4

    def test_the_service_with_the_most_replicas_votes_for_the_model(self) -> None:
        held = one_model_on_two_device_types().profiles

        assert members_of({"infer-a": 1, "infer-b": 3, "infer-c": 1}, held) == (
            "infer-b",
            "infer-c",
        )

    def test_a_tie_in_replicas_goes_to_the_cheaper_service(self) -> None:
        held = one_model_on_two_device_types().profiles

        assert members_of({"infer-a": 2, "infer-b": 2}, held) == ("infer-a",)

    def test_a_tie_in_replicas_and_cost_goes_to_the_first_name(self) -> None:
        held = {
            name: profile(service_name=name, instance_id=f"i-{name}", model="m")
            for name in ("infer-b", "infer-a")
        }

        assert members_of({"infer-b": 1, "infer-a": 1}, held) == ("infer-a",)


class TestScaleReductionWithNothingToChooseFrom:
    def test_scale_reduction_has_no_candidate_to_score(self) -> None:
        held = ScaleReduction(max_ensemble_size=2, evaluation_window=500).decide(
            observed(right=4, held={})
        )

        assert held.action.kind == "keep"
        assert held.candidates == 0

    def test_scale_reduction_scores_no_candidate_it_cannot_answer_for(self) -> None:
        unanswerable = ConsumerSpecification(
            contract=contract(),
            cdms=CdmSet.model_validate({"definitions": RESPONSE_TIME_ONLY}),
        )
        held = observed(right=4, held=pool())

        decided = ScaleReduction(max_ensemble_size=2, evaluation_window=500).decide(
            replace(held, consumer=unanswerable)
        )

        assert decided.action.kind == "keep"
        assert decided.candidates == 6

    def test_a_service_with_no_throughput_cannot_be_scaled(self) -> None:
        held = ScaleReduction(max_ensemble_size=1, evaluation_window=500).decide(
            observed(
                right=4,
                held=pool(**{"infer-b": {"throughput_rps": 0.0, "n_queries": 0}}),
            )
        )

        assert held.candidates == 2


class TestScaleReductionChargesSpareCapacity:
    def fast_and_slow(self) -> Observation:
        return observed(
            right=4,
            held=pool(
                **{
                    "infer-a": {"throughput_rps": 35.2},
                    "infer-b": {"throughput_rps": 3.1},
                    "infer-c": {"cost_per_1k": 25.0},
                }
            ),
            cost_weight=1.0,
        )

    def test_of_two_services_of_equal_cost_the_one_with_less_spare_capacity_wins(self) -> None:
        decided = ScaleReduction(max_ensemble_size=1, evaluation_window=500).decide(
            self.fast_and_slow()
        )

        assert decided.action.members == ("infer-b",)


def with_a_past_member(base: Observation, rows: tuple[dict[str, object], ...]) -> Observation:
    return replace(base, member_window=Window((*base.member_window, *rows)))


def wrong_old_answers(name: str, queries: int) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "query_id": f"q-old-{index}",
            "service_name": name,
            "predicted_label": "bus",
            "confidence": 0.9,
            "true_label": "car",
            "total_ms": 20.0,
        }
        for index in range(queries)
    )


class TestScaleReductionEstimatesNonMembersFromProfiles:
    def past_member(self) -> Observation:
        changes: dict[str, Mapping[str, object]] = {
            "infer-b": {
                "cost_per_1k": 1.0,
                "accuracy_by_class": {"car": 1.0},
                "confidence_by_class": {"car": 0.88},
            },
            "infer-c": {"cost_per_1k": 25.0},
        }
        base = observed(right=4, held=pool(**changes), cost_weight=1.0)
        return with_a_past_member(base, wrong_old_answers("infer-b", 4))

    def test_a_past_member_is_scored_from_its_profile_not_its_old_rows(self) -> None:
        held = ScaleReduction(max_ensemble_size=1, evaluation_window=500).decide(self.past_member())

        assert held.action.members == ("infer-b",)

    def test_every_service_answers_every_labeled_query(self) -> None:
        held = self.past_member()

        rows = labeled_rows(held, held.profiles)

        queries = {
            name: {row.query_id for row in rows if row.service_name == name}
            for name in held.profiles
        }
        labeled = {f"q-{index}" for index in range(4)} | {f"q-old-{index}" for index in range(4)}
        assert all(answered == labeled for answered in queries.values())

    def test_no_service_is_scored_from_its_real_rows(self) -> None:
        held = observed(serving=("infer-a", "infer-b"), right=4, held=pool())

        rows = labeled_rows(held, held.profiles)

        assert {row.confidence for row in rows} == {0.88}


class TestScaleReductionSelectsFromTheProfilesOnly:
    def serving(self, name: str) -> Observation:
        held = pool(
            **{
                "infer-a": {"accuracy_by_class": {"car": 0.9}},
                "infer-b": {"accuracy_by_class": {"car": 0.8}},
                "infer-c": {"accuracy_by_class": {"car": 0.1}},
            }
        )
        return observed(serving=(name,), right=5, wrong=5, held=held)

    def test_the_choice_does_not_depend_on_what_serves_now(self) -> None:
        choose = ScaleReduction(max_ensemble_size=1, evaluation_window=500)

        chosen = {
            choose.decide(self.serving(name)).action.members for name in ("infer-a", "infer-b")
        }

        assert chosen == {("infer-a",)}


class TestScaleReductionReportsBothCounts:
    def test_the_candidates_are_the_count_of_the_reduced_space(self) -> None:
        held = ScaleReduction(max_ensemble_size=2, evaluation_window=500).decide(
            observed(right=4, held=two_devices())
        )

        assert held.candidates == 2 + 1 + 2**4

    def test_the_count_of_deployments_it_scores_is_reported_beside_the_candidates(
        self,
    ) -> None:
        held = ScaleReduction(max_ensemble_size=2, evaluation_window=500).decide(
            observed(right=4, held=two_devices())
        )

        assert (held.candidates, held.scored) == (2 + 1 + 2**4, 2 + 1 + 5)

    def test_the_deployments_it_scores_are_logged_beside_the_candidates(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        held = observed(right=4, held=two_devices())

        with caplog.at_level(logging.INFO):
            ScaleReduction(max_ensemble_size=2, evaluation_window=500).decide(held)

        assert "stage_tag=select process_tag=scale_reduction" in caplog.text
        assert f"candidates={2 + 1 + 2**4} scored={2 + 1 + 5}" in caplog.text

    def test_the_logged_count_is_the_number_of_scored_deployments(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        held = one_model_on_two_device_types()

        with caplog.at_level(logging.INFO):
            decided = ScaleReduction(max_ensemble_size=2, evaluation_window=500).decide(held)

        assert f"scored={decided.scored} " in caplog.text


def test_the_request_rate_comes_from_its_last_queries_only() -> None:
    older = at_rate(observed(right=36, held=two_devices()), 80.0)
    recent = at_rate(observed(right=4, held=two_devices()), 20.0)
    later = AT + timedelta(seconds=36 / 80.0 + 1.0)
    shifted = tuple(
        {**row, "query_id": f"q-new-{index}", "t_arrive": later + timedelta(seconds=index / 20.0)}
        for index, row in enumerate(recent.window)
    )
    answers = tuple(
        {**row, "query_id": f"q-new-{str(row['query_id']).removeprefix('q-')}"}
        for row in recent.member_window
    )
    held = replace(
        older,
        window=Window((*older.window, *shifted)),
        member_window=Window((*older.member_window, *answers)),
    )
    policy = ScaleReduction(max_ensemble_size=2, evaluation_window=4)

    assert policy.decide(held).candidates == policy.decide(recent).candidates


def a_fine_split_is_cheapest() -> Observation:
    held = {
        "infer-a": profile(
            service_name="infer-a",
            instance_id="i-a",
            model="m",
            throughput_rps=16.0,
            cost_per_1k=1.0,
        ),
        "infer-b": profile(
            service_name="infer-b",
            instance_id="i-b",
            model="m",
            throughput_rps=5.0,
            cost_per_1k=2.0,
        ),
        "infer-c": profile(
            service_name="infer-c",
            instance_id="i-c",
            model="n",
            throughput_rps=20.0,
            cost_per_1k=1.5,
        ),
    }
    return observed(serving=tuple(held), right=4, held=held, cost_weight=1.0)


class TestScaleReductionChargesEveryReplica:
    def test_over_provisioning_makes_the_cheaper_service_the_dearer_deployment(self) -> None:
        held = {
            "infer-a": profile(
                service_name="infer-a",
                instance_id="i-a",
                model="m",
                throughput_rps=10.0,
                cost_per_1k=2.0,
            ),
            "infer-b": profile(
                service_name="infer-b",
                instance_id="i-b",
                model="n",
                throughput_rps=15.0,
                cost_per_1k=1.8,
            ),
        }

        decided = ScaleReduction(max_ensemble_size=1, evaluation_window=500).decide(
            observed(right=4, held=held, cost_weight=1.0)
        )

        assert decided.action.members == ("infer-a",)

    def test_the_reduced_space_misses_the_cheapest_split_of_a_model(self) -> None:
        held = a_fine_split_is_cheapest()
        fast, slow = held.profiles["infer-a"], held.profiles["infer-b"]
        naive_best = min(
            (
                count_fast * fast.cost_per_1k * fast.throughput_rps
                + count_slow * slow.cost_per_1k * slow.throughput_rps
            )
            / 20.0
            for count_fast in range(5)
            for count_slow in range(5)
            if count_fast * fast.throughput_rps + count_slow * slow.throughput_rps >= 20.0
        )

        decided = ScaleReduction(max_ensemble_size=1, evaluation_window=500).decide(held)

        assert decided.action.members == ("infer-c",)
        assert [naive_best, held.profiles["infer-c"].cost_per_1k] == pytest.approx([1.3, 1.5])
