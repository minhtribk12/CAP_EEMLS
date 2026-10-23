from __future__ import annotations

import dataclasses
from collections.abc import Callable
from datetime import datetime

import numpy as np
import pytest

from eemls.domain.consumer_defined_metrics import CdmSet
from eemls.domain.records import InferenceServiceProfile
from eemls.domain.types import (
    EnsembleDeployment,
    EnsembleMember,
    Row,
    RuntimeContext,
)
from eemls.ensemble_optimization.approach import ConsumerSpecification, Observation
from eemls.ensemble_optimization.approach_loading import policy_of
from eemls.ensemble_optimization.approaches.cap.approach import (
    CONTEXT_KEYS,
    Cap,
    CapSettings,
    FeedbackDelayQueue,
    LearnedValues,
    PendingAction,
    is_wanted,
)
from eemls.ensemble_optimization.approaches.cap.context_signals import SIGNAL_NAMES
from eemls.ensemble_optimization.approaches.cap.identity_tables import IdentityTables
from eemls.ensemble_optimization.approaches.cap.learned_weights import Weights, standard_normal
from eemls.ensemble_optimization.approaches.cap.night_detection import NightDetector
from eemls.ensemble_optimization.approaches.cap.switching_margin import SwitchingMargin
from eemls.ensemble_optimization.testing.samples import GENERAL_ACCURACY, contract
from eemls.evaluation_engine.window import Window
from eemls.testing.records import AT, context, member, profile

CLASSES = ("car", "bus", "cyclist")

POOL = ("infer-a", "infer-b", "infer-c")


def settings(**changes: object) -> CapSettings:
    fields: dict[str, object] = {
        "kind": "cap",
        "max_ensemble_size": 3,
        "evaluation_window": 500,
        "initial_switching_margin": 0.25,
        "contributions_before_margin": 30,
        "prior_precision": 1.0,
        "noise_variance": 1.0,
        "night_confidence_drop": 0.1,
    }
    return CapSettings.model_validate(fields | changes)


def consumer(delay_intervals: int = 1) -> ConsumerSpecification:
    asked = contract().model_copy(
        update={"feedback": {"fraction": 0.5, "delay_intervals": delay_intervals}}
    )
    return ConsumerSpecification(
        contract=asked, cdms=CdmSet.model_validate({"definitions": GENERAL_ACCURACY})
    )


def answer(query_id: str, label: str, confidence: float, true_label: str | None) -> Row:
    return {
        "query_id": query_id,
        "predicted_label": label,
        "predicted_confidence": confidence,
        "true_label": true_label,
        "cost_per_1k": 7.0,
        "response_time_ms": 20.0,
    }


def served(
    query_id: str, service_name: str, label: str, confidence: float, true: str | None
) -> Row:
    return {
        "query_id": query_id,
        "service_name": service_name,
        "predicted_label": label,
        "confidence": confidence,
        "true_label": true,
        "total_ms": 20.0,
    }


def profiles(*names: str) -> dict[str, InferenceServiceProfile]:
    return {
        name: profile(
            service_name=name,
            instance_id=f"i-{name}",
            accuracy_by_class={"car": 0.9, "bus": 0.6, "cyclist": 0.5},
            confidence_by_class={"car": 0.88, "bus": 0.6, "cyclist": 0.5},
        )
        for name in names
    }


def observed(
    *,
    serving: tuple[str, ...] = ("infer-a",),
    labeled: bool = True,
    at: RuntimeContext | None = None,
    queries: int = 2,
    held: ConsumerSpecification | None = None,
    pool: tuple[str, ...] = POOL,
    wrong_on: tuple[int, ...] | None = None,
    t: datetime = AT,
    confidence: float = 0.8,
) -> Observation:
    rows = tuple(
        answer(f"q-{index}", "car" if index % 2 == 0 else "bus", confidence, None)
        if not labeled
        else answer(f"q-{index}", "car" if index % 2 == 0 else "bus", confidence, "car")
        for index in range(queries)
    )
    # The other members outvote the first member on the turned queries.
    turned = tuple(range(queries)) if wrong_on is None else wrong_on
    members = tuple(
        served(
            f"q-{index}",
            name,
            "car" if place == 0 or index not in turned else "bus",
            0.8 if place == 0 else 0.9,
            "car" if labeled else None,
        )
        for index in range(queries)
        for place, name in enumerate(serving)
    )
    return Observation(
        consumer=held if held is not None else consumer(),
        t=t,
        context=at if at is not None else context(),
        deployment=EnsembleDeployment(
            deployment_id="d-1", members=tuple(member_named(name) for name in serving)
        ),
        available=tuple(member_named(name) for name in pool),
        profiles=profiles(*POOL),
        window=Window(rows),
        member_window=Window(members),
        cdm_values={},
        score=None,
    )


def member_named(name: str) -> EnsembleMember:
    return member().model_copy(update={"service_name": name})


def built(seed: int = 0, **changes: object) -> Cap:
    held = settings(**changes)
    return Cap(
        classes=CLASSES,
        max_ensemble_size=held.max_ensemble_size,
        evaluation_window=held.evaluation_window,
        initial_switching_margin=held.initial_switching_margin,
        contributions_before_margin=held.contributions_before_margin,
        prior_precision=held.prior_precision,
        noise_variance=held.noise_variance,
        night_confidence_drop=held.night_confidence_drop,
        seed=seed,
    )


def instances(monkeypatch: pytest.MonkeyPatch, owner: type, method: str) -> list[object]:
    held: list[object] = []
    real: Callable[..., object] = getattr(owner, method)

    def spy(self: object, *args: object, **kwargs: object) -> object:
        held.append(self)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(owner, method, spy)
    return held


def taken(monkeypatch: pytest.MonkeyPatch) -> list[PendingAction]:
    held: list[PendingAction] = []
    real = FeedbackDelayQueue.push

    def push(queue: FeedbackDelayQueue, one: PendingAction) -> PendingAction | None:
        held.append(one)
        return real(queue, one)

    monkeypatch.setattr(FeedbackDelayQueue, "push", push)
    return held


def tables_of(held: list[object]) -> IdentityTables:
    tables = held[-1]
    assert isinstance(tables, IdentityTables)
    return tables


class TestTheQueueOfActionsWaitingForTheirFeedback:
    def record(self, name: str) -> PendingAction:
        return PendingAction(
            signals={name: (1.0,)}, context=context(period_name=name, period_index=1)
        )

    def test_nothing_comes_back_before_the_labels_of_its_queries_have_arrived(self) -> None:
        held = FeedbackDelayQueue(delay_intervals=2)

        assert held.push(self.record("one")) is None
        assert held.push(self.record("two")) is None
        assert held.push(self.record("three")) is None

    def test_the_action_comes_back_once_the_labels_of_its_interval_have_arrived(self) -> None:
        held = FeedbackDelayQueue(delay_intervals=2)

        for name in ("one", "two", "three"):
            held.push(self.record(name))
        due = held.push(self.record("four"))

        assert due is not None
        assert due.context.period_name == "one"

    def test_a_delay_of_one_interval_gives_back_the_action_two_decisions_later(self) -> None:
        held = FeedbackDelayQueue(delay_intervals=1)

        assert held.push(self.record("one")) is None
        assert held.push(self.record("two")) is None
        due = held.push(self.record("three"))

        assert due is not None and due.context.period_name == "one"

    def test_they_come_back_in_the_order_they_were_taken(self) -> None:
        held = FeedbackDelayQueue(delay_intervals=1)

        held.push(self.record("one"))
        held.push(self.record("two"))
        first = held.push(self.record("three"))
        second = held.push(self.record("four"))

        assert first is not None and first.context.period_name == "one"
        assert second is not None and second.context.period_name == "two"

    def test_it_holds_only_the_actions_still_waiting(self) -> None:
        held = FeedbackDelayQueue(delay_intervals=2)

        for name in ("one", "two", "three", "four", "five"):
            held.push(self.record(name))
        due = held.push(self.record("six"))

        assert due is not None and due.context.period_name == "three"


class TestOneActionComesOut:
    def test_a_decision_names_exactly_one_action(self) -> None:
        held = built().decide(observed())

        assert held.action.kind in ("keep", "add", "remove", "replace")

    def test_it_never_selects_a_whole_ensemble(self) -> None:
        cap = built()

        for _ in range(8):
            held = cap.decide(observed(serving=("infer-a", "infer-b")))

            assert held.action.kind != "select"
            assert held.action.members == ()

    def test_every_available_inference_service_is_weighed(self) -> None:
        held = built().decide(observed())

        assert held.candidates == len(POOL)

    def test_both_parts_of_the_work_are_timed(self) -> None:
        held = built().decide(observed())

        assert held.select_ms is not None and held.select_ms >= 0.0
        assert held.contribution_ms is not None and held.contribution_ms >= 0.0

    def test_it_decides_before_any_label_has_arrived(self) -> None:
        held = built().decide(observed(labeled=False))

        assert held.action.kind in ("keep", "add", "remove", "replace")

    def test_the_same_seed_decides_the_same_thing_twice(self) -> None:
        one = built().decide(observed())
        other = built().decide(observed())

        assert (one.action.kind, one.action.member_in) == (
            other.action.kind,
            other.action.member_in,
        )


class TestWhatItLearnsAndWhen:
    def test_nothing_is_learned_until_the_queue_gives_a_record_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        learned = instances(monkeypatch, Weights, "learn")

        built().decide(observed())

        assert learned == []

    def test_the_contributions_of_the_record_reach_the_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        learned = instances(monkeypatch, Weights, "learn")
        held = built()

        for _ in range(3):
            held.decide(observed(serving=("infer-a", "infer-b")))

        assert learned

    def test_the_margin_is_measured_from_the_contributions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        learned = instances(monkeypatch, SwitchingMargin, "learn")
        held = built(contributions_before_margin=1)

        for _ in range(3):
            held.decide(observed(serving=("infer-a", "infer-b")))

        assert learned

    def test_a_consumer_keeps_its_own_tables_and_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        learned = instances(monkeypatch, Weights, "learn")
        held = built()
        other = consumer().contract.model_copy(update={"consumer_id": "c2_safety"})

        for _ in range(3):
            held.decide(observed(serving=("infer-a", "infer-b")))
        before = len(learned)
        held.decide(
            observed(
                serving=("infer-a", "infer-b"),
                held=ConsumerSpecification(contract=other, cdms=consumer().cdms),
            )
        )

        assert before > 0
        assert len(learned) == before


class TestWhereInTheSimulatedDayItActs:
    def test_the_time_of_day_is_the_place_the_newest_query_was_sent_at(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        records = taken(monkeypatch)
        held = observed()
        rows = tuple({**row, "time_of_day": 0.1 * place} for place, row in enumerate(held.window))

        built().decide(dataclasses.replace(held, window=Window(rows)))

        place = SIGNAL_NAMES(CLASSES).index("time_of_day")
        assert [one[place] for one in records[0].signals.values()] == pytest.approx(
            [0.1] * len(POOL)
        )


class TestTheRuleThatDecidesWhetherTheTargetHoldsAService:
    def test_a_member_stays_while_the_difference_is_above_minus_the_margin(self) -> None:
        assert is_wanted(-0.2, is_member=True, margin=0.25)

    def test_a_member_below_minus_the_margin_has_left_the_target(self) -> None:
        assert not is_wanted(-0.3, is_member=True, margin=0.25)

    def test_a_member_exactly_at_minus_the_margin_has_left_the_target(self) -> None:
        assert not is_wanted(-0.25, is_member=True, margin=0.25)

    def test_an_outside_service_is_taken_above_the_margin(self) -> None:
        assert is_wanted(0.3, is_member=False, margin=0.25)

    def test_an_outside_service_below_the_margin_is_not_taken(self) -> None:
        assert not is_wanted(0.2, is_member=False, margin=0.25)

    def test_an_outside_service_exactly_at_the_margin_is_not_taken(self) -> None:
        assert not is_wanted(0.25, is_member=False, margin=0.25)

    def test_a_member_is_kept_where_an_outside_service_is_not_taken(self) -> None:
        assert is_wanted(0.0, is_member=True, margin=0.25)
        assert not is_wanted(0.0, is_member=False, margin=0.25)


class TestTheNoiseFollowsTheSpreadOfTheContributions:
    def test_small_contributions_do_not_flip_a_member_above_the_margin(self) -> None:
        held = LearnedValues(
            weights=Weights(dimension=1, prior_precision=1.0, draw=standard_normal(0)),
            margin=SwitchingMargin(initial=0.25, contributions_before_margin=30),
            tables=IdentityTables(CONTEXT_KEYS),
            queue=FeedbackDelayQueue(delay_intervals=1),
            condition=NightDetector(night_confidence_drop=0.1),
            noise_variance=1.0,
        )
        for contribution in np.random.default_rng(1).normal(0.1, 0.12, 60):
            held.learn((1.0,), context(), "infer-a", float(contribution))
        margin = held.margin.value()

        differences = [held.weights.sampled().scores((1.0,)).difference for _ in range(200)]

        assert held.weights.expected((1.0,)) * 2 > margin
        assert all(is_wanted(one, is_member=True, margin=margin) for one in differences)


def share_of(record: PendingAction) -> tuple[float, ...]:
    return next(iter(record.signals.values()))[: len(CLASSES)]


class TestHowFarBackItReads:
    def test_it_reads_the_window_its_own_file_declares(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        records = taken(monkeypatch)

        built(evaluation_window=1).decide(observed(queries=2))

        assert share_of(records[0]) == pytest.approx((0.0, 1.0, 0.0))

    def test_a_wider_window_reads_every_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        records = taken(monkeypatch)

        built(evaluation_window=500).decide(observed(queries=2))

        assert share_of(records[0]) == pytest.approx((0.5, 0.5, 0.0))


class TestLateFeedbackIsCreditedToTheContextThatDecided:
    def test_the_contribution_lands_under_the_period_of_the_action(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        learned = instances(monkeypatch, IdentityTables, "learn")
        held = built()
        morning = context(period_name="car", period_index=1)
        evening = context(period_name="bus", period_index=2)

        held.decide(observed(serving=("infer-a", "infer-b"), at=morning))
        held.decide(observed(serving=("infer-a", "infer-b"), at=evening))
        held.decide(observed(serving=("infer-a", "infer-b"), at=evening))

        tables = tables_of(learned)
        assert tables.by_key["period_name"].contribution(morning, "infer-b") == pytest.approx(-1.0)
        assert tables.by_key["period_name"].contribution(evening, "infer-b") == 0.0

    def test_the_contribution_lands_under_the_condition_of_the_action(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        learned = instances(monkeypatch, IdentityTables, "learn")
        held = built()
        by_day = context(period_name="car", period_index=1, condition="day")
        by_night = context(period_name="car", period_index=1, condition="night")

        held.decide(observed(serving=("infer-a", "infer-b"), at=by_day, confidence=0.9))
        held.decide(observed(serving=("infer-a", "infer-b"), at=by_day, confidence=0.5))
        held.decide(observed(serving=("infer-a", "infer-b"), at=by_day, confidence=0.9))
        held.decide(observed(serving=("infer-a", "infer-b"), at=by_day, confidence=0.9))

        tables = tables_of(learned)
        assert tables.by_key["condition"].contribution(by_day, "infer-b") == pytest.approx(-1.0)
        assert tables.by_key["condition"].contribution(by_night, "infer-b") == pytest.approx(-1.0)

    def test_the_condition_the_context_carries_is_not_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        learned = instances(monkeypatch, IdentityTables, "learn")
        held = built()
        by_day = context(period_name="car", period_index=1, condition="day")
        by_night = context(period_name="car", period_index=1, condition="night")

        for _ in range(3):
            held.decide(observed(serving=("infer-a", "infer-b"), at=by_night))

        tables = tables_of(learned)
        assert tables.by_key["condition"].contribution(by_day, "infer-b") == pytest.approx(-1.0)
        assert tables.by_key["condition"].contribution(by_night, "infer-b") == 0.0

    def test_a_service_that_was_not_available_then_is_not_credited_now(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        learned = instances(monkeypatch, IdentityTables, "learn")
        held = built()
        morning = context(period_name="car", period_index=1)

        held.decide(
            observed(serving=("infer-a", "infer-b"), at=morning, pool=("infer-a", "infer-b"))
        )
        held.decide(observed(serving=("infer-a", "infer-b"), at=morning, pool=POOL))
        held.decide(observed(serving=("infer-a", "infer-b"), at=morning, pool=POOL))

        tables = tables_of(learned)
        assert tables.by_key["period_name"].contribution(morning, "infer-b") == pytest.approx(-1.0)
        assert tables.by_key["period_name"].contribution(morning, "infer-c") == 0.0

    def test_only_the_queries_of_its_own_window_are_measured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        learned = instances(monkeypatch, IdentityTables, "learn")
        held = built(evaluation_window=1)
        morning = context(period_name="car", period_index=1)

        for _ in range(3):
            held.decide(
                observed(serving=("infer-a", "infer-b"), at=morning, queries=2, wrong_on=(0,))
            )

        tables = tables_of(learned)
        assert tables.by_key["period_name"].contribution(morning, "infer-b") == pytest.approx(0.0)

    def test_a_longer_delay_holds_the_credit_back_for_longer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        learned = instances(monkeypatch, Weights, "learn")
        held = built()
        patient = consumer(delay_intervals=3)

        for _ in range(4):
            held.decide(observed(serving=("infer-a", "infer-b"), held=patient))

        assert learned == []

        held.decide(observed(serving=("infer-a", "infer-b"), held=patient))

        assert learned


def other_consumer() -> ConsumerSpecification:
    other = consumer().contract.model_copy(update={"consumer_id": "c2_safety"})
    return ConsumerSpecification(contract=other, cdms=consumer().cdms)


def drawn(weights: object) -> float:
    assert isinstance(weights, Weights)
    return weights.sampled().scores([1.0] * len(SIGNAL_NAMES(CLASSES))).include


class TestEachConsumerDrawsFromItsOwnGenerator:
    def test_the_draws_of_a_consumer_do_not_depend_on_who_decided_before(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        weights = instances(monkeypatch, Weights, "sampled")
        built(seed=3).decide(observed(held=other_consumer()))
        after = built(seed=3)
        after.decide(observed())
        after.decide(observed(held=other_consumer()))

        assert drawn(weights[0]) == drawn(weights[2])

    def test_the_same_seed_draws_the_same_samples(self, monkeypatch: pytest.MonkeyPatch) -> None:
        weights = instances(monkeypatch, Weights, "sampled")
        built(seed=3).decide(observed())
        built(seed=3).decide(observed())

        assert drawn(weights[0]) == drawn(weights[1])

    def test_another_seed_draws_other_samples(self, monkeypatch: pytest.MonkeyPatch) -> None:
        weights = instances(monkeypatch, Weights, "sampled")
        built(seed=3).decide(observed())
        built(seed=4).decide(observed())

        assert drawn(weights[0]) != drawn(weights[1])

    def test_two_consumers_draw_other_samples(self, monkeypatch: pytest.MonkeyPatch) -> None:
        weights = instances(monkeypatch, Weights, "sampled")
        held = built(seed=3)
        held.decide(observed())
        held.decide(observed(held=other_consumer()))

        assert drawn(weights[0]) != drawn(weights[1])


class TestThePolicyBuiltFromItsFile:
    def test_the_same_run_seed_draws_the_same_samples(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        weights = instances(monkeypatch, Weights, "sampled")
        policy_of(settings(), classes=CLASSES, seed=7).decide(observed())
        policy_of(settings(), classes=CLASSES, seed=7).decide(observed())

        assert drawn(weights[0]) == drawn(weights[1])

    def test_another_run_seed_draws_other_samples(self, monkeypatch: pytest.MonkeyPatch) -> None:
        weights = instances(monkeypatch, Weights, "sampled")
        policy_of(settings(), classes=CLASSES, seed=7).decide(observed())
        policy_of(settings(), classes=CLASSES, seed=8).decide(observed())

        assert drawn(weights[0]) != drawn(weights[1])

    @pytest.mark.parametrize(("drop", "condition"), [(0.1, "night"), (0.3, "day")])
    def test_the_night_drop_its_own_file_declares_is_the_one_applied(
        self, monkeypatch: pytest.MonkeyPatch, drop: float, condition: str
    ) -> None:
        records = taken(monkeypatch)
        held = policy_of(settings(night_confidence_drop=drop), classes=CLASSES)
        held.decide(observed(confidence=0.9))
        held.decide(observed(confidence=0.75))

        assert records[-1].context.condition == condition
