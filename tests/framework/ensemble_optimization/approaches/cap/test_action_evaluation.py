from __future__ import annotations

from typing import Any

import pytest

from eemls.domain.consumer_defined_metrics import CdmSet
from eemls.domain.contract import ConsumerContract
from eemls.domain.records import InferenceServiceProfile
from eemls.domain.types import Row
from eemls.ensemble_optimization.approach import ConsumerSpecification
from eemls.ensemble_optimization.approaches.cap.action_evaluation import (
    CandidateScores,
    contributions,
)
from eemls.ensemble_optimization.testing.samples import (
    GENERAL_ACCURACY,
    NIGHT_ACCURACY,
    THREE_MEMBER_ANSWERS,
    contract,
    declared,
    live_window,
    night_contract,
    row,
    scored,
    three_member_answers,
)
from eemls.evaluation_engine.ensemble_scoring import MemberRow, answered_by
from eemls.evaluation_engine.profile_estimates import member_rows
from eemls.evaluation_engine.window import Window
from eemls.testing.records import profile


def served(
    query_id: str, service_name: str, label: str | None, confidence: float | None, true_label: str
) -> Row:
    return {
        "query_id": query_id,
        "service_name": service_name,
        "predicted_label": label,
        "confidence": confidence,
        "true_label": true_label,
        "total_ms": 20.0,
    }


def three_member_window() -> Window:
    return Window(
        tuple(
            served(query_id, name, label, confidence, true_label)
            for query_id, true_label, answers in THREE_MEMBER_ANSWERS
            for name, label, confidence in answers
        )
    )


def priced(*names: str, **changes: object) -> dict[str, InferenceServiceProfile]:
    return {name: profile(service_name=name, instance_id=f"i-{name}", **changes) for name in names}


def contribution(
    asked: ConsumerContract,
    cdms: dict[str, Any],
    rows: tuple[MemberRow, ...],
    *,
    ensemble: tuple[str, ...],
    member: str,
    window: Window | None = None,
) -> float | None:
    scores = CandidateScores(
        ConsumerSpecification(contract=asked, cdms=declared(cdms)),
        Window(()) if window is None else window,
    )
    return scores.contribution(rows, ensemble, member, answered_by(rows, ensemble))


def consumer() -> ConsumerSpecification:
    return ConsumerSpecification(
        contract=contract(), cdms=CdmSet.model_validate({"definitions": GENERAL_ACCURACY})
    )


def asked(
    window: Window,
    *,
    ensemble: tuple[str, ...],
    available: tuple[str, ...],
    profiles: dict[str, InferenceServiceProfile],
) -> dict[str, float]:
    return dict(
        contributions(
            consumer(),
            member_window=window,
            ensemble=ensemble,
            available=available,
            profiles=profiles,
            evaluation_window=500,
            window=Window(()),
        )
    )


class TestTheContributionOfOneMemberToTheEnsemble:
    def rows(self) -> tuple[MemberRow, ...]:
        answers = []
        for index in range(10):
            true_label = "car" if index % 2 == 0 else "truck"
            answers.append(
                row(
                    query_id=f"q-{index}",
                    service_name="infer-car",
                    predicted_label="car",
                    confidence=0.9 if true_label == "car" else 0.3,
                    true_label=true_label,
                )
            )
            answers.append(
                row(
                    query_id=f"q-{index}",
                    service_name="infer-truck",
                    predicted_label="truck",
                    confidence=0.9 if true_label == "truck" else 0.3,
                    true_label=true_label,
                )
            )
        return tuple(answers)

    def test_a_night_metric_scores_the_contribution_of_a_member(self) -> None:
        held = contribution(
            night_contract(),
            NIGHT_ACCURACY,
            three_member_answers(),
            ensemble=("A", "B", "C"),
            member="C",
            window=live_window(("A", "B", "C")),
        )

        assert held == pytest.approx(-0.5)

    def test_the_contribution_of_c_is_the_quarter_it_costs(self) -> None:
        held = contribution(
            contract(),
            GENERAL_ACCURACY,
            three_member_answers(),
            ensemble=("A", "B", "C"),
            member="C",
        )

        assert held == pytest.approx(-0.25)

    def test_a_duplicate_member_adds_about_nothing(self) -> None:
        rows = list(self.rows())
        for answer in self.rows():
            if answer.service_name == "infer-car":
                rows.append(answer.model_copy(update={"service_name": "infer-car-copy"}))

        added = contribution(
            contract(),
            GENERAL_ACCURACY,
            tuple(rows),
            ensemble=("infer-car", "infer-truck", "infer-car-copy"),
            member="infer-car-copy",
        )
        on_its_own = scored(contract(), GENERAL_ACCURACY, tuple(rows), ("infer-car-copy",))

        assert added == pytest.approx(0.0, abs=1e-9)
        assert on_its_own.total == pytest.approx(0.5)

    def test_a_member_that_covers_what_the_others_miss_adds_a_lot(self) -> None:
        added = contribution(
            contract(),
            GENERAL_ACCURACY,
            self.rows(),
            ensemble=("infer-car", "infer-truck"),
            member="infer-truck",
        )

        assert added == pytest.approx(0.5)

    def test_a_member_outside_the_ensemble_is_measured_by_what_it_would_add(self) -> None:
        added = contribution(
            contract(),
            GENERAL_ACCURACY,
            self.rows(),
            ensemble=("infer-car",),
            member="infer-truck",
        )

        assert added == pytest.approx(0.5)

    def test_both_ensembles_are_scored_on_the_queries_the_ensemble_answered(self) -> None:
        rows = (
            row(query_id="q-1", service_name="infer-a", confidence=0.9),
            row(query_id="q-1", service_name="infer-b", confidence=0.8),
            row(query_id="q-2", service_name="infer-b", confidence=0.8),
        )

        added = contribution(
            contract(),
            GENERAL_ACCURACY,
            rows,
            ensemble=("infer-a", "infer-b"),
            member="infer-b",
        )

        assert added == pytest.approx(0.5)

    def test_an_outside_member_is_scored_only_on_the_queries_the_ensemble_answered(self) -> None:
        rows = (
            row(query_id="q-1", service_name="infer-a", confidence=0.9),
            row(query_id="q-1", service_name="infer-d", predicted_label="bus", confidence=0.1),
            row(query_id="q-2", service_name="infer-d", predicted_label="bus", confidence=0.1),
        )

        added = contribution(
            contract(),
            GENERAL_ACCURACY,
            rows,
            ensemble=("infer-a",),
            member="infer-d",
        )

        assert added == pytest.approx(0.0)

    def test_removing_the_only_member_cannot_be_measured(self) -> None:
        added = contribution(
            contract(),
            GENERAL_ACCURACY,
            self.rows(),
            ensemble=("infer-car",),
            member="infer-car",
        )

        assert added is None

    def test_a_window_with_no_labeled_query_cannot_be_scored(self) -> None:
        unlabeled = tuple(answer.model_copy(update={"true_label": None}) for answer in self.rows())

        added = contribution(
            contract(),
            GENERAL_ACCURACY,
            unlabeled,
            ensemble=("infer-car", "infer-truck"),
            member="infer-truck",
        )

        assert added is None


class TestTheContributionOfAMember:
    def test_the_example_of_the_article_comes_out_of_the_window(self) -> None:
        held = asked(
            three_member_window(),
            ensemble=("A", "B", "C"),
            available=("A", "B", "C"),
            profiles=priced("A", "B", "C"),
        )

        assert held["C"] == pytest.approx(-0.25)

    def test_a_night_metric_measures_a_member_on_the_night_queries(self) -> None:
        held = contributions(
            ConsumerSpecification(
                contract=night_contract(),
                cdms=CdmSet.model_validate({"definitions": NIGHT_ACCURACY}),
            ),
            member_window=three_member_window(),
            ensemble=("A", "B", "C"),
            available=("A", "B", "C"),
            profiles=priced("A", "B", "C"),
            evaluation_window=500,
            window=live_window(("A", "B", "C")),
        )

        assert held["C"] == pytest.approx(-0.5)

    def test_a_member_the_others_cannot_do_without_contributes_nothing_measurable(self) -> None:
        held = asked(
            three_member_window(),
            ensemble=("A",),
            available=("A",),
            profiles=priced("A"),
        )

        assert held == {}

    def test_a_member_with_no_labeled_query_is_skipped_and_not_given_a_zero(self) -> None:
        window = Window(
            (
                served("q-1", "A", "car", 0.9, "car"),
                served("q-1", "B", "car", 0.7, "car"),
                served("q-2", "A", "bus", 0.8, "bus"),
                served("q-2", "B", "bus", 0.6, "bus"),
            )
        )

        held = asked(
            window,
            ensemble=("A", "B", "C"),
            available=("A", "B", "C"),
            profiles=priced("A", "B", "C"),
        )

        assert "C" not in held
        assert set(held) == {"A", "B"}


class TestBothScoresAreOnTheSameLabeledQueries:
    def test_an_outsider_that_is_always_outvoted_contributes_nothing(self) -> None:
        window = Window(
            (
                served("q-1", "A", "car", 0.9, "car"),
                served("q-2", "X", "car", 0.9, "car"),
                served("q-3", "A", "bus", 0.9, "bus"),
            )
        )
        never_right = priced("A", "X") | priced(
            "D", accuracy_by_class={"car": 0.0}, confidence_by_class={"car": 0.1}
        )

        held = asked(window, ensemble=("A",), available=("D",), profiles=never_right)

        assert held["D"] == pytest.approx(0.0)

    def test_a_member_is_measured_only_on_the_queries_it_answered(self) -> None:
        window = Window(
            (
                served("q-1", "A", "car", 0.9, "car"),
                served("q-2", "A", "car", 0.5, "car"),
                served("q-2", "B", "bus", 0.9, "car"),
            )
        )

        held = asked(window, ensemble=("A", "B"), available=("A", "B"), profiles=priced("A", "B"))

        assert held["B"] == pytest.approx(-1.0)

    def test_a_query_only_the_member_answered_counts_without_it_as_unanswered(self) -> None:
        window = Window(
            (
                served("q-1", "A", "car", 0.9, "car"),
                served("q-1", "B", "car", 0.8, "car"),
                served("q-2", "B", "car", 0.8, "car"),
            )
        )

        held = asked(window, ensemble=("A", "B"), available=("A", "B"), profiles=priced("A", "B"))

        assert held["B"] == pytest.approx(0.5)
        assert held["A"] == pytest.approx(0.0)


class TestTheEvaluationWindowOfAMember:
    def window(self) -> Window:
        return Window(
            (
                served("q-1", "A", "bus", 0.9, "car"),
                served("q-1", "B", "car", 0.5, "car"),
                served("q-2", "A", "car", 0.9, "car"),
                served("q-2", "B", "car", 0.5, "car"),
            )
        )

    def measured(self, evaluation_window: int) -> dict[str, float]:
        return dict(
            contributions(
                consumer(),
                member_window=self.window(),
                ensemble=("A", "B"),
                available=("A", "B"),
                profiles=priced("A", "B"),
                evaluation_window=evaluation_window,
                window=Window(()),
            )
        )

    def test_it_holds_only_the_most_recent_predictions_of_the_member(self) -> None:
        assert self.measured(1)["A"] == pytest.approx(0.0)

    def test_a_wider_one_reaches_the_older_predictions(self) -> None:
        assert self.measured(2)["A"] == pytest.approx(-0.5)


class TestTheContributionOfAnInferenceServiceOutsideTheEnsemble:
    def test_it_is_estimated_from_its_own_profile(self) -> None:
        held = asked(
            three_member_window(),
            ensemble=("A", "B"),
            available=("A", "B", "D"),
            profiles=priced("A", "B", "D"),
        )

        assert "D" in held

    def test_an_inference_service_with_no_profile_cannot_be_estimated(self) -> None:
        held = asked(
            three_member_window(),
            ensemble=("A", "B"),
            available=("A", "B", "D"),
            profiles=priced("A", "B"),
        )

        assert "D" not in held

    def test_a_profile_saying_nothing_of_the_classes_in_the_window_is_not_credited(self) -> None:
        held = asked(
            three_member_window(),
            ensemble=("A", "B"),
            available=("A", "B", "D"),
            profiles=priced("A", "B")
            | priced("D", accuracy_by_class={"boat": 0.9}, confidence_by_class={"boat": 0.9}),
        )

        assert "D" not in held

    def test_the_stronger_of_two_contributes_more(self) -> None:
        window = Window(
            (
                served("q-1", "A", "bus", 0.5, "car"),
                served("q-2", "A", "bus", 0.4, "car"),
            )
        )
        strong = priced("A") | priced(
            "D", accuracy_by_class={"car": 0.95}, confidence_by_class={"car": 0.95}
        )
        weak = priced("A") | priced(
            "D", accuracy_by_class={"car": 0.10}, confidence_by_class={"car": 0.10}
        )

        better = asked(window, ensemble=("A",), available=("D",), profiles=strong)
        worse = asked(window, ensemble=("A",), available=("D",), profiles=weak)

        assert better["D"] == pytest.approx(1.0)
        assert worse["D"] == pytest.approx(0.0)

    def test_one_that_is_often_wrong_on_the_class_lowers_the_score(self) -> None:
        window = Window(
            (
                served("q-1", "A", "car", 0.6, "car"),
                served("q-2", "A", "bus", 0.6, "bus"),
            )
        )
        wrong = priced("A") | priced(
            "D",
            accuracy_by_class={"car": 0.0, "bus": 1.0},
            confidence_by_class={"car": 0.9, "bus": 0.9},
        )

        held = asked(window, ensemble=("A",), available=("D",), profiles=wrong)

        assert held["D"] == pytest.approx(-0.5)

    def test_what_it_costs_counts_against_what_it_adds(self) -> None:
        window = Window((served("q-1", "A", "bus", 0.5, "car"),))
        cheap = priced("A") | priced(
            "D", cost_per_1k=1.0, accuracy_by_class={"car": 0.9}, confidence_by_class={"car": 0.9}
        )
        dear = priced("A") | priced(
            "D", cost_per_1k=25.0, accuracy_by_class={"car": 0.9}, confidence_by_class={"car": 0.9}
        )

        held = ConsumerSpecification(
            contract=contract(cost_weight=1.0),
            cdms=CdmSet.model_validate({"definitions": GENERAL_ACCURACY}),
        )
        less = dict(
            contributions(
                held,
                member_window=window,
                ensemble=("A",),
                available=("D",),
                profiles=dear,
                evaluation_window=500,
                window=Window(()),
            )
        )
        more = dict(
            contributions(
                held,
                member_window=window,
                ensemble=("A",),
                available=("D",),
                profiles=cheap,
                evaluation_window=500,
                window=Window(()),
            )
        )

        assert more["D"] > less["D"]


RESPONSE_TIME = {
    "response_time_ms": {
        "measured": "per_window",
        "window": 500,
        "unit": "milliseconds",
        "description": "Mean time from the query arriving to the result being ready",
        "specification": {"function": "average", "attribute": "response_time_ms"},
    }
}


class TestOnlyTheLabeledQueriesAreMeasured:
    def timed(self) -> ConsumerSpecification:
        return ConsumerSpecification(
            contract=ConsumerContract.model_validate(
                {
                    "consumer_id": "c1_navigation",
                    "description": "Wants its answers quickly",
                    "prioritized_classes": ("car",),
                    "request_rate_per_s": 20.0,
                    "objectives": {
                        "response_time_ms": {
                            "min_value": 0.0,
                            "max_value": 500.0,
                            "weight": 1.0,
                            "direction": "minimize",
                            "mapping": "linear",
                            "mapping_base": None,
                        }
                    },
                    "cost_weight": 0.0,
                    "cost_range_max_per_1k": 30.0,
                    "feedback": {"fraction": 0.5, "delay_intervals": 15},
                }
            ),
            cdms=CdmSet.model_validate({"definitions": RESPONSE_TIME}),
        )

    def rows(self, *, b_unlabeled_ms: float) -> Window:
        return Window(
            (
                {**served("q-1", "A", "car", 0.9, "car"), "total_ms": 20.0},
                {**served("q-1", "B", "car", 0.7, "car"), "total_ms": 20.0},
                {
                    "query_id": "q-2",
                    "service_name": "A",
                    "predicted_label": "car",
                    "confidence": 0.9,
                    "true_label": None,
                    "total_ms": 20.0,
                },
                {
                    "query_id": "q-2",
                    "service_name": "B",
                    "predicted_label": "car",
                    "confidence": 0.7,
                    "true_label": None,
                    "total_ms": b_unlabeled_ms,
                },
            )
        )

    def test_an_unlabeled_query_does_not_change_what_a_member_contributed(self) -> None:
        held = self.timed()
        quick = dict(
            contributions(
                held,
                member_window=self.rows(b_unlabeled_ms=20.0),
                ensemble=("A", "B"),
                available=("A", "B"),
                profiles=priced("A", "B"),
                evaluation_window=500,
                window=Window(()),
            )
        )
        slow = dict(
            contributions(
                held,
                member_window=self.rows(b_unlabeled_ms=400.0),
                ensemble=("A", "B"),
                available=("A", "B"),
                profiles=priced("A", "B"),
                evaluation_window=500,
                window=Window(()),
            )
        )

        assert quick == pytest.approx(slow)


class TestWhenNothingCanBeMeasured:
    def test_a_window_with_no_labeled_query_gives_no_contribution_at_all(self) -> None:
        window = Window(
            (
                {
                    "query_id": "q-1",
                    "service_name": "A",
                    "predicted_label": "car",
                    "confidence": 0.9,
                    "true_label": None,
                    "total_ms": 20.0,
                },
            )
        )

        held = asked(window, ensemble=("A", "B"), available=("A", "B"), profiles=priced("A", "B"))

        assert held == {}

    def test_an_empty_window_gives_no_contribution_at_all(self) -> None:
        held = asked(
            Window(()), ensemble=("A", "B"), available=("A", "B"), profiles=priced("A", "B")
        )

        assert held == {}


class TestTheWorkOneEvaluationDoes:
    def test_the_ensemble_that_served_is_scored_once_for_every_service(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import eemls.evaluation_engine.ensemble_scoring as ensemble_scoring

        scored: list[tuple[str, ...]] = []
        real = ensemble_scoring.score_candidate

        def counted(*args: object, **kwargs: object) -> object:
            scored.append(tuple(args[3] if len(args) > 3 else kwargs["ensemble"]))  # type: ignore[arg-type]
            return real(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(ensemble_scoring, "score_candidate", counted)

        asked(
            three_member_window(),
            ensemble=("A", "B", "C"),
            available=("A", "B", "C", "D"),
            profiles=priced("A", "B", "C", "D"),
        )

        assert scored.count(("A", "B", "C")) == 1

    def test_the_contributions_are_the_ones_each_pair_of_scores_gives(self) -> None:
        rows = member_rows(three_member_window(), priced("A", "B", "C"))
        cdms = CdmSet.model_validate({"definitions": GENERAL_ACCURACY})
        from eemls.evaluation_engine.ensemble_scoring import score_candidate

        def total(ensemble: tuple[str, ...]) -> float:
            parts = score_candidate(contract(), cdms, rows, ensemble)
            return parts.total  # type: ignore[union-attr]

        held = asked(
            three_member_window(),
            ensemble=("A", "B", "C"),
            available=("A", "B", "C"),
            profiles=priced("A", "B", "C"),
        )

        assert held["C"] == pytest.approx(total(("A", "B", "C")) - total(("A", "B")))
        assert held["A"] == pytest.approx(total(("A", "B", "C")) - total(("B", "C")))
