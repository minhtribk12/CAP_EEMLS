from __future__ import annotations

import pytest
from pydantic import ValidationError

from eemls.domain.types import AggregatedResult, MemberPrediction
from eemls.orchestration.aggregation import AggregationError, aggregate_predictions
from eemls.testing.records import prediction


def answered(label: str, confidence: float, name: str = "a", **changes: object) -> MemberPrediction:
    return prediction(
        instance_id=f"i-{name}", predicted_label=label, confidence=confidence, **changes
    )


def silent(status: str, name: str = "z") -> MemberPrediction:
    return prediction(
        instance_id=f"i-{name}",
        predicted_label=None,
        confidence=None,
        status=status,
        inference_ms=0.0,
    )


def aggregating(*predictions: MemberPrediction, n_members: int = 0) -> AggregatedResult:
    return aggregate_predictions(
        query_id="q-1",
        deployment_id="c1_navigation-0003",
        n_members=n_members or max(len(predictions), 1),
        predictions=predictions,
    )


class TestTheLabelTheEnsembleAnswers:
    def test_members_that_agree_answer_with_their_mean_confidence(self) -> None:
        result = aggregating(answered("car", 0.9, "a"), answered("car", 0.7, "b"))

        assert result.predicted_label == "car"
        assert result.confidence == pytest.approx(0.8)

    def test_the_label_with_the_highest_mean_wins_and_not_the_one_with_most_votes(self) -> None:
        result = aggregating(
            answered("bus", 0.4, "a"), answered("bus", 0.5, "b"), answered("car", 0.9, "c")
        )

        assert result.predicted_label == "car"
        assert result.confidence == pytest.approx(0.9)

    def test_the_mean_of_the_winner_is_over_its_own_voters(self) -> None:
        result = aggregating(
            answered("car", 0.9, "a"), answered("car", 0.5, "b"), answered("bus", 0.1, "c")
        )

        assert result.confidence == pytest.approx(0.7)

    def test_it_names_the_deployment_that_answered_and_how_many_voted(self) -> None:
        result = aggregating(answered("car", 0.9, "a"), silent("timeout"), n_members=3)

        assert (result.query_id, result.deployment_id) == ("q-1", "c1_navigation-0003")
        assert (result.n_members, result.n_answered) == (3, 1)

    def test_it_cannot_say_more_members_answered_than_serve(self) -> None:
        with pytest.raises(ValidationError, match="more than serve"):
            aggregating(answered("car", 0.9, "a"), answered("car", 0.8, "b"), n_members=1)


class TestATie:
    def test_it_is_broken_by_the_label_however_many_members_named_each(self) -> None:
        result = aggregating(
            answered("car", 0.6, "a"), answered("car", 0.6, "b"), answered("bus", 0.6, "c")
        )

        assert result.predicted_label == "bus"

    def test_a_tie_is_broken_by_the_label_so_a_run_repeats(self) -> None:
        result = aggregating(answered("car", 0.6, "a"), answered("bus", 0.6, "b"))

        assert result.predicted_label == "bus"

    def test_the_order_the_members_answered_in_never_changes_the_answer(self) -> None:
        one = answered("car", 0.6, "a")
        two = answered("bus", 0.6, "b")
        three = answered("bus", 0.6, "c")

        assert aggregating(one, two, three) == aggregating(three, one, two)


class TestWhoVotes:
    @pytest.mark.parametrize("status", ["timeout", "error", "refused"])
    def test_a_member_that_did_not_answer_does_not_vote(self, status: str) -> None:
        result = aggregating(answered("car", 0.9, "a"), silent(status))

        assert (result.predicted_label, result.n_answered) == ("car", 1)

    def test_a_member_outside_the_ensemble_is_recorded_but_does_not_vote(self) -> None:
        result = aggregating(
            answered("car", 0.5, "a"), answered("bus", 0.99, "b", in_ensemble=False)
        )

        assert (result.predicted_label, result.n_answered) == ("car", 1)


class TestWhenNobodyAnswers:
    def test_an_ensemble_whose_members_all_failed_answers_nothing(self) -> None:
        with pytest.raises(AggregationError, match="q-1"):
            aggregating(silent("timeout", "a"), silent("error", "b"))

    def test_an_ensemble_nobody_was_asked_of_answers_nothing(self) -> None:
        with pytest.raises(AggregationError, match="q-1"):
            aggregating()

    def test_only_members_outside_the_ensemble_answering_is_still_nothing(self) -> None:
        with pytest.raises(AggregationError, match="q-1"):
            aggregating(answered("car", 0.9, "a", in_ensemble=False))
