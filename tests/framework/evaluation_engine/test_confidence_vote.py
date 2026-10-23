from __future__ import annotations

import itertools

import pytest

from eemls.evaluation_engine.confidence_vote import confidence_vote


class TestWhatASetOfAnswersComesTo:
    def test_the_label_with_the_highest_mean_wins(self) -> None:
        vote = confidence_vote([("car", 0.9), ("car", 0.7), ("bus", 0.85)])

        assert vote is not None
        assert (vote.label, vote.voters) == ("bus", 3)
        assert vote.confidence == pytest.approx(0.85)

    def test_the_confidence_is_the_mean_over_the_members_that_named_the_winner(self) -> None:
        vote = confidence_vote([("car", 0.9), ("car", 0.7), ("bus", 0.1)])

        assert vote is not None
        assert (vote.label, vote.voters) == ("car", 3)
        assert vote.confidence == pytest.approx(0.8)

    def test_a_tie_is_broken_by_the_label_so_a_run_repeats(self) -> None:
        vote = confidence_vote([("car", 0.6), ("bus", 0.6)])

        assert vote is not None and vote.label == "bus"

    def test_no_answers_come_to_nothing(self) -> None:
        assert confidence_vote([]) is None

    @pytest.mark.parametrize("order", list(itertools.permutations([0.1, 0.2, 0.3])))
    def test_the_order_the_members_answered_in_never_moves_the_mean(
        self, order: tuple[float, ...]
    ) -> None:
        vote = confidence_vote([("car", one) for one in order])

        assert vote is not None
        same = confidence_vote([("car", 0.1), ("car", 0.2), ("car", 0.3)])
        assert same is not None
        assert vote.confidence == same.confidence


class TestAnAnswerWithoutALabelOrAConfidence:
    def test_it_is_not_a_vote(self) -> None:
        vote = confidence_vote([("car", 0.6), (None, 0.9), ("bus", None)])

        assert vote is not None
        assert (vote.label, vote.voters) == ("car", 1)
