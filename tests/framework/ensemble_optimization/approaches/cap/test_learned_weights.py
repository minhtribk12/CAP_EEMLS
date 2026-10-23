from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from eemls.ensemble_optimization.approaches.cap.learned_weights import (
    Draw,
    Weights,
    standard_normal,
)


def no_draw(size: int) -> NDArray[np.float64]:
    return np.zeros(size, dtype=np.float64)


def built(
    dimension: int = 1,
    *,
    draw: Draw = no_draw,
    precision: float = 1.0,
) -> Weights:
    return Weights(dimension=dimension, prior_precision=precision, draw=draw)


class TestBeforeAnyFeedback:
    def test_both_scores_are_zero_under_the_mean_weights(self) -> None:
        held = built()

        assert held.expected([1.0]) == pytest.approx(0.0)

    def test_the_difference_of_the_two_scores_is_zero(self) -> None:
        held = built().sampled().scores([1.0])

        assert held.difference == pytest.approx(0.0)


class TestOneUpdateByHand:
    def test_the_include_weight_is_the_posterior_mean(self) -> None:
        held = built()

        held.learn([1.0], 0.5, noise_variance=1.0)

        assert held.expected([1.0]) == pytest.approx(0.25)

    def test_the_exclude_weight_moves_the_other_way_by_the_same_amount(self) -> None:
        held = built()

        held.learn([1.0], 0.5, noise_variance=1.0)
        scores = held.sampled().scores([1.0])

        assert scores.include == pytest.approx(0.25)
        assert scores.exclude == pytest.approx(-0.25)
        assert scores.difference == pytest.approx(0.5)

    def test_a_second_update_follows_the_same_arithmetic(self) -> None:
        held = built()

        held.learn([1.0], 0.5, noise_variance=1.0)
        held.learn([1.0], 0.5, noise_variance=1.0)

        assert held.expected([1.0]) == pytest.approx(1.0 / 3.0)

    def test_a_larger_signal_carries_more_of_the_update(self) -> None:
        held = built()

        held.learn([2.0], 0.5, noise_variance=1.0)

        assert held.expected([1.0]) == pytest.approx(0.2)

    def test_a_noisier_reading_moves_the_weights_less(self) -> None:
        held = built()

        held.learn([1.0], 0.5, noise_variance=4.0)

        assert held.expected([1.0]) == pytest.approx(0.1)


class TestTheDifferenceTakesTheSignOfTheContribution:
    def test_a_service_that_helped_is_expected_to_help(self) -> None:
        held = built()

        for _ in range(20):
            held.learn([1.0], 0.4, noise_variance=1.0)

        assert held.sampled().scores([1.0]).difference > 0.0

    def test_a_service_that_hurt_is_expected_to_hurt(self) -> None:
        held = built()

        for _ in range(20):
            held.learn([1.0], -0.4, noise_variance=1.0)

        assert held.sampled().scores([1.0]).difference < 0.0

    def test_the_feedback_of_one_service_reaches_a_service_with_the_same_signals(self) -> None:
        held = built(2)

        held.learn([1.0, 0.0], 0.5, noise_variance=1.0)

        assert held.expected([1.0, 0.0]) > 0.0
        assert held.expected([0.0, 1.0]) == pytest.approx(0.0)


def _spread_of(held: Weights, draws: int = 40) -> float:
    return float(np.std([held.sampled().scores([1.0]).include for _ in range(draws)]))


class TestTheSamplingOfOneAction:
    def test_one_draw_scores_every_service_of_that_action(self) -> None:
        held = built(1, draw=standard_normal(7))

        sampled = held.sampled()

        assert sampled.scores([1.0]).include == sampled.scores([1.0]).include

    def test_two_actions_draw_differently(self) -> None:
        held = built(1, draw=standard_normal(7))

        first = held.sampled().scores([1.0]).include
        second = held.sampled().scores([1.0]).include

        assert first != second

    def test_a_seeded_draw_repeats_a_run_exactly(self) -> None:
        one = built(1, draw=standard_normal(11)).sampled().scores([1.0])
        two = built(1, draw=standard_normal(11)).sampled().scores([1.0])

        assert (one.include, one.exclude) == (two.include, two.exclude)

    def test_a_wide_distribution_varies_more_than_a_narrow_one(self) -> None:
        wide = built(1, draw=standard_normal(3))
        narrow = built(1, draw=standard_normal(3))
        for _ in range(50):
            narrow.learn([1.0], 0.3, noise_variance=1.0)

        assert _spread_of(wide) > _spread_of(narrow)

    def test_the_expected_score_does_not_move_between_actions(self) -> None:
        held = built(1, draw=standard_normal(5))
        held.learn([1.0], 0.5, noise_variance=1.0)

        assert held.expected([1.0]) == pytest.approx(held.expected([1.0]))
        assert held.expected([1.0]) == pytest.approx(0.25)
