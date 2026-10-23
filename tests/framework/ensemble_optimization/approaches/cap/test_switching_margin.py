from __future__ import annotations

import math

import pytest

from eemls.ensemble_optimization.approaches.cap.switching_margin import SwitchingMargin


def built(*, initial: float = 0.25, before: int = 2) -> SwitchingMargin:
    return SwitchingMargin(initial=initial, contributions_before_margin=before)


class TestBeforeThereIsEnoughToMeasure:
    def test_it_is_the_initial_margin_with_no_contribution_at_all(self) -> None:
        assert built().value() == pytest.approx(0.25)

    def test_it_is_still_the_initial_margin_one_contribution_short(self) -> None:
        held = built(before=30)

        for _ in range(29):
            held.learn(0.5)

        assert held.value() == pytest.approx(0.25)
        assert held.contributions == 29

    def test_the_initial_margin_is_whatever_the_configuration_says(self) -> None:
        assert built(initial=0.4).value() == pytest.approx(0.4)


class TestOnceThereIsEnough:
    def test_it_becomes_the_spread_of_the_contributions(self) -> None:
        held = built(before=2)

        held.learn(0.1)
        held.learn(0.3)

        assert held.value() == pytest.approx(0.1)

    def test_it_is_the_population_spread_and_not_the_sample_one(self) -> None:
        held = built(before=2)

        held.learn(0.1)
        held.learn(0.3)

        sample_form = math.sqrt(((0.1 - 0.2) ** 2 + (0.3 - 0.2) ** 2) / 1)
        assert held.value() == pytest.approx(0.1)
        assert held.value() != pytest.approx(sample_form)

    def test_a_wider_spread_is_a_wider_margin(self) -> None:
        narrow, wide = built(), built()

        for one in (0.10, 0.12, 0.11):
            narrow.learn(one)
        for one in (-0.5, 0.0, 0.5):
            wide.learn(one)

        assert wide.value() > narrow.value()

    def test_contributions_that_never_vary_leave_no_margin_to_pass(self) -> None:
        held = built()

        held.learn(0.2)
        held.learn(0.2)

        assert held.value() == pytest.approx(0.0)

    def test_the_sign_of_a_contribution_does_not_shrink_the_margin(self) -> None:
        held = built()

        held.learn(-0.3)
        held.learn(0.3)

        assert held.value() == pytest.approx(0.3)

    def test_it_is_never_negative(self) -> None:
        held = built()

        for one in (0.4, -0.4, 0.0, 1.0, -1.0):
            held.learn(one)
            assert held.value() >= 0.0


class TestItMeasuresWithoutKeepingEveryReading:
    def test_it_matches_the_spread_of_the_whole_sequence(self) -> None:
        held = built()
        readings = [0.4, -0.2, 0.9, 0.1, -0.7, 0.33, 0.05]

        for one in readings:
            held.learn(one)

        mean = sum(readings) / len(readings)
        expected = math.sqrt(sum((one - mean) ** 2 for one in readings) / len(readings))
        assert held.value() == pytest.approx(expected)

    def test_it_counts_what_it_has_measured(self) -> None:
        held = built()

        for one in (0.1, 0.2, 0.3):
            held.learn(one)

        assert held.contributions == 3

    def test_a_long_run_stays_accurate(self) -> None:
        held = built()
        readings = [0.5 + index * 1e-6 for index in range(5000)]

        for one in readings:
            held.learn(one)

        mean = sum(readings) / len(readings)
        expected = math.sqrt(sum((one - mean) ** 2 for one in readings) / len(readings))
        assert held.value() == pytest.approx(expected, abs=1e-12)
