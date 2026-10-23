from __future__ import annotations

import logging

import pytest

from eemls.domain.consumer_defined_metrics import CdmSet
from eemls.ensemble_optimization.testing.samples import GENERAL_ACCURACY, answered, contract
from eemls.evaluation_engine.consumer_evaluation import evaluated_window, window_size_for
from eemls.evaluation_engine.window import Window

CDMS = CdmSet.model_validate({"definitions": GENERAL_ACCURACY})


def narrowed(window: int) -> CdmSet:
    held = {name: {**body, "window": window} for name, body in GENERAL_ACCURACY.items()}
    return CdmSet.model_validate({"definitions": held})


class TestHowFarBackItReads:
    def test_it_reads_as_far_back_as_the_longest_window_declared(self) -> None:
        assert window_size_for(CDMS, at_least=0) == 500
        assert window_size_for(narrowed(2), at_least=0) == 2

    def test_it_reads_at_least_as_far_back_as_the_approach_asks(self) -> None:
        assert window_size_for(narrowed(2), at_least=500) == 500

    def test_a_cdm_reaching_further_than_the_approach_sets_the_window_itself(self) -> None:
        assert window_size_for(CDMS, at_least=100) == 500

    def test_an_approach_that_reads_nothing_leaves_the_window_to_the_cdms(self) -> None:
        assert window_size_for(narrowed(2), at_least=0) == 2


class TestWhatAWindowScores:
    def test_an_empty_window_has_no_value_and_no_score(self) -> None:
        assert evaluated_window(contract(), CDMS, Window(())) == ({}, None)

    def test_a_window_nobody_was_charged_for_has_no_value_and_no_score(self) -> None:
        window = Window(({**answered("q-1", "car", "car"), "cost_per_1k": None},))

        assert evaluated_window(contract(), CDMS, window) == ({}, None)

    def test_a_labeled_window_gives_its_values_and_its_score(self) -> None:
        window = Window((answered("q-1", "car", "car"), answered("q-2", "bus", "car")))

        values, parts = evaluated_window(contract(), CDMS, window)

        assert values["general_accuracy"] == pytest.approx(0.5)
        assert parts is not None
        assert parts.total == pytest.approx(0.5)

    def test_a_window_with_no_label_yet_gives_no_score_and_says_so(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        window = Window((answered("q-1", "car", None),))

        with caplog.at_level(logging.INFO):
            _, parts = evaluated_window(contract(), CDMS, window)

        assert parts is None
        assert "process_tag=orchestrator error_type=window_not_scored" in caplog.text
