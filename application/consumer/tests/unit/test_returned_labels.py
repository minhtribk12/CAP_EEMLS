from __future__ import annotations

import random

from eemls_consumer.feedback import LabelsToReturn


def labels(*, fraction: float, label_delay_queries: int = 10, seed: int = 7) -> LabelsToReturn:
    return LabelsToReturn(
        fraction=fraction,
        label_delay_queries=label_delay_queries,
        draw=random.Random(seed).random,  # noqa: S311
    )


def recorded_every_query(held: LabelsToReturn, queries: int) -> LabelsToReturn:
    for number in range(queries):
        held.recorded(query_id=f"q-{number}", true_label="car", at_query=0, kept=held.kept())
    return held


class TestTheShare:
    def test_about_half_of_the_queries_are_returned_at_a_share_of_one_half(self) -> None:
        held = recorded_every_query(labels(fraction=0.5), 1000)

        assert 450 <= len(held.rest()) <= 550

    def test_a_share_of_zero_returns_nothing(self) -> None:
        held = recorded_every_query(labels(fraction=0.0), 100)

        assert held.rest() == ()

    def test_a_share_of_one_returns_everything(self) -> None:
        held = recorded_every_query(labels(fraction=1.0), 100)

        assert len(held.rest()) == 100


class TestTheDecision:
    def test_the_same_seed_gives_the_same_decisions(self) -> None:
        first, again = labels(fraction=0.5), labels(fraction=0.5)

        assert [first.kept() for _ in range(50)] == [again.kept() for _ in range(50)]

    def test_a_query_not_kept_is_not_recorded(self) -> None:
        held = labels(fraction=1.0)

        held.recorded(query_id="q-1", true_label="car", at_query=0, kept=False)

        assert held.rest() == ()


class TestTheDelay:
    """The delay counts queries: an interval is a number of the consumer's own queries."""

    def test_no_label_comes_back_before_the_delay_has_passed(self) -> None:
        held = labels(fraction=1.0)
        held.recorded(query_id="q-1", true_label="car", at_query=100, kept=True)

        assert held.due(after_queries=109) == ()
        assert len(held.rest()) == 1

    def test_the_label_comes_back_once_the_delay_has_passed(self) -> None:
        held = labels(fraction=1.0)
        held.recorded(query_id="q-1", true_label="car", at_query=100, kept=True)

        due = held.due(after_queries=110)

        assert [(one.query_id, one.true_label) for one in due] == [("q-1", "car")]
        assert held.rest() == ()

    def test_a_label_is_handed_back_once(self) -> None:
        held = labels(fraction=1.0)
        held.recorded(query_id="q-1", true_label="car", at_query=100, kept=True)
        held.due(after_queries=110)

        assert held.due(after_queries=120) == ()

    def test_only_the_labels_whose_delay_has_passed_come_back(self) -> None:
        held = labels(fraction=1.0)
        held.recorded(query_id="q-1", true_label="car", at_query=100, kept=True)
        held.recorded(query_id="q-2", true_label="bus", at_query=105, kept=True)

        due = held.due(after_queries=112)

        assert [one.query_id for one in due] == ["q-1"]
        assert [one.query_id for one in held.rest()] == ["q-2"]

    def test_the_rest_comes_back_when_the_run_is_over(self) -> None:
        held = labels(fraction=1.0)
        held.recorded(query_id="q-1", true_label="car", at_query=100, kept=True)
        held.recorded(query_id="q-2", true_label="bus", at_query=105, kept=True)

        rest = held.rest()

        assert [one.query_id for one in rest] == ["q-1", "q-2"]
        assert held.rest() == ()
