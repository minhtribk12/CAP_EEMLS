from __future__ import annotations

import pytest

from eemls.domain.records import InferenceServiceProfile
from eemls.domain.types import Row
from eemls.ensemble_optimization.testing.samples import THREE_MEMBER_ANSWERS
from eemls.evaluation_engine.profile_estimates import estimated_member_rows, member_rows
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


class TestTheRowsACandidateIsScoredFrom:
    def test_every_member_is_priced_by_its_own_profile(self) -> None:
        held = member_rows(three_member_window(), priced("A", "B", "C", cost_per_1k=4.0))

        assert {one.cost_per_1k for one in held} == {4.0}
        assert len(held) == 12

    def test_a_member_with_no_profile_cannot_be_priced_so_it_is_left_out(self) -> None:
        held = member_rows(three_member_window(), priced("A", "B"))

        assert {one.service_name for one in held} == {"A", "B"}


class TestWhatItWouldHaveAnswered:
    def test_it_answers_the_true_label_at_its_average_confidence_when_it_is_right(self) -> None:
        rows = member_rows(Window((served("q-1", "A", "car", 0.5, "car"),)), priced("A"))

        held = estimated_member_rows(
            profile(
                service_name="D",
                accuracy_by_class={"car": 1.0},
                confidence_by_class={"car": 0.8},
            ),
            labeled=rows,
        )

        assert [one.predicted_label for one in held] == ["car"]
        assert held[0].confidence == pytest.approx(0.8)

    def test_it_is_right_on_the_share_of_a_class_its_accuracy_on_that_class_gives(self) -> None:
        rows = member_rows(
            Window(tuple(served(f"q-{index}", "A", "car", 0.5, "car") for index in range(4))),
            priced("A"),
        )

        held = estimated_member_rows(
            profile(
                service_name="D",
                accuracy_by_class={"car": 0.75},
                confidence_by_class={"car": 0.8},
            ),
            labeled=rows,
        )

        assert [one.predicted_label == "car" for one in held].count(True) == 3
        assert {one.confidence for one in held} == {0.8}

    def test_the_same_window_gives_the_same_answers_every_time(self) -> None:
        rows = member_rows(
            Window(tuple(served(f"q-{index}", "A", "bus", 0.5, "car") for index in range(7))),
            priced("A"),
        )
        one = profile(
            service_name="D", accuracy_by_class={"car": 0.4}, confidence_by_class={"car": 0.8}
        )

        assert estimated_member_rows(one, labeled=rows) == estimated_member_rows(one, labeled=rows)

    def test_when_wrong_it_answers_the_strongest_other_label_the_members_voted(self) -> None:
        rows = member_rows(
            Window(
                (
                    served("q-1", "A", "bus", 0.3, "car"),
                    served("q-1", "B", "truck", 0.6, "car"),
                )
            ),
            priced("A", "B"),
        )

        held = estimated_member_rows(
            profile(
                service_name="D", accuracy_by_class={"car": 0.0}, confidence_by_class={"car": 0.8}
            ),
            labeled=rows,
        )

        assert [one.predicted_label for one in held] == ["truck"]
        assert held[0].confidence == pytest.approx(0.8)

    def test_a_member_answer_without_a_label_or_a_confidence_is_not_a_vote(self) -> None:
        rows = member_rows(
            Window(
                (
                    served("q-1", "A", "bus", 0.3, "car"),
                    served("q-1", "B", "truck", None, "car"),
                    served("q-1", "C", None, 0.9, "car"),
                )
            ),
            priced("A", "B", "C"),
        )

        held = estimated_member_rows(
            profile(
                service_name="D", accuracy_by_class={"car": 0.0}, confidence_by_class={"car": 0.8}
            ),
            labeled=rows,
        )

        assert [one.predicted_label for one in held] == ["bus"]

    def test_when_every_member_was_right_its_wrong_answer_is_another_label_of_the_window(
        self,
    ) -> None:
        rows = member_rows(
            Window(
                (
                    served("q-1", "A", "car", 0.5, "car"),
                    served("q-2", "A", "bus", 0.5, "bus"),
                )
            ),
            priced("A"),
        )

        held = estimated_member_rows(
            profile(
                service_name="D",
                accuracy_by_class={"car": 0.0, "bus": 1.0},
                confidence_by_class={"car": 0.8, "bus": 0.8},
            ),
            labeled=rows,
        )

        assert [(one.query_id, one.predicted_label) for one in held] == [
            ("q-1", "bus"),
            ("q-2", "bus"),
        ]

    def test_it_answers_nothing_about_a_class_its_profile_does_not_cover(self) -> None:
        rows = member_rows(
            Window(
                (
                    served("q-1", "A", "car", 0.5, "car"),
                    served("q-2", "A", "boat", 0.5, "boat"),
                )
            ),
            priced("A"),
        )

        held = estimated_member_rows(
            profile(
                service_name="D",
                accuracy_by_class={"car": 0.9},
                confidence_by_class={"car": 0.8},
            ),
            labeled=rows,
        )

        assert [one.query_id for one in held] == ["q-1"]

    def test_it_is_timed_and_priced_by_its_profile(self) -> None:
        rows = member_rows(Window((served("q-1", "A", "car", 0.5, "car"),)), priced("A"))

        held = estimated_member_rows(
            profile(service_name="D", mean_ms=55.0, cost_per_1k=7.0, throughput_rps=12.0),
            labeled=rows,
        )

        assert (held[0].total_ms, held[0].cost_per_1k, held[0].throughput_rps) == (55.0, 7.0, 12.0)
