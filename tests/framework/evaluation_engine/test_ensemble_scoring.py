from __future__ import annotations

import pytest

from eemls.ensemble_optimization.testing.samples import (
    GENERAL_ACCURACY,
    NIGHT_ACCURACY,
    contract,
    declared,
    live_window,
    night_contract,
    row,
    scored,
    three_member_answers,
)
from eemls.evaluation_engine.aggregation_functions import CdmError
from eemls.evaluation_engine.ensemble_scoring import (
    MemberRow,
    aggregated_window,
    scored_window,
    voted_answer,
)
from eemls.evaluation_engine.window import Window


class TestACandidateCarriesTheContextOfEachQuery:
    def test_a_night_metric_of_a_candidate_matches_the_real_ensemble_on_the_same_rows(
        self,
    ) -> None:
        live = live_window(("A", "B", "C"))
        real, _ = scored_window(night_contract(), declared(NIGHT_ACCURACY), live)

        candidate, _ = scored_window(
            night_contract(),
            declared(NIGHT_ACCURACY),
            aggregated_window(three_member_answers(), ("A", "B", "C"), window=live),
        )

        assert real["night_accuracy"] == pytest.approx(0.5)
        assert candidate["night_accuracy"] == pytest.approx(real["night_accuracy"])


class TestTheWorkedExampleOfTheArticle:
    def test_the_ensemble_holding_c_answers_three_of_the_four(self) -> None:
        held = scored(contract(), GENERAL_ACCURACY, three_member_answers(), ("A", "B", "C"))

        assert held.total == pytest.approx(0.75)

    def test_the_ensemble_without_c_answers_all_four(self) -> None:
        held = scored(contract(), GENERAL_ACCURACY, three_member_answers(), ("A", "B"))

        assert held.total == pytest.approx(1.00)

    def test_the_query_it_turns_is_the_one_it_was_sure_about(self) -> None:
        whole = aggregated_window(three_member_answers(), ("A", "B", "C"))
        rest = aggregated_window(three_member_answers(), ("A", "B"))

        turned = {row["query_id"]: row["predicted_label"] for row in whole}
        kept = {row["query_id"]: row["predicted_label"] for row in rest}
        assert (turned["q-3"], kept["q-3"]) == ("motorcycle", "cyclist")

    def test_every_other_query_is_answered_the_same_either_way(self) -> None:
        whole = aggregated_window(three_member_answers(), ("A", "B", "C"))
        rest = aggregated_window(three_member_answers(), ("A", "B"))

        turned = {row["query_id"]: row["predicted_label"] for row in whole}
        kept = {row["query_id"]: row["predicted_label"] for row in rest}
        assert [turned[one] for one in ("q-1", "q-2", "q-4")] == ["car", "pedestrian", "bus"]
        assert [kept[one] for one in ("q-1", "q-2", "q-4")] == ["car", "pedestrian", "bus"]


class TestTheAggregatedResult:
    def test_one_member_gives_its_own_answer(self) -> None:
        cast = voted_answer((row(),))

        assert cast is not None
        assert cast[0] == "car"
        assert cast[1] == pytest.approx(0.9)

    def test_the_class_with_the_highest_mean_confidence_wins(self) -> None:
        answers = (
            row(service_name="infer-a", predicted_label="car", confidence=0.9),
            row(service_name="infer-b", predicted_label="truck", confidence=0.6),
            row(service_name="infer-c", predicted_label="truck", confidence=0.7),
        )

        cast = voted_answer(answers)

        assert cast is not None
        assert cast[0] == "car"
        assert cast[1] == pytest.approx(0.9)

    def test_two_members_agreeing_beat_a_more_confident_one_on_the_mean(self) -> None:
        answers = (
            row(service_name="infer-a", predicted_label="car", confidence=0.6),
            row(service_name="infer-b", predicted_label="truck", confidence=0.55),
            row(service_name="infer-c", predicted_label="truck", confidence=0.65),
        )

        cast = voted_answer(answers)

        assert cast is not None
        assert cast[0] == "truck"
        assert cast[1] == pytest.approx(0.6)

    def test_a_member_that_did_not_answer_is_left_out(self) -> None:
        answers = (
            row(service_name="infer-a", predicted_label="car", confidence=0.9),
            row(service_name="infer-b", predicted_label=None, confidence=None),
        )

        cast = voted_answer(answers)

        assert cast is not None
        assert cast[0] == "car"
        assert cast[1] == pytest.approx(0.9)

    def test_a_result_with_nobody_answering_has_no_label(self) -> None:
        assert voted_answer((row(predicted_label=None, confidence=None),)) is None

    def test_a_tie_is_broken_by_the_class_name_so_a_run_repeats(self) -> None:
        answers = (
            row(service_name="infer-a", predicted_label="truck", confidence=0.8),
            row(service_name="infer-b", predicted_label="car", confidence=0.8),
        )

        cast = voted_answer(answers)

        assert cast is not None
        assert cast[0] == "car"
        assert cast[1] == pytest.approx(0.8)


class TestTheJointWindow:
    def rows(self) -> tuple[MemberRow, ...]:
        return (
            row(
                query_id="q-1",
                service_name="infer-a",
                predicted_label="car",
                confidence=0.9,
                true_label="car",
            ),
            row(
                query_id="q-1",
                service_name="infer-b",
                predicted_label="truck",
                confidence=0.8,
                true_label="car",
            ),
            row(
                query_id="q-2",
                service_name="infer-a",
                predicted_label="bus",
                confidence=0.4,
                true_label="truck",
            ),
            row(
                query_id="q-2",
                service_name="infer-b",
                predicted_label="truck",
                confidence=0.7,
                true_label="truck",
            ),
        )

    def test_one_row_per_query_not_one_per_member(self) -> None:
        window = aggregated_window(self.rows(), ("infer-a", "infer-b"))

        assert len(window) == 2

    def test_each_row_carries_the_result_of_the_candidate(self) -> None:
        window = aggregated_window(self.rows(), ("infer-a", "infer-b"))

        assert [row["predicted_label"] for row in window] == ["car", "truck"]

    def test_a_candidate_of_one_member_returns_that_member(self) -> None:
        window = aggregated_window(self.rows(), ("infer-a",))

        assert [row["predicted_label"] for row in window] == ["car", "bus"]

    def test_the_slowest_member_sets_the_response_time(self) -> None:
        rows = (
            row(query_id="q-1", service_name="infer-a", total_ms=20.0),
            row(query_id="q-1", service_name="infer-b", total_ms=140.0),
        )

        window = aggregated_window(rows, ("infer-a", "infer-b"))

        assert window[0]["response_time_ms"] == pytest.approx(140.0)

    def test_the_cost_is_the_sum_over_the_members(self) -> None:
        rows = (
            row(query_id="q-1", service_name="infer-a", cost_per_1k=3.5),
            row(query_id="q-1", service_name="infer-b", cost_per_1k=7.0),
        )

        window = aggregated_window(rows, ("infer-a", "infer-b"))

        assert window[0]["cost_per_1k"] == pytest.approx(10.5)

    def test_the_throughput_is_the_slowest_member(self) -> None:
        rows = (
            row(query_id="q-1", service_name="infer-a", throughput_rps=24.0),
            row(query_id="q-1", service_name="infer-b", throughput_rps=9.0),
        )

        window = aggregated_window(rows, ("infer-a", "infer-b"))

        assert window[0]["throughput_rps"] == pytest.approx(9.0)

    def test_the_true_label_is_carried_through(self) -> None:
        window = aggregated_window(self.rows(), ("infer-a", "infer-b"))

        assert [row["true_label"] for row in window] == ["car", "truck"]

    def test_a_member_the_candidate_does_not_hold_is_left_out(self) -> None:
        window = aggregated_window(self.rows(), ("infer-b",))

        assert [row["predicted_label"] for row in window] == ["truck", "truck"]

    def test_a_query_the_candidate_did_not_answer_stays_in_the_window_unanswered(self) -> None:
        rows = (
            row(query_id="q-1", service_name="infer-a"),
            row(query_id="q-2", service_name="infer-b", true_label="bus", predicted_label="bus"),
        )

        window = aggregated_window(rows, ("infer-a",), ("q-1", "q-2"))

        assert [row["query_id"] for row in window] == ["q-1", "q-2"]
        assert [row["predicted_label"] for row in window] == ["car", None]
        assert [row["true_label"] for row in window] == ["car", "bus"]
        assert window[1]["n_answered"] == 0

    def test_only_the_queries_asked_for_are_in_the_window(self) -> None:
        window = aggregated_window(self.rows(), ("infer-a", "infer-b"), ("q-2",))

        assert [row["query_id"] for row in window] == ["q-2"]

    def test_a_candidate_naming_a_member_no_row_holds_is_refused(self) -> None:
        with pytest.raises(CdmError, match="infer-z"):
            aggregated_window(self.rows(), ("infer-a", "infer-z"))

    def test_an_empty_candidate_is_refused(self) -> None:
        with pytest.raises(CdmError, match="no member"):
            aggregated_window(self.rows(), ())


class TestScoringByTheVoteAndNotBySummingMembers:
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

    def test_the_pair_scores_above_the_mean_of_the_two_members(self) -> None:
        together = scored(contract(), GENERAL_ACCURACY, self.rows(), ("infer-car", "infer-truck"))
        alone = scored(contract(), GENERAL_ACCURACY, self.rows(), ("infer-car",))

        assert together.total == pytest.approx(1.0)
        assert alone.total == pytest.approx(0.5)

    def test_the_cost_of_a_candidate_counts_every_member(self) -> None:
        cheap = scored(contract(cost_weight=0.15), GENERAL_ACCURACY, self.rows(), ("infer-car",))
        both = scored(
            contract(cost_weight=0.15), GENERAL_ACCURACY, self.rows(), ("infer-car", "infer-truck")
        )

        assert both.cost == pytest.approx(cheap.cost * 2)


class TestWhatACandidateIsPricedAndTimedBy:
    def rows(self) -> tuple[MemberRow, ...]:
        return (
            row(
                query_id="q-1",
                service_name="infer-a",
                total_ms=20.0,
                cost_per_1k=3.5,
                throughput_rps=24.0,
            ),
            row(
                query_id="q-1",
                service_name="infer-b",
                total_ms=50.0,
                cost_per_1k=7.0,
                throughput_rps=9.0,
            ),
            row(
                query_id="q-2",
                service_name="infer-a",
                total_ms=20.0,
                cost_per_1k=3.5,
                throughput_rps=24.0,
            ),
            row(
                query_id="q-3",
                service_name="infer-a",
                total_ms=20.0,
                cost_per_1k=3.5,
                throughput_rps=24.0,
            ),
            row(
                query_id="q-3",
                service_name="infer-b",
                total_ms=50.0,
                cost_per_1k=7.0,
                throughput_rps=9.0,
                predicted_label=None,
                confidence=None,
            ),
        )

    def test_a_query_is_priced_by_every_member_of_the_candidate(self) -> None:
        window = aggregated_window(self.rows(), ("infer-a", "infer-b"))

        assert [row["cost_per_1k"] for row in window] == [pytest.approx(10.5)] * 3

    def test_the_throughput_is_the_slowest_member_of_the_candidate(self) -> None:
        window = aggregated_window(self.rows(), ("infer-a", "infer-b"))

        assert [row["throughput_rps"] for row in window] == [pytest.approx(9.0)] * 3

    def test_the_row_says_how_many_answered_and_how_many_serve(self) -> None:
        window = aggregated_window(self.rows(), ("infer-a", "infer-b"))

        assert [(row["n_members"], row["n_answered"]) for row in window] == [
            (2, 2),
            (2, 1),
            (2, 1),
        ]

    def test_a_candidate_whose_members_all_abstained_is_still_priced(self) -> None:
        silent: tuple[MemberRow, ...] = (
            row(query_id="q-1", service_name="infer-a", predicted_label=None, confidence=None),
        )

        result = scored(contract(), GENERAL_ACCURACY, silent, ("infer-a",))

        assert result.cost == pytest.approx(0.0)

        result = scored(contract(), GENERAL_ACCURACY, silent, ("infer-a",))

        assert result.cost == pytest.approx(0.0)


class TestWhatTheMembersMustAgreeOn:
    def test_the_true_label_is_taken_whichever_member_carries_it(self) -> None:
        first = aggregated_window(
            (
                row(query_id="q-1", service_name="infer-a", true_label=None),
                row(query_id="q-1", service_name="infer-b", true_label="car"),
            ),
            ("infer-a", "infer-b"),
        )
        reversed_order = aggregated_window(
            (
                row(query_id="q-1", service_name="infer-b", true_label="car"),
                row(query_id="q-1", service_name="infer-a", true_label=None),
            ),
            ("infer-a", "infer-b"),
        )

        assert first[0]["true_label"] == "car"
        assert reversed_order[0]["true_label"] == "car"

    def test_members_that_disagree_on_the_true_label_are_refused_by_query(self) -> None:
        rows = (
            row(query_id="q-1", service_name="infer-a", true_label="car"),
            row(query_id="q-1", service_name="infer-b", true_label="truck"),
        )

        with pytest.raises(CdmError, match="q-1"):
            aggregated_window(rows, ("infer-a", "infer-b"))


class TestACandidateNamesEachMemberOnce:
    def test_a_member_named_twice_is_refused(self) -> None:
        with pytest.raises(CdmError, match="infer-a"):
            aggregated_window((row(service_name="infer-a"),), ("infer-a", "infer-a"))


class TestOnlyWhatTheContractNeedsIsEvaluated:
    def test_a_metric_the_contract_does_not_score_is_not_evaluated(self) -> None:
        broken = dict(GENERAL_ACCURACY)
        broken["unrelated"] = {
            "measured": "per_window",
            "window": 500,
            "unit": "ratio",
            "description": "Reads a column that holds text",
            "specification": {"function": "average", "attribute": "query_id"},
        }

        result = scored(contract(), broken, (row(),), ("infer-a",))

        assert result.total == pytest.approx(1.0)


class TestWhatCannotBePriced:
    def test_a_cost_that_is_not_a_number_is_refused(self) -> None:
        from eemls.evaluation_engine.ensemble_scoring import _cost_per_1k

        with pytest.raises(CdmError, match="not a cost"):
            _cost_per_1k(Window(({"cost_per_1k": "cheap"},)))

    def test_a_candidate_that_answered_no_query_is_refused(self) -> None:
        from eemls.evaluation_engine.ensemble_scoring import _cost_per_1k

        with pytest.raises(CdmError, match="no query of the window was priced"):
            _cost_per_1k(Window(()))

    def test_a_window_of_queries_nobody_answered_cannot_be_priced(self) -> None:
        from eemls.evaluation_engine.ensemble_scoring import _cost_per_1k

        with pytest.raises(CdmError, match="no query of the window was priced"):
            _cost_per_1k(Window(({"cost_per_1k": None}, {"cost_per_1k": None})))

    def test_only_the_queries_that_were_priced_are_averaged(self) -> None:
        from eemls.evaluation_engine.ensemble_scoring import _cost_per_1k

        held = Window(({"cost_per_1k": None}, {"cost_per_1k": 4.0}, {"cost_per_1k": 6.0}))

        assert _cost_per_1k(held) == pytest.approx(5.0)
