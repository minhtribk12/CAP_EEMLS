from __future__ import annotations

from eemls.ensemble_optimization.approaches.cap.action_selection import Ranking, action_toward

AVAILABLE = ("a", "b", "c", "d")


def ranked(**scores: float) -> Ranking:
    def of(name: str) -> float:
        return scores.get(name, 0.0)

    return of


def acted(
    target: tuple[str, ...],
    *,
    serving: tuple[str, ...],
    scores: dict[str, float] | None = None,
    margin: float = 0.1,
    limit: int = 3,
    available: tuple[str, ...] = AVAILABLE,
) -> tuple[str, str | None, str | None]:
    held = action_toward(
        target,
        serving=serving,
        available=available,
        ranked=ranked(**(scores or {})),
        margin=margin,
        limit=limit,
    )
    return held.kind, held.member_in, held.member_out


class TestCaseOneTheEnsembleHasRoom:
    def test_an_inference_service_the_target_holds_joins(self) -> None:
        assert acted(("a", "c"), serving=("a",)) == ("add", "c", None)

    def test_the_one_with_the_highest_expected_score_joins(self) -> None:
        held = acted(("a", "c", "d"), serving=("a",), scores={"c": 0.2, "d": 0.9})

        assert held == ("add", "d", None)

    def test_it_joins_while_there_is_room_for_it(self) -> None:
        assert acted(("a", "b", "c"), serving=("a", "b"), limit=3) == ("add", "c", None)


class TestCaseTwoTheEnsembleIsFull:
    def test_it_replaces_the_member_the_target_has_dropped(self) -> None:
        held = acted(("a", "b", "d"), serving=("a", "b", "c"), limit=3)

        assert held == ("replace", "d", "c")

    def test_the_weakest_of_the_dropped_members_is_the_one_replaced(self) -> None:
        held = acted(
            ("a", "d"), serving=("a", "b", "c"), limit=3, scores={"b": 0.5, "c": 0.1, "d": 0.9}
        )

        assert held == ("replace", "d", "c")

    def test_the_strongest_of_the_wanted_services_is_the_one_that_joins(self) -> None:
        held = acted(
            ("a", "b", "d"),
            serving=("a", "b", "c"),
            limit=3,
            scores={"c": 0.1, "d": 0.4},
        )

        assert held == ("replace", "d", "c")


class TestCaseThreeAMemberHasLeftTheTarget:
    def test_the_member_outside_the_target_is_removed(self) -> None:
        assert acted(("a",), serving=("a", "b"), limit=3) == ("remove", None, "b")

    def test_the_weakest_of_them_is_removed(self) -> None:
        held = acted(("a",), serving=("a", "b", "c"), scores={"b": 0.4, "c": 0.2})

        assert held == ("remove", None, "c")

    def test_the_only_member_serving_is_never_removed(self) -> None:
        assert acted((), serving=("a",), available=("a",)) == ("keep", None, None)

    def test_the_only_member_is_kept_although_the_target_has_dropped_it(self) -> None:
        held = acted((), serving=("a",), available=("a", "c"), scores={"a": 0.1, "c": 0.9})

        assert held == ("keep", None, None)


class TestCaseFourEveryMemberIsInTheTarget:
    def test_a_better_outside_service_replaces_the_weakest_member(self) -> None:
        held = acted(("a", "b"), serving=("a", "b"), limit=2, scores={"a": 0.9, "b": 0.2, "c": 0.5})

        assert held == ("replace", "c", "b")

    def test_it_has_to_be_better_by_more_than_the_margin(self) -> None:
        held = acted(
            ("a", "b"),
            serving=("a", "b"),
            limit=2,
            margin=0.4,
            scores={"a": 0.9, "b": 0.2, "c": 0.5},
        )

        assert held == ("keep", None, None)

    def test_a_margin_it_exactly_meets_is_not_more_than_the_margin(self) -> None:
        held = acted(
            ("a", "b"),
            serving=("a", "b"),
            limit=2,
            margin=0.3,
            scores={"a": 0.9, "b": 0.2, "c": 0.5},
        )

        assert held == ("keep", None, None)

    def test_a_full_ensemble_the_target_keeps_whole_is_still_compared(self) -> None:
        held = acted(
            ("a", "b", "c", "d"),
            serving=("a", "b", "c"),
            limit=3,
            scores={"a": 0.9, "b": 0.8, "c": 0.1, "d": 0.9},
        )

        assert held == ("replace", "d", "c")

    def test_nothing_outside_the_ensemble_leaves_nothing_to_compare(self) -> None:
        held = acted(("a", "b"), serving=("a", "b"), available=("a", "b"), limit=3)

        assert held == ("keep", None, None)


class TestCaseFiveNothingChanges:
    def test_a_target_that_is_the_ensemble_and_no_better_outsider_keeps_it(self) -> None:
        held = acted(("a", "b"), serving=("a", "b"), scores={"a": 0.9, "b": 0.8})

        assert held == ("keep", None, None)

    def test_an_empty_target_on_one_member_keeps_it(self) -> None:
        assert acted((), serving=("a",), available=("a",)) == ("keep", None, None)


class TestWhatMakesARunRepeat:
    def test_two_services_that_rank_alike_are_taken_by_name(self) -> None:
        held = acted(("a", "c", "d"), serving=("a",), scores={"c": 0.5, "d": 0.5})

        assert held == ("add", "c", None)

    def test_the_order_the_services_are_listed_in_does_not_decide(self) -> None:
        one = acted(("a", "c", "d"), serving=("a",), available=("a", "c", "d"))
        other = acted(("a", "c", "d"), serving=("a",), available=("a", "d", "c"))

        assert one == other

    def test_two_members_that_rank_alike_are_removed_by_name(self) -> None:
        held = acted(("a",), serving=("a", "b", "c"), scores={"b": 0.3, "c": 0.3})

        assert held == ("remove", None, "b")

    def test_the_order_the_members_are_listed_in_does_not_decide(self) -> None:
        alike = {"b": 0.3, "c": 0.3}
        one = acted((), serving=("b", "c"), available=("b", "c"), scores=alike)
        other = acted((), serving=("c", "b"), available=("c", "b"), scores=alike)

        assert one == other == ("remove", None, "b")
