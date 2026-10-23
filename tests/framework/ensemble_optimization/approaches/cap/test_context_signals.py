from __future__ import annotations

import pytest

from eemls.domain.types import Row
from eemls.ensemble_optimization.approaches.cap.context_signals import (
    SIGNAL_NAMES,
    ConsumerSignals,
    ServiceSignals,
    consumer_signals,
    mean_confidence,
    service_signals,
    strong_classes,
    time_of_day,
    vector,
    weighted_disagreement,
)
from eemls.evaluation_engine.window import Window, class_share
from eemls.testing.records import profile

CLASSES = ("car", "cyclist", "pedestrian")


def answered(query_id: str, service: str, label: str | None, confidence: float | None) -> Row:
    return {
        "query_id": query_id,
        "service_name": service,
        "predicted_label": label,
        "confidence": confidence,
        "in_ensemble": True,
        "true_label": None,
    }


def served(label: str | None, confidence: float | None = 0.8, true_label: str | None = None) -> Row:
    return {"predicted_label": label, "predicted_confidence": confidence, "true_label": true_label}


class TestTheClassesInDemand:
    def test_the_share_of_each_class_over_the_recent_answers(self) -> None:
        window = Window((served("car"), served("car"), served("cyclist"), served("pedestrian")))

        held = class_share(window, CLASSES, attribute="predicted_label")

        assert held == pytest.approx((0.5, 0.25, 0.25))

    def test_a_window_of_nothing_is_a_share_of_nothing(self) -> None:
        assert class_share(Window(()), CLASSES, attribute="predicted_label") == pytest.approx(
            (0.0, 0.0, 0.0)
        )

    def test_a_query_nobody_answered_is_not_a_class_in_demand(self) -> None:
        window = Window((served("car"), served(None)))

        assert class_share(window, CLASSES, attribute="predicted_label") == pytest.approx(
            (1.0, 0.0, 0.0)
        )

    def test_a_class_the_window_never_carried_has_a_share_of_zero(self) -> None:
        window = Window((served("car"),))

        held = class_share(window, CLASSES, attribute="predicted_label")

        assert held[CLASSES.index("pedestrian")] == pytest.approx(0.0)

    def test_an_answer_outside_the_declared_classes_is_not_counted(self) -> None:
        window = Window((served("car"), served("unicorn")))

        assert sum(class_share(window, CLASSES, attribute="predicted_label")) == pytest.approx(0.5)


class TestHowConfidentTheEnsembleIs:
    def test_the_mean_confidence_of_the_recent_answers(self) -> None:
        window = Window((served("car", 0.9), served("car", 0.5)))

        assert mean_confidence(window) == pytest.approx(0.7)

    def test_a_query_nobody_answered_lends_no_confidence(self) -> None:
        window = Window((served("car", 0.9), served(None, None)))

        assert mean_confidence(window) == pytest.approx(0.9)

    def test_a_window_of_nothing_is_no_confidence(self) -> None:
        assert mean_confidence(Window(())) is None


class TestWhereInTheDay:
    def test_it_is_the_place_in_the_day_the_newest_query_was_sent_at(self) -> None:
        window = Window(
            ({**served("car"), "time_of_day": 0.2}, {**served("car"), "time_of_day": 0.25})
        )

        assert time_of_day(window) == pytest.approx(0.25)

    def test_a_query_sent_without_a_place_in_the_day_is_passed_over(self) -> None:
        window = Window(
            ({**served("car"), "time_of_day": 0.4}, {**served("car"), "time_of_day": None})
        )

        assert time_of_day(window) == pytest.approx(0.4)

    def test_a_window_with_no_place_in_the_day_is_the_start_of_the_day(self) -> None:
        assert time_of_day(Window((served("car"),))) == pytest.approx(0.0)

    def test_the_consumer_signal_is_the_place_in_the_simulated_day(self) -> None:
        held = consumer_signals(
            window=Window(({**served("car"), "time_of_day": 0.5},)),
            classes=CLASSES,
            previous=None,
            n_members=1,
        )

        assert held.time_of_day == pytest.approx(0.5)


class TestTheStrongClassesOfOneService:
    def test_they_are_the_classes_it_answers_better_than_it_answers_overall(self) -> None:
        held = profile(
            general_accuracy=0.60,
            accuracy_by_class={"car": 0.9, "cyclist": 0.3, "pedestrian": 0.7},
        )

        assert strong_classes(held) == frozenset({"car", "pedestrian"})

    def test_a_service_with_no_measured_class_is_strong_on_nothing(self) -> None:
        assert strong_classes(profile(accuracy_by_class={})) == frozenset()

    def test_a_service_that_is_even_across_the_classes_is_strong_on_none_of_them(self) -> None:
        held = profile(general_accuracy=0.8, accuracy_by_class={"car": 0.8, "cyclist": 0.8})

        assert strong_classes(held) == frozenset()


class TestHowMuchOneServiceDisagrees:
    def test_a_member_against_the_rest_is_counted_by_its_confidence(self) -> None:
        window = Window(
            (
                answered("q-1", "a", "car", 0.9),
                answered("q-1", "b", "car", 0.8),
                answered("q-1", "c", "bus", 0.6),
            )
        )

        assert weighted_disagreement(window, "c") == pytest.approx(0.6)
        assert weighted_disagreement(window, "a") == pytest.approx(0.0)

    def test_it_is_a_rate_over_the_queries_that_service_answered(self) -> None:
        window = Window(
            (
                answered("q-1", "a", "car", 0.9),
                answered("q-1", "c", "bus", 0.6),
                answered("q-2", "a", "car", 0.9),
                answered("q-2", "c", "car", 0.5),
            )
        )

        assert weighted_disagreement(window, "c") == pytest.approx(0.3)

    def test_two_members_that_differ_each_disagree_with_all_the_others(self) -> None:
        window = Window(
            (
                answered("q-1", "a", "car", 0.9),
                answered("q-1", "b", "bus", 0.8),
            )
        )

        assert weighted_disagreement(window, "a") == pytest.approx(0.9)
        assert weighted_disagreement(window, "b") == pytest.approx(0.8)

    def test_a_tie_among_the_others_is_no_disagreement(self) -> None:
        window = Window(
            (
                answered("q-1", "a", "car", 0.9),
                answered("q-1", "b", "bus", 0.8),
                answered("q-1", "c", "truck", 0.7),
            )
        )

        assert weighted_disagreement(window, "a") == pytest.approx(0.9)
        window = Window(
            (
                answered("q-1", "a", "car", 0.9),
                answered("q-1", "b", "car", 0.8),
                answered("q-1", "c", "bus", 0.7),
            )
        )

        assert weighted_disagreement(window, "a") == pytest.approx(0.0)

    def test_the_count_does_not_depend_on_what_the_labels_are_called(self) -> None:
        one = Window((answered("q-1", "a", "car", 0.9), answered("q-1", "b", "zebra", 0.8)))
        two = Window((answered("q-1", "a", "zebra", 0.9), answered("q-1", "b", "car", 0.8)))

        assert weighted_disagreement(one, "a") == weighted_disagreement(two, "a")

    def test_a_service_that_served_nobody_recently_disagrees_with_nobody(self) -> None:
        window = Window((answered("q-1", "a", "car", 0.9),))

        assert weighted_disagreement(window, "nowhere") == pytest.approx(0.0)

    def test_a_query_of_one_member_alone_is_no_disagreement(self) -> None:
        window = Window((answered("q-1", "a", "car", 0.9),))

        assert weighted_disagreement(window, "a") == pytest.approx(0.0)

    def test_a_member_that_did_not_answer_lends_nothing(self) -> None:
        window = Window((answered("q-1", "a", "car", 0.9), answered("q-1", "c", None, None)))

        assert weighted_disagreement(window, "c") == pytest.approx(0.0)


class TestActingBeforeTheLabelsArrive:
    def window(self, labeled: bool) -> Window:
        return Window(
            (
                served("car", 0.9, true_label="car" if labeled else None),
                served("cyclist", 0.5, true_label="cyclist" if labeled else None),
            )
        )

    def test_no_consumer_signal_changes_when_the_labels_are_taken_away(self) -> None:
        with_labels = consumer_signals(
            window=self.window(True),
            classes=CLASSES,
            previous=None,
            n_members=2,
        )
        without = consumer_signals(
            window=self.window(False),
            classes=CLASSES,
            previous=None,
            n_members=2,
        )

        assert with_labels == without

    def test_only_the_arrived_label_signal_changes_for_a_service(self) -> None:
        def built(labeled: bool) -> ServiceSignals:
            return service_signals(
                strong=frozenset({"car"}),
                window=self.window(labeled),
                member_window=Window((answered("q-1", "a", "car", 0.9),)),
                service_name="a",
                previous_strong_share=None,
                period_contribution=0.14,
                condition_contribution=0.22,
            )

        with_labels, without = built(True), built(False)

        assert with_labels.strong_class_share_labeled == pytest.approx(0.5)
        assert without.strong_class_share_labeled == pytest.approx(0.0)
        assert with_labels.strong_class_share == without.strong_class_share
        assert with_labels.weighted_disagreement == without.weighted_disagreement
        assert with_labels.period_contribution == without.period_contribution


class TestTheVectorTheScoresAreComputedFrom:
    def consumer(self) -> ConsumerSignals:
        return consumer_signals(
            window=Window((served("car", 0.9), served("cyclist", 0.5))),
            classes=CLASSES,
            previous=None,
            n_members=2,
        )

    def test_the_consumer_signals_are_the_same_for_every_service(self) -> None:
        held = self.consumer()

        assert held.class_share == pytest.approx((0.5, 0.5, 0.0))
        assert held.mean_confidence == pytest.approx(0.7)
        assert held.n_members == pytest.approx(2.0)

    def test_the_change_since_the_previous_action_is_zero_on_the_first_action(self) -> None:
        held = self.consumer()

        assert held.class_share_change == pytest.approx((0.0, 0.0, 0.0))

    def test_the_change_is_against_the_share_of_the_previous_action(self) -> None:
        held = consumer_signals(
            window=Window((served("car"), served("car"))),
            classes=CLASSES,
            previous=(0.25, 0.75, 0.0),
            n_members=2,
        )

        assert held.class_share_change == pytest.approx((0.75, -0.75, 0.0))

    def test_one_service_adds_its_own_signals_and_its_two_looked_up_contributions(self) -> None:
        held = service_signals(
            strong=frozenset({"car"}),
            window=Window((served("car", 0.9), served("cyclist", 0.5))),
            member_window=Window((answered("q-1", "a", "car", 0.9),)),
            service_name="a",
            previous_strong_share=None,
            period_contribution=0.14,
            condition_contribution=0.22,
        )

        assert held.strong_class_share == pytest.approx(0.5)
        assert held.period_contribution == pytest.approx(0.14)
        assert held.condition_contribution == pytest.approx(0.22)

    def test_a_demand_that_rose_from_nothing_is_the_whole_rise(self) -> None:
        held = service_signals(
            strong=frozenset({"car"}),
            window=Window((served("car", 0.9), served("cyclist", 0.5))),
            member_window=Window((answered("q-1", "a", "car", 0.9),)),
            service_name="a",
            previous_strong_share=0.0,
            period_contribution=0.0,
            condition_contribution=0.0,
        )

        assert held.strong_class_share == pytest.approx(0.5)
        assert held.strong_class_share_change == pytest.approx(0.5)

    def test_the_first_action_of_a_service_shows_no_change(self) -> None:
        held = service_signals(
            strong=frozenset({"car"}),
            window=Window((served("car", 0.9), served("cyclist", 0.5))),
            member_window=Window((answered("q-1", "a", "car", 0.9),)),
            service_name="a",
            previous_strong_share=None,
            period_contribution=0.0,
            condition_contribution=0.0,
        )

        assert held.strong_class_share_change == pytest.approx(0.0)

    def test_a_demand_that_fell_to_nothing_is_the_whole_fall(self) -> None:
        held = service_signals(
            strong=frozenset({"car"}),
            window=Window((served("cyclist", 0.5),)),
            member_window=Window((answered("q-1", "a", "cyclist", 0.5),)),
            service_name="a",
            previous_strong_share=0.6,
            period_contribution=0.0,
            condition_contribution=0.0,
        )

        assert held.strong_class_share == pytest.approx(0.0)
        assert held.strong_class_share_change == pytest.approx(-0.6)

    def test_the_strong_classes_are_summed_in_one_order_whatever_the_set_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        names = ("bus", "car", "cyclist", "dog", "pedestrian", "scooter", "truck", "van")
        seen: list[tuple[str, ...]] = []

        def recorded(
            window: Window, classes: tuple[str, ...], *, attribute: str
        ) -> tuple[float, ...]:
            seen.append(tuple(classes))
            return tuple(0.0 for _ in classes)

        monkeypatch.setattr(
            "eemls.ensemble_optimization.approaches.cap.context_signals.class_share", recorded
        )
        for strong in (frozenset(names), frozenset(reversed(names))):
            service_signals(
                strong=strong,
                window=Window((served("car", 0.9),)),
                member_window=Window((answered("q-1", "a", "car", 0.9),)),
                service_name="a",
                previous_strong_share=None,
                period_contribution=0.0,
                condition_contribution=0.0,
            )

        assert seen == [names] * 4

    def test_the_vector_is_the_consumer_signals_then_the_service_signals(self) -> None:
        consumer = self.consumer()
        service = ServiceSignals(
            strong_class_share=0.5,
            strong_class_share_change=0.0,
            weighted_disagreement=0.0,
            strong_class_share_labeled=0.0,
            period_contribution=0.14,
            condition_contribution=0.22,
        )

        held = vector(consumer, service)

        assert len(held) == 2 * len(CLASSES) + 3 + 6
        assert held[-2:] == pytest.approx((0.14, 0.22))

    def test_every_place_of_the_vector_is_named(self) -> None:
        consumer = self.consumer()
        service = ServiceSignals(0.5, 0.0, 0.0, 0.0, 0.14, 0.22)

        assert len(vector(consumer, service)) == len(SIGNAL_NAMES(CLASSES))

    def test_two_services_differ_only_in_their_own_signals(self) -> None:
        consumer = self.consumer()
        one = vector(consumer, ServiceSignals(0.5, 0.0, 0.0, 0.0, 0.14, 0.22))
        two = vector(consumer, ServiceSignals(0.5, 0.0, 0.0, 0.0, 0.90, 0.22))

        assert one[: 2 * len(CLASSES) + 3] == two[: 2 * len(CLASSES) + 3]
        assert one != two
