from __future__ import annotations

from eemls.domain.types import Row
from eemls.ensemble_optimization.approaches.cap.night_detection import NightDetector
from eemls.evaluation_engine.window import Window

DROP = 0.1


def predicted(confidence: float | None, tag: str | None = None) -> Row:
    held = {"predicted_label": "car", "predicted_confidence": confidence}
    return held if tag is None else held | {"condition": tag}


def confident(*levels: float | None, tag: str | None = None) -> Window:
    return Window(tuple(predicted(one, tag) for one in levels))


class TestTheConditionComesFromTheConfidenceOfThePredictions:
    def test_the_first_window_is_read_as_day(self) -> None:
        assert NightDetector(night_confidence_drop=DROP).read(confident(0.9)) == "day"

    def test_a_drop_below_the_day_level_by_more_than_the_setting_is_night(self) -> None:
        held = NightDetector(night_confidence_drop=DROP)
        held.read(confident(0.9))

        assert held.read(confident(0.75)) == "night"

    def test_a_drop_within_the_setting_stays_day(self) -> None:
        held = NightDetector(night_confidence_drop=DROP)
        held.read(confident(0.9))

        assert held.read(confident(0.85)) == "day"

    def test_the_day_level_is_the_mean_confidence_of_the_day_windows(self) -> None:
        held = NightDetector(night_confidence_drop=DROP)
        held.read(confident(0.9))
        held.read(confident(0.82))

        assert held.read(confident(0.75)) == "night"

    def test_night_windows_do_not_lower_the_day_level(self) -> None:
        held = NightDetector(night_confidence_drop=DROP)
        held.read(confident(0.9))
        for _ in range(5):
            held.read(confident(0.6))

        assert held.read(confident(0.75)) == "night"

    def test_the_confidence_coming_back_to_the_day_level_is_day_again(self) -> None:
        held = NightDetector(night_confidence_drop=DROP)
        held.read(confident(0.9))
        held.read(confident(0.6))

        assert held.read(confident(0.88)) == "day"

    def test_a_window_with_no_confidence_keeps_the_last_condition(self) -> None:
        held = NightDetector(night_confidence_drop=DROP)
        held.read(confident(0.9))
        held.read(confident(0.6))

        assert held.read(confident(None)) == "night"
        assert held.read(Window(())) == "night"

    def test_the_condition_tag_of_an_image_is_not_read(self) -> None:
        held = NightDetector(night_confidence_drop=DROP)

        assert held.read(confident(0.9, tag="night")) == "day"
        assert held.read(confident(0.6, tag="day")) == "night"
