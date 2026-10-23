from __future__ import annotations

import itertools
from pathlib import Path

import pytest
import yaml
from eemls_consumer.main import schedule_of
from eemls_consumer.query_schedule import Schedule
from pydantic import ValidationError

from make_scenarios import FOLDER, SCENARIOS, rendered
from support.repository import ROOT

RATE = 20.0


def loaded(name: str) -> Schedule:
    return schedule_of(FOLDER / f"{name}.yaml", request_rate_per_s=RATE)


def held_by(condition: str) -> set[str]:
    sources = yaml.safe_load((ROOT / "config/datasets/sources.yaml").read_text())
    return {
        label
        for source in sources["sources"].values()
        if condition in source["conditions"]
        for label in source["contributes"]
    }


def test_the_folder_is_where_the_file_layout_puts_it() -> None:
    assert ROOT / "experiments" / "scenarios" == FOLDER


def test_the_folder_holds_exactly_the_generated_scenarios() -> None:
    assert sorted(path.stem for path in FOLDER.glob("*.yaml")) == sorted(SCENARIOS)


def test_there_are_fifty_to_fifty_five_scenarios() -> None:
    assert 50 <= len(SCENARIOS) <= 55


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_file_is_the_one_its_seed_generates(name: str) -> None:
    assert (FOLDER / f"{name}.yaml").read_text(encoding="utf-8") == rendered(name)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_file_loads_as_a_schedule(name: str) -> None:
    assert loaded(name).periods


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_file_states_its_day_as_its_periods_per_day_times_the_period_length(
    name: str,
) -> None:
    scenario = SCENARIOS[name]

    assert loaded(name).day_length_s == scenario.period_s * len(scenario.day)


def test_the_short_period_scenario_has_a_day_of_seven_hundred_twenty_seconds() -> None:
    assert loaded("recurring_period_120s").day_length_s == 720.0


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_period_asks_only_for_classes_its_condition_holds(name: str) -> None:
    for period in loaded(name).periods:
        assert set(period.class_mix) <= held_by(period.condition), period.name


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_event_falls_inside_the_run(name: str) -> None:
    schedule, length_s = loaded(name), SCENARIOS[name].length_s

    assert all(period.starts_at_s < length_s for period in schedule.periods)
    assert all(event.ends_at_s <= length_s for event in schedule.events)
    assert all(surge.ends_at_s <= length_s for surge in schedule.surges)


def test_the_same_seed_gives_the_same_events() -> None:
    assert rendered("unpredictable") == rendered("unpredictable")


def test_another_seed_gives_other_events() -> None:
    assert loaded("unpredictable") != loaded("unpredictable_seed_1")


class TestTheScenariosOfThePaper:
    def test_the_recurring_day_has_six_periods_over_day_and_night(self) -> None:
        schedule = loaded("recurring")
        day = schedule.periods[:6]

        assert len({period.name for period in day}) == 6
        assert {period.condition for period in day} == {"day", "night"}
        assert [period.name for period in schedule.periods[6:12]] == [p.name for p in day]

    def test_the_recurring_day_holds_no_event(self) -> None:
        schedule = loaded("recurring")

        assert schedule.events == ()
        assert schedule.surges == ()
        assert all(period.event is None for period in schedule.periods)

    def test_the_unpredictable_run_holds_the_four_kinds_of_event(self) -> None:
        schedule = loaded("unpredictable")
        produced = {period.event for period in schedule.periods}

        assert {"period_change", "class_burst"} <= produced
        assert schedule.events
        assert schedule.surges

    def test_the_unpredictable_run_keeps_the_recurring_day_underneath(self) -> None:
        recurring = {period.name for period in loaded("recurring").periods}
        unpredictable = {
            period.name for period in loaded("unpredictable").periods if period.event is None
        }

        assert unpredictable == recurring


def test_an_event_the_paper_does_not_name_is_refused(tmp_path: Path) -> None:
    held = yaml.safe_load(rendered("recurring"))
    held["periods"][1]["event"] = "earthquake"
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(held), encoding="utf-8")

    with pytest.raises(ValidationError, match="event"):
        schedule_of(path, request_rate_per_s=RATE)


def top_class(mix: dict[str, float]) -> str:
    highest = max(mix.values())
    tops = [label for label, weight in mix.items() if weight == highest]
    assert len(tops) == 1, tops
    return tops[0]


def test_each_period_of_the_recurring_day_is_dominated_by_its_own_class() -> None:
    sources = yaml.safe_load((ROOT / "config/datasets/sources.yaml").read_text())
    day = loaded("recurring").periods[:6]

    tops = [top_class(dict(period.class_mix)) for period in day]

    assert sorted(tops) == sorted(sources["dominant_classes"])


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_no_period_follows_one_of_the_same_name(name: str) -> None:
    periods = loaded(name).periods

    assert all(one.name != two.name for one, two in itertools.pairwise(periods))


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_two_windows_of_one_kind_never_touch(name: str) -> None:
    schedule = loaded(name)

    events, surges = schedule.events, schedule.surges

    assert all(one.ends_at_s < two.starts_at_s for one, two in itertools.pairwise(events))
    assert all(one.ends_at_s < two.starts_at_s for one, two in itertools.pairwise(surges))


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_a_burst_sits_inside_one_period_that_resumes_after_it(name: str) -> None:
    periods = loaded(name).periods

    for before, burst, after in zip(periods, periods[1:], periods[2:], strict=False):
        if burst.event == "class_burst":
            assert before.event != "class_burst"
            assert after.name == before.name
            assert after.class_mix == before.class_mix


class TestTheQuickStartScenario:
    def test_it_lasts_one_day_of_six_periods_of_twenty_seconds(self) -> None:
        schedule = loaded("quick_start")

        assert schedule.day_length_s == 120.0
        assert len(schedule.periods) == 6

    def test_it_asks_only_for_classes_open_images_holds_by_day(self) -> None:
        sources = yaml.safe_load((ROOT / "config/datasets/sources.yaml").read_text())
        open_images = set(sources["sources"]["open_images"]["contributes"])

        for period in loaded("quick_start").periods:
            assert period.condition == "day"
            assert set(period.class_mix) <= open_images, period.name
