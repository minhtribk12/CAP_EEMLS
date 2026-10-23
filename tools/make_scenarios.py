"""Write the experiment scenario files, each from its own seed.

Usage:
    python3 tools/make_scenarios.py           writes the active and the archived scenarios
    python3 tools/make_scenarios.py --check   fails when a committed file no longer matches
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

FOLDER = Path(__file__).resolve().parents[1] / "experiments" / "scenarios"
ARCHIVE_FOLDER = Path(__file__).resolve().parents[1] / "experiments" / "archived" / "scenarios"

DAY_TAIL = (
    "traffic_light",
    "traffic_sign",
    "traffic_cone",
    "road_bump",
    "road_curve",
    "car_stop_sign",
    "guardrail",
    "train",
    "boat",
    "airplane",
    "dog",
    "cat",
    "bird",
)
OPEN_IMAGES_DAY_TAIL = (
    "traffic_light",
    "traffic_sign",
    "train",
    "boat",
    "airplane",
    "dog",
    "cat",
    "bird",
)
NIGHT_TAIL = ("traffic_cone", "road_bump", "road_curve", "car_stop_sign", "guardrail")
TAIL_WEIGHT = 0.1
BURSTING = {
    "day": ("pedestrian", "cyclist", "motorcycle", "car", "truck", "bus"),
    "night": ("pedestrian", "cyclist", "car"),
}
BURST_WEIGHT = 7.0
HOST_WEIGHT = 3.0
SHIFT_SHARE = 0.4
MARGIN_SHARE = 0.1
ATTEMPTS = 1000


@dataclass(frozen=True)
class Shape:
    condition: str
    mix: Mapping[str, float]


SHAPES = {
    "morning_buses": Shape("day", {"bus": 5.0, "car": 2.0, "truck": 2.0, "pedestrian": 1.0}),
    "deliveries": Shape("day", {"truck": 5.0, "car": 2.0, "bus": 2.0, "motorcycle": 1.0}),
    "midday_motorcycles": Shape(
        "day", {"motorcycle": 5.0, "cyclist": 2.0, "pedestrian": 2.0, "car": 1.0}
    ),
    "afternoon_bikes": Shape(
        "day", {"cyclist": 5.0, "pedestrian": 2.0, "motorcycle": 2.0, "car": 1.0}
    ),
    "morning_commute": Shape("day", {"car": 5.0, "truck": 2.0, "bus": 2.0, "pedestrian": 1.0}),
    "school_run": Shape("day", {"pedestrian": 5.0, "cyclist": 3.0, "car": 2.0}),
    "night_traffic": Shape("night", {"car": 6.0, "pedestrian": 2.0, "cyclist": 2.0}),
    "night_walkers": Shape("night", {"pedestrian": 6.0, "cyclist": 2.0, "car": 2.0}),
    "night_cyclists": Shape("night", {"cyclist": 6.0, "pedestrian": 2.0, "car": 2.0}),
}

PAPER_DAY = (
    "morning_buses",
    "deliveries",
    "midday_motorcycles",
    "afternoon_bikes",
    "night_traffic",
    "night_walkers",
)
DAY_ONLY = (
    "morning_commute",
    "morning_buses",
    "deliveries",
    "school_run",
    "midday_motorcycles",
    "afternoon_bikes",
)
NIGHT_HEAVY = (
    "morning_buses",
    "deliveries",
    "midday_motorcycles",
    "night_traffic",
    "night_walkers",
    "night_cyclists",
)


@dataclass(frozen=True)
class Events:
    period_changes: int = 0
    class_bursts: int = 0
    quality_drops: int = 0
    query_surges: int = 0
    burst_s: float = 60.0
    drop_s: float = 120.0
    surge_s: float = 60.0
    surge_multiplier: float = 3.0


PAPER_EVENTS = Events(period_changes=1, class_bursts=2, quality_drops=1, query_surges=1)


@dataclass(frozen=True)
class Scenario:
    description: str
    day: tuple[str, ...] = PAPER_DAY
    period_s: float = 300.0
    days: int = 4
    seed: int = 0
    events: Events = field(default_factory=Events)
    shuffled: bool = False
    sharpness: float = 1.0
    day_tail: tuple[str, ...] = DAY_TAIL

    @property
    def length_s(self) -> float:
        return self.period_s * len(self.day) * self.days


def _catalogue() -> dict[str, Scenario]:
    held: dict[str, Scenario] = {
        "recurring": Scenario("Recurring day, no events."),
        "unpredictable": Scenario(
            "Recurring day with unpredictable events.",
            events=PAPER_EVENTS,
        ),
        "unpredictable_ten_days": Scenario(
            "Recurring day with unpredictable events, ten days.",
            events=PAPER_EVENTS,
            days=10,
        ),
        "recurring_period_120s": Scenario("Recurring day, short periods.", period_s=120.0),
        "recurring_period_600s": Scenario("Recurring day, long periods.", period_s=600.0),
        "recurring_period_1200s": Scenario(
            "Recurring day, very long periods.", period_s=1200.0, days=2
        ),
        "recurring_one_day": Scenario("Recurring day, one day.", days=1),
        "recurring_eight_days": Scenario("Recurring day, eight days.", days=8),
        "recurring_ten_days": Scenario("Recurring day, ten days.", days=10),
        "recurring_day_only": Scenario("Recurring day, daylight only.", day=DAY_ONLY),
        "recurring_night_heavy": Scenario("Recurring day, three night periods.", day=NIGHT_HEAVY),
        "recurring_mild_mix": Scenario("Recurring day, milder class mix.", sharpness=0.5),
        "recurring_sharp_mix": Scenario("Recurring day, sharper class mix.", sharpness=2.0),
        "smoke": Scenario(
            "Short smoke run with every event.",
            period_s=60.0,
            days=1,
            events=Events(1, 1, 1, 1, burst_s=10.0, drop_s=20.0, surge_s=10.0),
        ),
        "quick_start": Scenario(
            "Quick start: one short daylight day of Open Images classes.",
            day=DAY_ONLY,
            period_s=20.0,
            days=1,
            day_tail=OPEN_IMAGES_DAY_TAIL,
        ),
    }
    for seed in range(1, 10):
        held[f"unpredictable_seed_{seed}"] = Scenario(
            "Unpredictable events, another seed.",
            seed=seed,
            events=PAPER_EVENTS,
        )
    for seed in range(5):
        held[f"shuffled_seed_{seed}"] = Scenario(
            "Periods shuffled every day.",
            seed=seed,
            shuffled=True,
        )
    for seed in range(2):
        held[f"only_period_change_seed_{seed}"] = Scenario(
            "Period changes only.",
            seed=seed,
            events=Events(period_changes=2),
        )
        held[f"only_class_burst_seed_{seed}"] = Scenario(
            "Class bursts only.",
            seed=seed,
            events=Events(class_bursts=3),
        )
        held[f"only_quality_drop_seed_{seed}"] = Scenario(
            "Quality drops only.",
            seed=seed,
            events=Events(quality_drops=2),
        )
        held[f"only_query_surge_seed_{seed}"] = Scenario(
            "Query surges only.", seed=seed, events=Events(query_surges=2)
        )
        held[f"long_quality_drop_seed_{seed}"] = Scenario(
            "One long quality drop a day.",
            seed=seed,
            events=Events(quality_drops=1, drop_s=600.0),
        )
        held[f"large_query_surge_seed_{seed}"] = Scenario(
            "One large query surge a day.",
            seed=seed,
            events=Events(query_surges=1, surge_s=120.0, surge_multiplier=5.0),
        )
        held[f"frequent_class_bursts_seed_{seed}"] = Scenario(
            "Six class bursts a day.", seed=seed, events=Events(class_bursts=6)
        )
        held[f"unpredictable_day_only_seed_{seed}"] = Scenario(
            "Unpredictable events, daylight only.",
            day=DAY_ONLY,
            seed=seed,
            events=PAPER_EVENTS,
        )
        held[f"unpredictable_night_heavy_seed_{seed}"] = Scenario(
            "Unpredictable events, three night periods.",
            day=NIGHT_HEAVY,
            seed=seed,
            events=PAPER_EVENTS,
        )
        held[f"shuffled_unpredictable_seed_{seed}"] = Scenario(
            "Unpredictable events, shuffled periods.",
            seed=seed,
            events=PAPER_EVENTS,
            shuffled=True,
        )
    for seed in range(3):
        held[f"unpredictable_mild_seed_{seed}"] = Scenario(
            "Milder unpredictable events.",
            seed=seed,
            events=Events(
                1, 1, 1, 1, burst_s=30.0, drop_s=60.0, surge_s=30.0, surge_multiplier=2.0
            ),
        )
        held[f"unpredictable_severe_seed_{seed}"] = Scenario(
            "Stronger unpredictable events.",
            seed=seed,
            events=Events(
                2, 4, 2, 2, burst_s=120.0, drop_s=300.0, surge_s=120.0, surge_multiplier=4.0
            ),
        )
    return held


SCENARIOS = _catalogue()

ARCHIVED_KINDS: dict[str, tuple[str, dict[str, Any]]] = {
    "big_query_surges": (
        "Four query surges a day at four times the load.",
        {"events": Events(query_surges=4, surge_multiplier=4.0)},
    ),
    "calm_shuffled": ("Periods shuffled every day, no events.", {"shuffled": True}),
    "day_only_shuffled": (
        "Daylight only, shuffled, with unpredictable events.",
        {"day": DAY_ONLY, "shuffled": True, "events": PAPER_EVENTS},
    ),
    "everything_severe": (
        "Every event, stronger and more often, shuffled periods.",
        {
            "shuffled": True,
            "events": Events(
                3, 6, 3, 3, burst_s=120.0, drop_s=300.0, surge_s=120.0, surge_multiplier=5.0
            ),
        },
    ),
    "fifteen_minute_periods": (
        "Fifteen-minute periods with unpredictable events.",
        {"period_s": 900.0, "events": PAPER_EVENTS},
    ),
    "half_hour_quality_drop": (
        "One thirty-minute quality drop a day, ten-minute periods.",
        {"period_s": 600.0, "events": Events(quality_drops=1, drop_s=1800.0)},
    ),
    "many_short_bursts": (
        "Eight short class bursts a day.",
        {"events": Events(class_bursts=8, burst_s=30.0)},
    ),
    "night_heavy_shuffled": (
        "Three night periods, shuffled, with unpredictable events.",
        {"day": NIGHT_HEAVY, "shuffled": True, "events": PAPER_EVENTS},
    ),
    "one_minute_periods": (
        "One-minute periods with short unpredictable events.",
        {
            "period_s": 60.0,
            "events": Events(1, 2, 1, 1, burst_s=10.0, drop_s=20.0, surge_s=10.0),
        },
    ),
    "period_churn": ("Five unplanned period changes a day.", {"events": Events(period_changes=5)}),
    "repeated_quality_drops": (
        "Three five-minute quality drops a day.",
        {"events": Events(quality_drops=3, drop_s=300.0)},
    ),
    "short_periods": (
        "Two-minute periods, twice the unpredictable events.",
        {
            "period_s": 120.0,
            "events": Events(2, 2, 2, 2, burst_s=30.0, drop_s=60.0, surge_s=30.0),
        },
    ),
    "shuffled_days": (
        "Shuffled days with unpredictable events.",
        {"shuffled": True, "events": PAPER_EVENTS},
    ),
    "shuffled_mild_events": (
        "Shuffled periods with mild events.",
        {
            "shuffled": True,
            "events": Events(
                1, 1, 1, 1, burst_s=30.0, drop_s=60.0, surge_s=30.0, surge_multiplier=2.0
            ),
        },
    ),
    "very_flat_mix": (
        "Very flat class mix with unpredictable events.",
        {"sharpness": 0.25, "events": PAPER_EVENTS},
    ),
    "very_sharp_mix": (
        "Very sharp class mix with unpredictable events.",
        {"sharpness": 2.5, "events": PAPER_EVENTS},
    ),
}
ARCHIVED: dict[str, Scenario] = {
    f"{kind}_seed_{seed}": Scenario(description, seed=seed, days=10, **settings)
    for kind, (description, settings) in ARCHIVED_KINDS.items()
    for seed in range(10)
}


def _scenario(name: str) -> Scenario:
    return SCENARIOS[name] if name in SCENARIOS else ARCHIVED[name]


def _mix(shape: Shape, scenario: Scenario) -> dict[str, float]:
    tail = scenario.day_tail if shape.condition == "day" else NIGHT_TAIL
    dominant = {label: round(weight**scenario.sharpness, 3) for label, weight in shape.mix.items()}
    return dominant | dict.fromkeys(tail, TAIL_WEIGHT)


@dataclass(frozen=True)
class Row:
    name: str
    starts_at_s: float
    class_mix: Mapping[str, float]
    condition: str
    event: str | None = None

    def as_written(self) -> dict[str, object]:
        written: dict[str, object] = {
            "name": self.name,
            "starts_at_s": self.starts_at_s,
            "class_mix": dict(self.class_mix),
            "condition": self.condition,
            "weather": "clear",
        }
        if self.event is not None:
            written["event"] = self.event
        return written


def _day_starts(scenario: Scenario, rng: random.Random) -> list[Row]:
    rows: list[Row] = []
    for day in range(scenario.days):
        order = scenario.day
        if scenario.shuffled:
            order = tuple(rng.sample(scenario.day, len(scenario.day)))
            while rows and order[0] == rows[-1].name:
                order = tuple(rng.sample(scenario.day, len(scenario.day)))
        for index, name in enumerate(order):
            shape = SHAPES[name]
            rows.append(
                Row(
                    name=name,
                    starts_at_s=(day * len(order) + index) * scenario.period_s,
                    class_mix=_mix(shape, scenario),
                    condition=shape.condition,
                )
            )
    return rows


def _changed(rows: list[Row], scenario: Scenario, rng: random.Random) -> list[Row]:
    per_day = len(scenario.day)
    moved = list(rows)
    for day in range(scenario.days):
        for index in rng.sample(range(1, per_day), scenario.events.period_changes):
            shift = rng.uniform(-SHIFT_SHARE, SHIFT_SHARE) * scenario.period_s
            row = moved[day * per_day + index]
            moved[day * per_day + index] = replace(
                row, starts_at_s=float(round(row.starts_at_s + shift)), event="period_change"
            )
    return moved


def _bursts(rows: list[Row], scenario: Scenario, rng: random.Random) -> list[Row]:
    held = list(rows)
    day_s = scenario.period_s * len(scenario.day)
    for day in range(scenario.days):
        for _ in range(scenario.events.class_bursts):
            held = _one_burst(held, day * day_s, (day + 1) * day_s, scenario.events.burst_s, rng)
    return held


def _one_burst(
    rows: list[Row], day_start: float, day_end: float, length: float, rng: random.Random
) -> list[Row]:
    for _ in range(ATTEMPTS):
        start = float(round(rng.uniform(day_start, day_end - length)))
        index = max(i for i, row in enumerate(rows) if row.starts_at_s <= start)
        host = rows[index]
        host_end = rows[index + 1].starts_at_s if index + 1 < len(rows) else day_end
        margin = MARGIN_SHARE * (host_end - host.starts_at_s)
        if host.event == "class_burst":
            continue
        if start < host.starts_at_s + margin or start + length > host_end - margin:
            continue
        dominant = SHAPES[host.name].mix
        top = max(dominant, key=lambda label: dominant[label])
        label = rng.choice([one for one in BURSTING[host.condition] if one != top])
        total = sum(host.class_mix.values())
        class_mix = {
            one: round(HOST_WEIGHT * weight / total, 3) for one, weight in host.class_mix.items()
        }
        class_mix[label] = round(class_mix.get(label, 0.0) + BURST_WEIGHT, 3)
        burst = Row(f"burst_{label}", start, class_mix, host.condition, "class_burst")
        resumed = replace(host, starts_at_s=start + length, event=None)
        return [*rows[: index + 1], burst, resumed, *rows[index + 1 :]]
    raise ValueError(f"no room for a class burst of {length} s between {day_start} and {day_end}")


def _windows(
    count: int, length: float, scenario: Scenario, rng: random.Random
) -> list[tuple[float, float]]:
    day_s = scenario.period_s * len(scenario.day)
    held: list[tuple[float, float]] = []
    for day in range(scenario.days):
        for _ in range(count):
            for _ in range(ATTEMPTS):
                start = float(round(rng.uniform(day * day_s, (day + 1) * day_s - length)))
                if all(start + length < one or start > two for one, two in held):
                    held.append((start, start + length))
                    break
            else:
                raise ValueError(f"no room for a window of {length} s on day {day}")
    return sorted(held)


def scenario_body(name: str) -> dict[str, object]:
    scenario = _scenario(name)
    rng = random.Random(scenario.seed)  # noqa: S311
    rows = _bursts(_changed(_day_starts(scenario, rng), scenario, rng), scenario, rng)
    body: dict[str, object] = {
        "day_length_s": scenario.period_s * len(scenario.day),
        "periods": [row.as_written() for row in rows],
    }
    events = scenario.events
    drops = _windows(events.quality_drops, events.drop_s, scenario, rng)
    if drops:
        body["events"] = [
            {"starts_at_s": start, "ends_at_s": end, "image_quality": "blurred"}
            for start, end in drops
        ]
    surges = _windows(events.query_surges, events.surge_s, scenario, rng)
    if surges:
        body["surges"] = [
            {"starts_at_s": start, "ends_at_s": end, "multiplier": events.surge_multiplier}
            for start, end in surges
        ]
    return body


def rendered(name: str) -> str:
    return yaml.safe_dump(scenario_body(name), sort_keys=False, width=100)


def write(folder: Path, names: Mapping[str, Scenario]) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / f"{name}.yaml").write_text(rendered(name), encoding="utf-8")
    print(f"{len(names)} scenarios written to {folder}")
    return 0


def check(folder: Path, names: Mapping[str, Scenario]) -> int:
    stale = [
        name
        for name in names
        if not (folder / f"{name}.yaml").is_file()
        or (folder / f"{name}.yaml").read_text(encoding="utf-8") != rendered(name)
    ]
    extra = sorted({path.stem for path in folder.glob("*.yaml")} - set(names))
    if stale or extra:
        print(
            f"stale: {stale}; not generated: {extra}; run tools/make_scenarios.py",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the scenario files.")
    parser.add_argument(
        "--check", action="store_true", help="verify the committed files instead of writing them"
    )
    args = parser.parse_args(argv)
    run = check if args.check else write
    return max(run(FOLDER, SCENARIOS), run(ARCHIVE_FOLDER, ARCHIVED))


if __name__ == "__main__":
    sys.exit(main())
