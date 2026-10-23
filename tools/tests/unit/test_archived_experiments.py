from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pytest
from eemls_consumer.main import schedule_of

from eemls_tools.config_tree import load_config
from eemls_tools.runs import plan_of
from make_scenarios import ARCHIVED, rendered
from support.repository import ROOT

TREE = ROOT / "config"
ARCHIVE = ROOT / "experiments" / "archived"
SCENARIOS = sorted((ARCHIVE / "scenarios").glob("*.yaml"))
PLANS = sorted((ARCHIVE / "runs").glob("*.yaml"))
SEEDED = re.compile(r"(?P<kind>.+)_seed_(?P<seed>\d+)")
KINDS = 16
SEEDS = 10
RATE = 20.0


def _kinds() -> dict[str, set[int]]:
    seeds: dict[str, set[int]] = defaultdict(set)
    for path in SCENARIOS:
        matched = SEEDED.fullmatch(path.stem)
        assert matched, path.name
        seeds[matched["kind"]].add(int(matched["seed"]))
    return seeds


def test_every_kind_has_ten_seeds() -> None:
    seeds = _kinds()
    assert len(seeds) == KINDS
    assert all(held == set(range(SEEDS)) for held in seeds.values())


def test_the_seeds_of_a_kind_differ() -> None:
    for kind in _kinds():
        bodies = {
            (ARCHIVE / "scenarios" / f"{kind}_seed_{seed}.yaml").read_text(encoding="utf-8")
            for seed in range(SEEDS)
        }
        assert len(bodies) == SEEDS, kind


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda path: path.stem)
def test_an_archived_scenario_loads(path: Path) -> None:
    assert schedule_of(path, request_rate_per_s=RATE)


def test_there_is_a_plan_per_archived_scenario() -> None:
    assert {path.stem for path in PLANS} == set(ARCHIVED)


@pytest.mark.parametrize("path", PLANS, ids=lambda path: path.stem)
def test_an_archived_plan_loads_against_the_committed_tree(path: Path) -> None:
    plan = plan_of(path, config=load_config(TREE), scenarios=ARCHIVE / "scenarios")
    assert plan.approaches
    assert plan.scenario == path.stem


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda path: path.stem)
def test_an_archived_scenario_is_the_one_its_seed_generates(path: Path) -> None:
    assert path.read_text(encoding="utf-8") == rendered(path.stem)


def test_every_archived_scenario_lasts_ten_days() -> None:
    assert {one.days for one in ARCHIVED.values()} == {10}
    assert {path.stem for path in SCENARIOS} == set(ARCHIVED)


@pytest.mark.parametrize("path", PLANS, ids=lambda path: path.stem)
def test_an_archived_plan_runs_every_approach_ten_seeds_for_its_whole_scenario(
    path: Path,
) -> None:
    plan = plan_of(path, config=load_config(TREE), scenarios=ARCHIVE / "scenarios")
    article = plan_of(
        ROOT / "experiments" / "runs" / "recurring_ten_days.yaml",
        config=load_config(TREE),
        scenarios=ROOT / "experiments" / "scenarios",
    )

    assert set(plan.approaches) == set(article.approaches)
    assert plan.seeds == tuple(range(SEEDS))
    assert plan.duration_s == ARCHIVED[plan.scenario].length_s
