from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from eemls.domain.types import Action, Row
from eemls.ensemble_optimization.approach import Observation, Policy, PolicyDecision
from eemls.ensemble_optimization.approach_loading import approaches
from eemls.ensemble_optimization.testing import (
    check_conformance,
    check_ensemble_size,
    check_same_seed_same_decisions,
    check_valid_actions,
)
from eemls.ensemble_optimization.testing.samples import answered, observed

ROOT = Path(__file__).resolve().parents[3]
CLASSES = ("car", "bus")


def rows(queries: int, *, unlabeled: int) -> tuple[Row, ...]:
    return tuple(
        answered(
            f"q-{index}",
            "car" if index % 3 else "bus",
            "car" if index < queries - unlabeled else None,
        )
        for index in range(queries)
    )


def observations(*, unlabeled: int = 4) -> tuple[Observation, ...]:
    return tuple(
        observed(serving=("infer-a", "infer-b"), given=rows(4 * (step + 1), unlabeled=unlabeled))
        for step in range(4)
    )


class Answers:
    name = "answers"

    def __init__(self, choose: Callable[[Observation], Action]) -> None:
        self.choose = choose

    def decide(self, observed: Observation) -> PolicyDecision:
        return PolicyDecision(action=self.choose(observed))


def keep(observed: Observation) -> Action:
    return Action(kind="keep")


def answers(choose: Callable[[Observation], Action]) -> Callable[[], Policy]:
    return lambda: Answers(choose)


class TestWhatTheKitRefuses:
    def test_an_action_naming_a_service_that_is_not_available(self) -> None:
        with pytest.raises(AssertionError, match="not available"):
            check_valid_actions(
                answers(lambda _: Action(kind="add", member_in="infer-z")), observations()
            )

    def test_an_ensemble_above_the_size_limit(self) -> None:
        with pytest.raises(AssertionError, match="above the limit of 2"):
            check_ensemble_size(
                answers(lambda _: Action(kind="add", member_in="infer-c")),
                observations(),
                max_ensemble_size=2,
            )

    def test_two_builds_on_one_seed_that_decide_differently(self) -> None:
        built = iter(
            [Answers(keep), Answers(lambda _: Action(kind="remove", member_out="infer-a"))]
        )

        with pytest.raises(AssertionError, match="same seed"):
            check_same_seed_same_decisions(lambda: next(built), observations())


def test_a_policy_that_reads_the_labels_of_the_current_queries_conforms() -> None:
    def on_the_newest_label(observed: Observation) -> Action:
        if observed.window[-1]["true_label"] is None:
            return Action(kind="keep")
        return Action(kind="remove", member_out="infer-a")

    check_conformance(answers(on_the_newest_label), observations(unlabeled=0), max_ensemble_size=2)


def test_a_policy_that_keeps_its_ensemble_conforms() -> None:
    check_conformance(answers(keep), observations(), max_ensemble_size=2)


@pytest.mark.parametrize("path", sorted((ROOT / "config" / "approaches").glob("*.yaml")))
def test_every_built_in_approach_conforms(path: Path) -> None:
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    approach = approaches()[body["kind"]]
    settings = approach.Settings.model_validate(body)

    def build() -> Policy:
        built = approach.build(settings, classes=CLASSES, seed=3)
        assert isinstance(built, Policy)
        return built

    check_conformance(build, observations(), max_ensemble_size=settings.max_ensemble_size)
