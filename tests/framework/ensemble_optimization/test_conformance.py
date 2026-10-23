from __future__ import annotations

import pytest

from eemls.config.reading import Reader
from eemls.domain.types import Row
from eemls.ensemble_optimization.approach import ApproachSettings, Observation, Policy
from eemls.ensemble_optimization.approach_loading import approaches, load_approaches, policy_of
from eemls.ensemble_optimization.testing import check_conformance
from eemls.ensemble_optimization.testing.samples import answered, observed
from eemls.testing.trees import TREE

CLASSES = ("car", "bus")

APPROACHES = load_approaches(Reader(TREE))


def rows(queries: int, *, unlabeled: int) -> tuple[Row, ...]:
    return tuple(
        answered(
            f"q-{index}",
            "car" if index % 3 else "bus",
            "car" if index < queries - unlabeled else None,
        )
        for index in range(queries)
    )


def observations() -> tuple[Observation, ...]:
    return tuple(
        observed(serving=("infer-a", "infer-b"), given=rows(4 * (step + 1), unlabeled=4))
        for step in range(4)
    )


def test_every_installed_approach_has_a_committed_policy() -> None:
    assert {one.kind for one in APPROACHES.values()} == set(approaches())


@pytest.mark.parametrize("settings", APPROACHES.values(), ids=lambda one: one.kind)
def test_the_approach_conforms(settings: ApproachSettings) -> None:
    def build() -> Policy:
        return policy_of(settings, classes=CLASSES, seed=3)

    check_conformance(build, observations(), max_ensemble_size=settings.max_ensemble_size)
