from __future__ import annotations

from collections.abc import Sequence

from eemls.domain.types import Action
from eemls.ensemble_optimization.approach import (
    API_VERSION,
    Approach,
    ApproachSettings,
    Observation,
    PolicyDecision,
)


class KeepTheEnsemble:
    name = "keep_the_ensemble"

    def decide(self, observed: Observation) -> PolicyDecision:
        return PolicyDecision(action=Action(kind="keep"))


def build(settings: ApproachSettings, *, classes: Sequence[str], seed: int) -> KeepTheEnsemble:
    return KeepTheEnsemble()


approach = Approach(
    name=KeepTheEnsemble.name, api_version=API_VERSION, Settings=ApproachSettings, build=build
)
