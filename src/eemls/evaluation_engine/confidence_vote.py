from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from eemls.domain.types import FrozenModel, Identifier, PositiveCount, Ratio


class Vote(FrozenModel):
    label: Identifier
    confidence: Ratio
    voters: PositiveCount


def confidence_vote(answers: Iterable[tuple[str | None, float | None]]) -> Vote | None:
    named: dict[str, list[float]] = {}
    for label, confidence in answers:
        if label is not None and confidence is not None:
            named.setdefault(label, []).append(confidence)
    if not named:
        return None
    label, confidences = max(sorted(named.items()), key=lambda held: mean(held[1]))
    return Vote(
        label=label,
        confidence=mean(confidences),
        voters=sum(len(held) for held in named.values()),
    )


def mean(confidences: Sequence[float]) -> float:
    return math.fsum(sorted(confidences)) / len(confidences)
