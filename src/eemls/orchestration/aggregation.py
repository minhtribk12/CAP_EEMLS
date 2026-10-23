from __future__ import annotations

from collections.abc import Sequence

from eemls.domain.errors import EemlsError
from eemls.domain.types import AggregatedResult, MemberPrediction
from eemls.evaluation_engine.confidence_vote import confidence_vote


class AggregationError(EemlsError): ...


def aggregate_predictions(
    *, query_id: str, deployment_id: str, n_members: int, predictions: Sequence[MemberPrediction]
) -> AggregatedResult:
    vote = confidence_vote(
        (one.predicted_label, one.confidence) for one in predictions if one.in_ensemble
    )
    if vote is None:
        raise AggregationError(f"{query_id}: no member of the ensemble answered")
    return AggregatedResult(
        query_id=query_id,
        predicted_label=vote.label,
        confidence=vote.confidence,
        deployment_id=deployment_id,
        n_members=n_members,
        n_answered=vote.voters,
    )
