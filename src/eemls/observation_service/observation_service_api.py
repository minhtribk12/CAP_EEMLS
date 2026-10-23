from __future__ import annotations

from eemls.domain.errors import EemlsError
from eemls.domain.records import Decision, Feedback, InferenceServiceProfile, ServedQuery
from eemls.domain.types import (
    FrozenModel,
    Identifier,
    MemberPrediction,
    NonNegativeCount,
    Timestamp,
)


class ObservationServiceError(EemlsError): ...


class RowBatch(FrozenModel):
    run_id: Identifier
    sender: Identifier
    queries: tuple[ServedQuery, ...] = ()
    member_predictions: tuple[MemberPrediction, ...] = ()
    feedback: tuple[Feedback, ...] = ()
    decisions: tuple[Decision, ...] = ()

    @property
    def rows(self) -> int:
        return (
            len(self.queries)
            + len(self.member_predictions)
            + len(self.feedback)
            + len(self.decisions)
        )


class Stored(FrozenModel):
    stored: NonNegativeCount


class ProfileUpdateRequest(FrozenModel):
    at: Timestamp


class ProfileUpdateAnswer(FrozenModel):
    profiles: tuple[InferenceServiceProfile, ...]
