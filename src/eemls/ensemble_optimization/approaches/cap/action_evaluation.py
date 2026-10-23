from __future__ import annotations

from collections.abc import Mapping, Sequence

from eemls.domain.records import InferenceServiceProfile
from eemls.ensemble_optimization.approach import ConsumerSpecification
from eemls.evaluation_engine import ensemble_scoring
from eemls.evaluation_engine.contract_scoring import NotScored, ScoreParts
from eemls.evaluation_engine.ensemble_scoring import MemberRow
from eemls.evaluation_engine.profile_estimates import estimated_member_rows, member_rows
from eemls.evaluation_engine.window import Window


def contributions(
    consumer: ConsumerSpecification,
    *,
    member_window: Window,
    ensemble: Sequence[str],
    available: Sequence[str],
    profiles: Mapping[str, InferenceServiceProfile],
    evaluation_window: int,
    window: Window,
) -> Mapping[str, float]:
    rows = member_rows(member_window, profiles)
    inside = set(ensemble)
    labeled = tuple(
        row for row in rows if row.true_label is not None and row.service_name in inside
    )
    serving = tuple(name for name in ensemble if name in {row.service_name for row in labeled})
    if not serving:
        return {}
    scores = CandidateScores(consumer, window)
    held: dict[str, float] = {}
    for name in available:
        one = (
            _of_a_member(
                scores, labeled, serving, _evaluation_window(rows, name, evaluation_window), name
            )
            if name in inside
            else _of_an_idle_one(scores, labeled, serving, profiles.get(name))
        )
        if one is not None:
            held[name] = one
    return held


class CandidateScores:
    def __init__(self, consumer: ConsumerSpecification, window: Window) -> None:
        self._consumer = consumer
        self._window = window
        self._held: dict[tuple[tuple[str, ...], tuple[str, ...]], ScoreParts | NotScored] = {}

    def of(
        self, rows: Sequence[MemberRow], ensemble: tuple[str, ...], queries: tuple[str, ...]
    ) -> ScoreParts | NotScored:
        key = (ensemble, queries)
        if key not in self._held:
            self._held[key] = self._scored(rows, ensemble, queries)
        return self._held[key]

    def contribution(
        self,
        rows: Sequence[MemberRow],
        ensemble: tuple[str, ...],
        member: str,
        queries: tuple[str, ...],
    ) -> float | None:
        without = tuple(name for name in ensemble if name != member)
        if not without:
            return None
        base = self.of(rows, ensemble, queries)
        if member in ensemble:
            whole, rest = base, self._scored(rows, without, queries)
        else:
            whole, rest = self._scored(rows, (*ensemble, member), queries), base
        if isinstance(whole, NotScored) or isinstance(rest, NotScored):
            return None
        return whole.total - rest.total

    def _scored(
        self, rows: Sequence[MemberRow], ensemble: tuple[str, ...], queries: tuple[str, ...]
    ) -> ScoreParts | NotScored:
        return ensemble_scoring.score_candidate(
            self._consumer.contract, self._consumer.cdms, rows, ensemble, queries, self._window
        )


def _evaluation_window(rows: Sequence[MemberRow], name: str, size: int) -> frozenset[str]:
    predicted = [row.query_id for row in rows if row.service_name == name and row.predicted_label]
    return frozenset(predicted[-size:])


def _of_a_member(
    scores: CandidateScores,
    labeled: Sequence[MemberRow],
    serving: tuple[str, ...],
    window: frozenset[str],
    name: str,
) -> float | None:
    on = tuple(row for row in labeled if row.query_id in window)
    queries = ensemble_scoring.answered_by(on, (name,))
    if not queries:
        return None
    present = {row.service_name for row in on}
    return scores.contribution(on, tuple(one for one in serving if one in present), name, queries)


def _of_an_idle_one(
    scores: CandidateScores,
    labeled: Sequence[MemberRow],
    serving: tuple[str, ...],
    one: InferenceServiceProfile | None,
) -> float | None:
    if one is None:
        return None
    would = estimated_member_rows(one, labeled=labeled)
    if not would:
        return None
    queries = ensemble_scoring.answered_by(labeled, serving)
    return scores.contribution((*labeled, *would), serving, one.service_name, queries)
