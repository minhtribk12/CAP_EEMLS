from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from eemls.domain.consumer_defined_metrics import CdmSet
from eemls.domain.contract import ConsumerContract
from eemls.domain.records import ServedQuery
from eemls.domain.statistics import is_number
from eemls.domain.types import (
    FrozenModel,
    Identifier,
    Milliseconds,
    NonNegativeNumber,
    Query,
    Ratio,
    Row,
    RuntimeContext,
)
from eemls.evaluation_engine.aggregation_functions import CdmError, Value
from eemls.evaluation_engine.confidence_vote import confidence_vote
from eemls.evaluation_engine.contract_scoring import NotScored, ScoreParts, score
from eemls.evaluation_engine.engine import EvaluationEngine
from eemls.evaluation_engine.window import Window

QUERY_CONTEXT = (
    *(name for name in Query.model_fields if name in ServedQuery.model_fields),
    *RuntimeContext.model_fields,
)


class MemberRow(FrozenModel):
    query_id: Identifier
    service_name: Identifier
    predicted_label: Identifier | None
    confidence: Ratio | None
    true_label: Identifier | None
    total_ms: Milliseconds
    cost_per_1k: NonNegativeNumber
    throughput_rps: NonNegativeNumber


def voted_answer(answers: Sequence[MemberRow]) -> tuple[str, float] | None:
    vote = confidence_vote((answer.predicted_label, answer.confidence) for answer in answers)
    if vote is None:
        return None
    return vote.label, vote.confidence


def aggregated_window(
    rows: Sequence[MemberRow],
    ensemble: Sequence[str],
    queries: Sequence[str] | None = None,
    window: Window | None = None,
) -> Window:
    serving = tuple(ensemble)
    if not serving:
        raise CdmError("a candidate ensemble holds no member")
    repeated = sorted({name for name in serving if serving.count(name) > 1})
    if repeated:
        raise CdmError(f"the candidate names {repeated} more than once")
    known = {row.service_name for row in rows}
    unknown = sorted(set(serving) - known)
    if unknown:
        raise CdmError(f"no row holds the members {unknown}")
    costs = _per_member(rows, lambda row: row.cost_per_1k)
    rates = _per_member(rows, lambda row: row.throughput_rps)
    members = set(serving)
    labels = _true_labels(rows)
    by_query: dict[str, list[MemberRow]] = {query_id: [] for query_id in queries or ()}
    for row in rows:
        if row.service_name in members and (queries is None or row.query_id in by_query):
            by_query.setdefault(row.query_id, []).append(row)
    contexts = {
        str(one["query_id"]): {name: one[name] for name in QUERY_CONTEXT if name in one}
        for one in window or ()
    }
    return Window(
        tuple(
            {
                **contexts.get(query_id, {}),
                **_aggregated_row(query_id, answers, labels.get(query_id), serving, costs, rates),
            }
            for query_id, answers in by_query.items()
        )
    )


def _per_member(rows: Sequence[MemberRow], of: Callable[[MemberRow], float]) -> Mapping[str, float]:
    held: dict[str, list[float]] = {}
    for row in rows:
        held.setdefault(row.service_name, []).append(of(row))
    return {name: sum(values) / len(values) for name, values in held.items()}


def _aggregated_row(
    query_id: str,
    answers: Sequence[MemberRow],
    true_label: str | None,
    serving: Sequence[str],
    costs: Mapping[str, float],
    rates: Mapping[str, float],
) -> Row:
    cast = voted_answer(answers)
    return {
        "query_id": query_id,
        "predicted_label": cast[0] if cast else None,
        "predicted_confidence": cast[1] if cast else None,
        "true_label": true_label,
        "response_time_ms": max(answer.total_ms for answer in answers) if answers else None,
        "cost_per_1k": sum(costs[name] for name in serving),
        "throughput_rps": min(rates[name] for name in serving),
        "n_members": len(serving),
        "n_answered": sum(1 for answer in answers if answer.predicted_label is not None),
    }


def _true_labels(rows: Sequence[MemberRow]) -> Mapping[str, str]:
    held: dict[str, set[str]] = {}
    for row in rows:
        if row.true_label is not None:
            held.setdefault(row.query_id, set()).add(row.true_label)
    for query_id, labels in held.items():
        if len(labels) > 1:
            raise CdmError(
                f"the members of {query_id} disagree on its true label: {sorted(labels)}"
            )
    return {query_id: min(labels) for query_id, labels in held.items()}


def scored_window(
    contract: ConsumerContract,
    cdms: CdmSet,
    window: Window,
) -> tuple[Mapping[str, Value], ScoreParts | NotScored]:
    evaluator = EvaluationEngine(cdms)
    wanted = contract.scored() | {level.cdm for level in contract.requirements}
    values = {name: evaluator.of(name, window) for name in wanted if name in cdms.definitions}
    return values, score(contract, values, cost_per_1k=_cost_per_1k(window))


def score_candidate(
    contract: ConsumerContract,
    cdms: CdmSet,
    rows: Sequence[MemberRow],
    ensemble: Sequence[str],
    queries: Sequence[str] | None = None,
    window: Window | None = None,
) -> ScoreParts | NotScored:
    _, parts = scored_window(contract, cdms, aggregated_window(rows, ensemble, queries, window))
    return parts


def _cost_per_1k(window: Window) -> float:
    costs = []
    for row in window:
        held = row["cost_per_1k"]
        if held is None:
            continue
        if not is_number(held):
            raise CdmError(f"a query was priced at {held!r}, which is not a cost")
        costs.append(float(held))
    if not costs:
        raise CdmError("no query of the window was priced, so it cannot be scored")
    return sum(costs) / len(costs)


def answered_by(rows: Sequence[MemberRow], ensemble: Sequence[str]) -> tuple[str, ...]:
    members = set(ensemble)
    return tuple(dict.fromkeys(row.query_id for row in rows if row.service_name in members))
