from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from eemls.domain.consumer_defined_metrics import CdmSet
from eemls.domain.contract import ConsumerContract
from eemls.domain.records import InferenceServiceProfile
from eemls.domain.types import EnsembleDeployment, EnsembleMember, Row, RuntimeContext
from eemls.ensemble_optimization.approach import ConsumerSpecification, Observation
from eemls.evaluation_engine.contract_scoring import ScoreParts
from eemls.evaluation_engine.ensemble_scoring import (
    MemberRow,
    aggregated_window,
    score_candidate,
)
from eemls.evaluation_engine.window import Window
from eemls.testing.records import AT, context, member, profile


def declared(definitions: dict[str, Any]) -> CdmSet:
    return CdmSet.model_validate({"definitions": definitions})


def scored(
    asked: ConsumerContract,
    cdms: dict[str, Any],
    rows: tuple[MemberRow, ...],
    ensemble: tuple[str, ...],
) -> ScoreParts:
    result = score_candidate(asked, declared(cdms), rows, ensemble)
    if not isinstance(result, ScoreParts):
        raise TypeError(result)
    return result


ACCURACY = {
    "min_value": 0.0,
    "max_value": 1.0,
    "weight": 1.0,
    "direction": "maximize",
    "mapping": "linear",
    "mapping_base": None,
}


def contract(cost_weight: float = 0.0) -> ConsumerContract:
    return ConsumerContract.model_validate(
        {
            "consumer_id": "c1_navigation",
            "description": "Counts and classifies vehicles",
            "prioritized_classes": ("car", "truck", "bus"),
            "request_rate_per_s": 20.0,
            "objectives": {"general_accuracy": ACCURACY},
            "cost_weight": cost_weight,
            "cost_range_max_per_1k": 30.0,
            "feedback": {"fraction": 0.5, "delay_intervals": 15},
        }
    )


GENERAL_ACCURACY = {
    "general_accuracy": {
        "measured": "per_window",
        "window": 500,
        "unit": "ratio",
        "description": "Share of labeled queries the result matched",
        "specification": {
            "function": "ratio",
            "inputs": [
                {
                    "function": "count",
                    "scopes": [{"attribute": "predicted_label", "same_as": "true_label"}],
                },
                {"function": "count", "scopes": [{"attribute": "true_label", "observed": True}]},
            ],
        },
    }
}


def row(**changes: Any) -> MemberRow:
    fields: dict[str, Any] = {
        "query_id": "q-1",
        "service_name": "infer-a",
        "predicted_label": "car",
        "confidence": 0.9,
        "true_label": "car",
        "total_ms": 20.0,
        "cost_per_1k": 3.5,
        "throughput_rps": 24.0,
    }
    return MemberRow.model_validate(fields | changes)


THREE_MEMBER_ANSWERS = (
    ("q-1", "car", (("A", "car", 0.80), ("B", "car", 0.70), ("C", "car", 0.90))),
    (
        "q-2",
        "pedestrian",
        (("A", "pedestrian", 0.60), ("B", "cyclist", 0.50), ("C", "pedestrian", 0.70)),
    ),
    ("q-3", "cyclist", (("A", "cyclist", 0.50), ("B", "cyclist", 0.40), ("C", "motorcycle", 0.95))),
    ("q-4", "bus", (("A", "bus", 0.70), ("B", "bus", 0.60), ("C", "bus", 0.80))),
)


def three_member_answers() -> tuple[MemberRow, ...]:
    return tuple(
        row(
            query_id=query_id,
            service_name=name,
            predicted_label=label,
            confidence=confidence,
            true_label=true_label,
        )
        for query_id, true_label, answers in THREE_MEMBER_ANSWERS
        for name, label, confidence in answers
    )


NIGHT_ACCURACY = {
    "night_accuracy": {
        "measured": "per_window",
        "window": 500,
        "unit": "ratio",
        "description": "Share of labeled night queries the result matched",
        "specification": {
            "function": "ratio",
            "inputs": [
                {
                    "function": "count",
                    "scopes": [
                        {"attribute": "condition", "value": "night"},
                        {"attribute": "predicted_label", "same_as": "true_label"},
                    ],
                },
                {
                    "function": "count",
                    "scopes": [
                        {"attribute": "condition", "value": "night"},
                        {"attribute": "true_label", "observed": True},
                    ],
                },
            ],
        },
    }
}


def night_contract() -> ConsumerContract:
    return contract().model_copy(update={"objectives": {"night_accuracy": ACCURACY}})


def live_window(ensemble: tuple[str, ...]) -> Window:
    conditions = {"q-1": "day", "q-2": "night", "q-3": "night", "q-4": "day"}
    return Window(
        tuple(
            {**one, "condition": conditions[str(one["query_id"])]}
            for one in aggregated_window(three_member_answers(), ensemble)
        )
    )


POOL = ("infer-a", "infer-b", "infer-c")

RESPONSE_TIME_ONLY = {
    "response_time_ms": {
        "measured": "per_window",
        "window": 500,
        "unit": "milliseconds",
        "description": "Mean time from the query arriving to the result being ready",
        "specification": {"function": "average", "attribute": "response_time_ms"},
    }
}


def with_name(name: str) -> EnsembleMember:
    return member().model_copy(update={"service_name": name})


def replace_id(deployment: EnsembleDeployment, deployment_id: str) -> EnsembleDeployment:
    return deployment.model_copy(update={"deployment_id": deployment_id})


def answered(query_id: str, label: str, true_label: str | None) -> Row:
    return {
        "query_id": query_id,
        "predicted_label": label,
        "predicted_confidence": 0.8,
        "true_label": true_label,
        "cost_per_1k": 7.0,
        "response_time_ms": 20.0,
        "deployment_id": "d-1",
    }


def profiles(**changes: Mapping[str, object]) -> dict[str, InferenceServiceProfile]:
    return {
        name: profile(service_name=name, instance_id=f"i-{name}", **dict(changes.get(name, {})))
        for name in POOL
    }


def observed(
    *,
    serving: tuple[str, ...] = ("infer-a",),
    right: int = 1,
    wrong: int = 0,
    held: dict[str, InferenceServiceProfile] | None = None,
    at: RuntimeContext | None = None,
    cost_weight: float = 0.0,
    given: tuple[Row, ...] | None = None,
) -> Observation:
    rows = (
        given
        if given is not None
        else tuple(answered(f"q-{index}", "car", "car") for index in range(right))
        + tuple(answered(f"q-w{index}", "bus", "car") for index in range(wrong))
    )
    return Observation(
        consumer=ConsumerSpecification(
            contract=contract(cost_weight=cost_weight),
            cdms=CdmSet.model_validate({"definitions": GENERAL_ACCURACY}),
        ),
        t=AT,
        context=at if at is not None else context(),
        deployment=EnsembleDeployment(
            deployment_id="d-1", members=tuple(with_name(name) for name in serving)
        ),
        available=tuple(with_name(name) for name in POOL),
        profiles=profiles() if held is None else held,
        window=Window(rows),
        member_window=Window(
            tuple(
                {
                    "query_id": str(row["query_id"]),
                    "service_name": name,
                    "predicted_label": row["predicted_label"],
                    "confidence": 0.8,
                    "true_label": row["true_label"],
                    "total_ms": 20.0,
                }
                for row in rows
                for name in serving
            )
        ),
        cdm_values={},
        score=None,
    )
