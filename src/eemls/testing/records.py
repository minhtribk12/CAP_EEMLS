from __future__ import annotations

from datetime import UTC, datetime

from eemls.domain.records import Decision, Feedback, InferenceServiceProfile, Instance, ServedQuery
from eemls.domain.types import (
    Action,
    EnsembleMember,
    MemberPrediction,
    Query,
    RuntimeContext,
)

AT = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)


def member(model: str = "b6", device: str = "jetson_orin_nano") -> EnsembleMember:
    return EnsembleMember(service_name=f"infer-{model}-{device}", model=model, device=device)


def context(**changes: object) -> RuntimeContext:
    fields: dict[str, object] = {
        "period_name": "evening_rush",
        "period_index": 3,
        "condition": "night",
        "weather": "fog",
    }
    return RuntimeContext.model_validate(fields | changes)


def query(**changes: object) -> Query:
    fields: dict[str, object] = {
        "query_id": "q-1",
        "consumer_id": "c1_navigation",
        "image_id": "img-7",
        "reply_topic": "eemls/results/c1_navigation",
        "correlation_id": "corr-1",
        "t_arrive": AT,
    }
    return Query.model_validate(fields | changes)


def served(**changes: object) -> ServedQuery:
    fields: dict[str, object] = {
        "query_id": "q-1",
        "consumer_id": "c1_navigation",
        "image_id": "img-7",
        "t_arrive": AT,
        "t_result_ready": AT.replace(microsecond=174000),
        "wait_ms": 160.0,
        "aggregate_ms": 1.5,
        "context": context(),
        "deployment_id": "v-1",
        "n_members": 3,
        "n_answered": 3,
        "predicted_label": "car",
        "predicted_confidence": 0.82,
        "cost_per_1k": 10.5,
        "throughput_rps": 18.0,
        "outcome": "accepted",
    }
    return ServedQuery.model_validate(fields | changes)


def prediction(**changes: object) -> MemberPrediction:
    fields: dict[str, object] = {
        "query_id": "q-1",
        "instance_id": "i-1",
        "member": member(),
        "node": "orin-1",
        "replica": 1,
        "in_ensemble": True,
        "predicted_label": "car",
        "confidence": 0.9,
        "fetch_ms": 2.0,
        "inference_ms": 18.0,
        "total_ms": 21.0,
        "provider": "CUDAExecutionProvider",
        "number_format": "float32",
        "cache_hit": True,
        "status": "ok",
    }
    return MemberPrediction.model_validate(fields | changes)


def feedback(**changes: object) -> Feedback:
    fields: dict[str, object] = {
        "query_id": "q-1",
        "consumer_id": "c1_navigation",
        "true_label": "car",
        "t_feedback": AT,
        "delay_ms": 15000.0,
    }
    return Feedback.model_validate(fields | changes)


def decision(**changes: object) -> Decision:
    fields: dict[str, object] = {
        "decision_id": "d-1",
        "consumer_id": "c1_navigation",
        "t": AT,
        "policy": "cap",
        "action": Action(kind="keep"),
        "context": context(),
        "accepted": False,
        "reason": "no candidate cleared the margin",
        "deployment_before": "v-1",
        "deployment_after": "v-1",
        "score": 0.81,
        "cdm_values": {"accuracy_car": 0.86},
        "candidates": 5,
        "scored": None,
        "decide_ms": 38.0,
        "cdm_ms": 20.0,
        "contribution_ms": 10.0,
        "select_ms": 5.0,
    }
    return Decision.model_validate(fields | changes)


def instance(**changes: object) -> Instance:
    fields: dict[str, object] = {
        "instance_id": "i-1",
        "service_name": "infer-b6-orin",
        "model": "b6",
        "device": "jetson_orin_nano",
        "node": "orin-1",
        "replica": 2,
        "provider": "TensorrtExecutionProvider",
        "number_format": "float32",
        "runtime_version": "1.20.1",
        "image_tag": "eemls/inference:jetson-abc123",
        "config_hash": "a" * 64,
        "started_at": AT,
    }
    return Instance.model_validate(fields | changes)


def profile(**changes: object) -> InferenceServiceProfile:
    fields: dict[str, object] = {
        "model": "b6",
        "device": "jetson_orin_nano",
        "node": "orin-1",
        "instance_id": "i-1",
        "service_name": "infer-b6-orin",
        "window_start": AT,
        "window_end": AT.replace(minute=5),
        "mean_ms": 42.0,
        "p50_ms": 40.0,
        "p95_ms": 61.0,
        "max_ms": 88.0,
        "throughput_rps": 23.5,
        "n_queries": 500,
        "energy_wh_per_1k": 1.8,
        "cost_per_1k": 3.5,
        "general_accuracy": 0.81,
        "accuracy_by_class": {"car": 0.9, "truck": 0.72},
        "confidence_by_class": {"car": 0.88, "truck": 0.64},
        "source": "measured",
    }
    return InferenceServiceProfile.model_validate(fields | changes)
