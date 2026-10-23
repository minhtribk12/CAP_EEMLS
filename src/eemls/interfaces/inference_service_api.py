from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PredictionAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str | None
    instance_id: str
    service_name: str
    model: str
    device: str
    node: str
    replica: int
    predicted_label: str
    confidence: float
    fetch_ms: float
    inference_ms: float
    total_ms: float
    execution_provider: str
    number_format: str
    cache_hit: bool
    status: str
