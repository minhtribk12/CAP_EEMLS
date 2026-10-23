from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from eemls.domain.types import DEFAULT_IMAGE_QUALITY, ImageQuality, TimeOfDay


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    consumer_id: str
    image_id: str
    reply_topic: str
    correlation_id: str
    image_quality: ImageQuality = DEFAULT_IMAGE_QUALITY
    time_of_day: TimeOfDay | None = None


class ReturnedLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    true_label: str
