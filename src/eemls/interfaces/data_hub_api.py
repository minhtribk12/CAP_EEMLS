from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ImageMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    label: str
    dataset: str
    original_label: str
    condition: str
    image_quality: str
