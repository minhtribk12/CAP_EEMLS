from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator

from eemls.config.schema import (
    Digest,
    FrozenMap,
    FrozenModel,
    NonNegativeCount,
    Number,
    PositiveCount,
    PositiveNumber,
)


class InferenceServiceEntry(FrozenModel):
    model: str
    device_type: str
    replicas: PositiveCount


class Autoscale(FrozenModel):
    """Copies of each service scale between its `replicas` and `max_replicas`."""

    enabled: bool
    max_replicas: NonNegativeCount
    cpu_percent: PositiveCount
    cooldown_s: Annotated[int, Field(ge=0, le=3600, strict=True)]


class InferenceServicesConfig(FrozenModel):
    services: FrozenMap[InferenceServiceEntry]
    autoscale: Autoscale

    @model_validator(mode="after")
    def _the_ceiling_holds_every_service(self) -> InferenceServicesConfig:
        ceiling = self.autoscale.max_replicas
        for name, service in sorted(self.services.items()):
            if service.replicas > ceiling:
                raise ValueError(
                    f"autoscale max_replicas {ceiling} is below the copies "
                    f"{name} starts with {service.replicas}"
                )
        return self


class ModelSource(FrozenModel):
    hugging_face_id: str
    onnx_file: str
    sha256: Digest


class BackboneSource(ModelSource):
    name: str | None = None


class ModelSourcesConfig(FrozenModel):
    models: FrozenMap[ModelSource] = Field(default_factory=dict)
    base_models: FrozenMap[BackboneSource] = Field(default_factory=dict)


class ModelEntry(FrozenModel):
    """One exported model; source, file and digest are None without `local/models.yaml`."""

    hugging_face_id: str | None = None
    base_model: str
    onnx_file: str | None = None
    sha256: Digest | None = None
    input_height: PositiveCount
    input_width: PositiveCount
    normalization_mean: tuple[Number, Number, Number]
    normalization_std: tuple[Number, Number, Number]
    classes: tuple[str, ...]
    memory_mb: PositiveNumber
    variant_of: str | None


class ModelRepositoryConfig(FrozenModel):
    models: FrozenMap[ModelEntry]
