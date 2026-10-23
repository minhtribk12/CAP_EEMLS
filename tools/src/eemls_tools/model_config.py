from __future__ import annotations

from typing import Literal

from eemls.config.models import InferenceServicesConfig, ModelRepositoryConfig
from eemls.config.schema import (
    Digest,
    FrozenMap,
    FrozenModel,
    NonNegativeCount,
    Number,
    PositiveCount,
    PositiveNumber,
    Ratio,
)


class BaseModelEntry(FrozenModel):
    name: str | None = None
    hugging_face_id: str | None = None
    onnx_file: str | None = None
    sha256: Digest | None = None
    family: Literal["light_cnn", "cnn", "transformer", "hybrid", "mlp"]
    input_height: PositiveCount
    input_width: PositiveCount
    normalization_mean: tuple[Number, Number, Number]
    normalization_std: tuple[Number, Number, Number]
    weights_mb: PositiveNumber


class VariationRecipeConfig(FrozenModel):
    base_models: FrozenMap[BaseModelEntry]
    specialisms: FrozenMap[tuple[str, ...]]
    variations_per_base_model: NonNegativeCount
    train_images_per_class: NonNegativeCount
    specialist_class_weight: PositiveNumber
    frozen_base_model: bool
    seed: NonNegativeCount
    class_weight_floor: Ratio


class VariationProfile(FrozenModel):
    variant_of: str
    specialism: tuple[str, ...]
    class_weight: PositiveNumber
    train_images_per_class: PositiveCount
    seed: NonNegativeCount
    accuracy_by_class: FrozenMap[Ratio]
    accuracy_by_class_blurred: FrozenMap[Ratio]


class VariationProfilesConfig(FrozenModel):
    variations: FrozenMap[VariationProfile]


class ModelsConfig(FrozenModel):
    model_repository: ModelRepositoryConfig
    inference_services: InferenceServicesConfig
    variations: VariationRecipeConfig
    variation_profiles: VariationProfilesConfig
