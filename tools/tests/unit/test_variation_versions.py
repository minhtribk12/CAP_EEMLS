from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eemls_tools.datasets import DatasetsConfig
from eemls_tools.model_config import VariationRecipeConfig
from support.repository import ROOT

VERSIONS = sorted((ROOT / "experiments" / "variations").glob("v*.yaml"))
BACKBONES = {f"b{number}" for number in range(1, 17)}
BACKBONES_PER_VERSION = 8
# Tail classes with too few clear images to fit a specialist on.
SCARCE = {"guardrail", "traffic_cone", "road_bump", "car_stop_sign", "road_curve"}


def _recipe(path: Path) -> VariationRecipeConfig:
    return VariationRecipeConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _dataset_classes() -> set[str]:
    body = yaml.safe_load(
        (ROOT / "config" / "datasets" / "sources.yaml").read_text(encoding="utf-8")
    )
    return set(DatasetsConfig.model_validate(body).classes)


def test_there_are_twenty_versions() -> None:
    assert len(VERSIONS) == 20


@pytest.mark.parametrize("path", VERSIONS, ids=lambda path: path.stem)
def test_a_version_is_a_recipe_over_eight_known_backbones(path: Path) -> None:
    recipe = _recipe(path)
    assert len(recipe.base_models) == BACKBONES_PER_VERSION
    assert set(recipe.base_models) <= BACKBONES


@pytest.mark.parametrize("path", VERSIONS, ids=lambda path: path.stem)
def test_a_version_names_a_specialism_for_each_backbone_from_the_dataset(path: Path) -> None:
    recipe = _recipe(path)
    assert set(recipe.specialisms) == set(recipe.base_models)
    named = {name for classes in recipe.specialisms.values() for name in classes}
    assert named <= _dataset_classes()
    assert all(1 <= len(classes) <= 3 for classes in recipe.specialisms.values())
    assert not named & SCARCE


@pytest.mark.parametrize("path", VERSIONS, ids=lambda path: path.stem)
def test_a_specialist_is_named_after_a_class_without_an_underscore(path: Path) -> None:
    assert all("_" not in classes[0] for classes in _recipe(path).specialisms.values())


@pytest.mark.parametrize("path", VERSIONS, ids=lambda path: path.stem)
def test_a_version_keeps_the_settings_build_variations_builds_with(path: Path) -> None:
    recipe = _recipe(path)
    assert recipe.variations_per_base_model == 2
    assert recipe.class_weight_floor == 0.0
    assert recipe.frozen_base_model


def test_a_backbone_has_the_same_description_in_every_version() -> None:
    seen: dict[str, object] = {}
    for path in VERSIONS:
        for key, entry in _recipe(path).base_models.items():
            assert seen.setdefault(key, entry) == entry, f"{path.stem}: {key}"


def test_the_versions_together_cover_every_backbone_and_family() -> None:
    recipes = [_recipe(path) for path in VERSIONS]
    keys = {key for recipe in recipes for key in recipe.base_models}
    families = {entry.family for recipe in recipes for entry in recipe.base_models.values()}
    assert keys == BACKBONES
    assert families == {"light_cnn", "cnn", "transformer", "hybrid", "mlp"}


def test_no_two_versions_are_the_same() -> None:
    bodies = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in VERSIONS]
    assert len({yaml.safe_dump(body, sort_keys=True) for body in bodies}) == len(bodies)
