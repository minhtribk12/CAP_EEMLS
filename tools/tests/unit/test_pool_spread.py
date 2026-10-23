from __future__ import annotations

import pytest

from eemls.config.models import ModelRepositoryConfig
from eemls_tools.config_tree import load_config
from eemls_tools.model_config import VariationProfile, VariationProfilesConfig
from support.repository import ROOT

TREE = ROOT / "config"


def variation_profiles() -> VariationProfilesConfig:
    held = load_config(TREE).models.variation_profiles
    if not held.variations:
        pytest.skip("the pool is not built; run: python3 tools/build_variations.py --build")
    return held


def repository() -> ModelRepositoryConfig:
    return load_config(TREE).models.model_repository


def mean_of(reached: dict[str, float], over: tuple[str, ...]) -> float:
    held = [reached[name] for name in over if name in reached]
    return sum(held) / len(held) if held else 0.0


def generalist_of(name: str, held: VariationProfilesConfig) -> VariationProfile:
    return held.variations[f"{held.variations[name].variant_of}_general"]


class TestWhatThePoolHolds:
    def test_it_holds_two_variations_of_every_backbone(self) -> None:
        held = variation_profiles()
        recipe = load_config(TREE).models.variations

        assert len(held.variations) == len(recipe.base_models) * recipe.variations_per_base_model

    def test_every_variation_is_registered_like_any_other_model(self) -> None:
        if not (TREE / "local" / "models.yaml").is_file():
            pytest.skip("config/local/models.yaml is not here; run: tools/new_local.py")
        held = variation_profiles()
        models = repository().models

        for name, one in held.variations.items():
            assert name in models, name
            assert models[name].variant_of == one.variant_of, name
            assert models[name].onnx_file == f"{name}.onnx", name

    def test_every_variation_answers_the_classes_the_dataset_holds(self) -> None:
        models = repository().models
        classes = load_config(TREE).datasets.classes

        for name in variation_profiles().variations:
            assert models[name].classes == classes, name

    def test_every_variation_records_how_it_was_built(self) -> None:
        recipe = load_config(TREE).models.variations

        for name, one in variation_profiles().variations.items():
            assert one.train_images_per_class == recipe.train_images_per_class, name
            assert one.seed == recipe.seed, name
            assert one.class_weight == recipe.specialist_class_weight, name

    def test_one_of_every_pair_is_a_generalist_and_one_a_specialist(self) -> None:
        held = variation_profiles().variations
        generals = [name for name, one in held.items() if not one.specialism]
        specialists = [name for name, one in held.items() if one.specialism]

        assert len(generals) == len(specialists)
        assert all(name.endswith("_general") for name in generals)

    def test_every_specialist_is_weighted_for_the_classes_the_recipe_names(self) -> None:
        recipe = load_config(TREE).models.variations

        for name, one in variation_profiles().variations.items():
            if one.specialism:
                assert tuple(one.specialism) == tuple(recipe.specialisms[one.variant_of]), name


class TestTheSpreadAConsumerChoosesBetween:
    def test_a_specialist_is_ahead_on_the_classes_it_was_weighted_for(self) -> None:
        held = variation_profiles()
        ahead = []

        for name, one in held.variations.items():
            if not one.specialism:
                continue
            general = generalist_of(name, held)
            mine = mean_of(dict(one.accuracy_by_class), tuple(one.specialism))
            theirs = mean_of(dict(general.accuracy_by_class), tuple(one.specialism))
            ahead.append((name, mine - theirs))

        assert [one for one in ahead if one[1] > 0.0], ahead

    def test_a_specialist_pays_for_it_over_the_classes_as_a_whole(self) -> None:
        held = variation_profiles()
        behind = []

        for name, one in held.variations.items():
            if not one.specialism:
                continue
            general = generalist_of(name, held)
            mine = sum(one.accuracy_by_class.values()) / len(one.accuracy_by_class)
            theirs = sum(general.accuracy_by_class.values()) / len(general.accuracy_by_class)
            behind.append((name, mine - theirs))

        assert all(one[1] <= 0.0 for one in behind), behind

    def test_no_one_variation_is_the_best_answer_for_every_class(self) -> None:
        held = variation_profiles().variations
        best = set()

        for name in load_config(TREE).datasets.classes:
            scored = {
                one: record.accuracy_by_class[name]
                for one, record in held.items()
                if name in record.accuracy_by_class
            }
            if scored:
                best.add(max(sorted(scored), key=lambda one: scored[one]))

        assert len(best) > 1, best

    def test_the_pool_spans_a_range_of_accuracy(self) -> None:
        held = variation_profiles().variations
        means = [
            sum(one.accuracy_by_class.values()) / len(one.accuracy_by_class)
            for one in held.values()
        ]

        assert max(means) - min(means) > 0.05


class TestWhatTheBlurredCopiesDo:
    def test_every_variation_loses_accuracy_on_the_blurred_copies(self) -> None:
        fell = []

        for name, one in variation_profiles().variations.items():
            clear = sum(one.accuracy_by_class.values()) / len(one.accuracy_by_class)
            blurred = sum(one.accuracy_by_class_blurred.values()) / len(
                one.accuracy_by_class_blurred
            )
            fell.append((name, clear - blurred))

        assert all(one[1] >= 0.0 for one in fell), fell
        assert [one for one in fell if one[1] > 0.02], fell

    def test_the_blurred_accuracy_is_recorded_for_the_same_classes(self) -> None:
        for name, one in variation_profiles().variations.items():
            assert set(one.accuracy_by_class) == set(one.accuracy_by_class_blurred), name
