from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eemls.config.datasets import DatasetClasses
from eemls.config.models import ModelEntry
from eemls.config.reading import Reader, root_of
from eemls_inference_service.backends.onnx_runtime import TOP_K
from eemls_inference_service.config import load_inference_config
from eemls_inference_service.model_loading import load

ROOT = Path(__file__).resolve().parents[4]
UNDER = ROOT / "models"
MODEL = "b3"
DIGEST = "a86af476bbfa58ec86cae5f17d3d17d88c3c2b489e8ae2bd7e02421257f14dc2"

# ImageNet classes that count as a car.
VEHICLES = frozenset(
    {
        "sports car, sport car",
        "convertible",
        "beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon",
        "cab, hack, taxi, taxicab",
        "limousine, limo",
        "minivan",
        "jeep, landrover",
        "pickup, pickup truck",
        "car wheel",
        "grille, radiator grille",
        "racer, race car, racing car",
    }
)

pytestmark = pytest.mark.integration


def fixture_of(name: str) -> Path:
    held = UNDER / name
    if not (held / "model.onnx").exists():
        pytest.skip(f"{held / 'model.onnx'} is missing; run: tools/fetch_models.py {name}")
    return held


def cars(count: int = 5) -> tuple[bytes, ...]:
    held = sorted((ROOT / "data" / "hub" / "day" / "clear" / "car").glob("open_images-*.jpg"))
    if len(held) < count:
        pytest.skip("the prepared dataset is not there; run: python3 tools/prepare_dataset.py")
    return tuple(path.read_bytes() for path in held[:count])


def entry_of(where: Path) -> ModelEntry:
    labels: dict[str, Any] = json.loads((where / "config.json").read_text(encoding="utf-8"))
    prepared: dict[str, Any] = json.loads(
        (where / "preprocessor_config.json").read_text(encoding="utf-8")
    )
    named = labels["id2label"]
    return ModelEntry.model_validate(
        {
            "hugging_face_id": "google/b3_1.0_224",
            "base_model": MODEL,
            "onnx_file": "model.onnx",
            "sha256": DIGEST,
            "input_height": prepared["crop_size"]["height"],
            "input_width": prepared["crop_size"]["width"],
            "normalization_mean": tuple(prepared["image_mean"]),
            "normalization_std": tuple(prepared["image_std"]),
            "classes": tuple(named[str(index)] for index in range(len(named))),
            "memory_mb": 14.0,
            "variant_of": None,
        }
    )


class TestTheExportOnDisk:
    def test_the_file_is_the_one_the_entry_declares(self) -> None:
        where = fixture_of(MODEL)

        held = load(
            MODEL,
            entry_of(where),
            under=UNDER,
            providers=("CPUExecutionProvider",),
            session_threads=1,
            warmup_requests=0,
        )

        assert held.provider() == "CPUExecutionProvider"
        held.close()

    def test_the_labels_come_from_the_export_and_cover_imagenet(self) -> None:
        held = entry_of(fixture_of(MODEL))

        assert len(held.classes) == 1001
        assert held.classes[0] == "background"


class TestWhatItAnswersAboutARealImage:
    def backend(self) -> Any:
        return load(
            MODEL,
            entry_of(fixture_of(MODEL)),
            under=UNDER,
            providers=("CPUExecutionProvider",),
            session_threads=1,
            warmup_requests=1,
        )

    def test_a_photograph_of_a_car_is_recognized_as_a_kind_of_car(self) -> None:
        held = self.backend()

        answers = [held.predict(image) for image in cars(8)]
        held.close()

        vehicles = [one for one in answers if one.label in VEHICLES]
        assert len(vehicles) >= 2, [one.label for one in answers]

    def test_it_is_confident_about_the_ones_it_recognizes(self) -> None:
        held = self.backend()

        answers = [held.predict(image) for image in cars(8)]
        held.close()

        best = max(answers, key=lambda one: one.confidence)
        assert best.label in VEHICLES
        assert best.confidence > 0.5

    def test_every_answer_is_ranked_with_its_own_label_first(self) -> None:
        held = self.backend()

        answers = [held.predict(image) for image in cars(8)]
        held.close()

        for one in answers:
            assert one.ranked[0] == (one.label, one.confidence)
            assert 0.0 < one.confidence <= 1.0

    def test_the_same_image_gives_the_same_answer_twice(self) -> None:
        held = self.backend()
        image = cars(1)[0]

        one = held.predict(image)
        other = held.predict(image)
        held.close()

        assert (one.label, one.confidence) == (other.label, other.confidence)


def variant(name: str) -> Any:
    config = load_inference_config(ROOT / "config")
    entry = config.models.model_repository.models.get(name)
    if entry is None or entry.onnx_file is None or not (UNDER / name / entry.onnx_file).exists():
        pytest.skip(f"{name} is not built; run: python3 tools/build_variations.py --build")
    return load(
        name,
        entry,
        under=UNDER,
        providers=("CPUExecutionProvider",),
        session_threads=1,
        warmup_requests=1,
    )


def images_of(label: str, count: int = 8) -> tuple[bytes, ...]:
    held = sorted((ROOT / "data" / "hub" / "day" / "clear" / label).glob("open_images-*.jpg"))
    if len(held) < count:
        pytest.skip("the prepared dataset is not there; run: python3 tools/prepare_dataset.py")
    return tuple(path.read_bytes() for path in held[:count])


class TestAVariationAnswersTheClassesAConsumerAsksAbout:
    def test_every_answer_is_a_class_the_dataset_holds(self) -> None:
        held = variant("b3_general")
        classes = set(
            Reader(root_of(ROOT / "config")).build(DatasetClasses, "datasets/sources.yaml").classes
        )

        answers = [held.predict(image) for image in images_of("car")]
        held.close()

        assert {one.label for one in answers} <= classes

    def test_a_car_is_the_answer_a_car_gets_most_often(self) -> None:
        import collections

        held = variant("b3_general")

        answers = collections.Counter(held.predict(image).label for image in images_of("car", 40))
        held.close()

        ranked = answers.most_common()
        assert ranked[0][0] == "car", ranked
        assert ranked[0][1] > 2 * ranked[1][1], ranked

    def test_it_ranks_the_few_classes_a_service_reports(self) -> None:
        held = variant("b3_general")

        answer = held.predict(images_of("car")[1])
        held.close()

        assert len(answer.ranked) == TOP_K
        assert [one for _, one in answer.ranked] == sorted(
            (one for _, one in answer.ranked), reverse=True
        )
        assert answer.confidence <= 1.0

    def test_it_is_sure_about_the_ones_it_recognizes(self) -> None:
        held = variant("b3_general")

        answers = [held.predict(image) for image in images_of("car", 40)]
        held.close()

        sure = [one for one in answers if one.label == "car" and one.confidence > 0.9]
        assert len(sure) >= 8, len(sure)


class TestTheSpecialistEarnsItsPlaceInThePool:
    def test_it_beats_its_own_generalist_on_the_classes_it_was_weighted_for(self) -> None:
        general = variant("b3_general")
        special = variant("b3_motorcycle")
        images = images_of("motorcycle")

        theirs = sum(1 for one in images if general.predict(one).label == "motorcycle")
        mine = sum(1 for one in images if special.predict(one).label == "motorcycle")
        general.close()
        special.close()

        assert mine >= theirs, (mine, theirs)
