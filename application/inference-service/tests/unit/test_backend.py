from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from eemls_inference_service.backends.backend import Preprocessing, ranked, softmax

IMAGENET = Preprocessing(
    input_height=224,
    input_width=224,
    normalization_mean=(0.485, 0.456, 0.406),
    normalization_std=(0.229, 0.224, 0.225),
)
HALVES = Preprocessing(
    input_height=224,
    input_width=224,
    normalization_mean=(0.5, 0.5, 0.5),
    normalization_std=(0.5, 0.5, 0.5),
)


def image_bytes(colour: tuple[int, int, int] = (128, 64, 32), size: int = 300) -> bytes:
    held = io.BytesIO()
    Image.new("RGB", (size, size), colour).save(held, format="JPEG")
    return held.getvalue()


class TestTheArrayItBuilds:
    def test_it_is_the_shape_the_model_takes(self) -> None:
        held = IMAGENET.array_of(image_bytes())

        assert held.shape == (1, 3, 224, 224)
        assert held.dtype == np.float32

    def test_an_image_of_another_size_is_resized(self) -> None:
        assert IMAGENET.array_of(image_bytes(size=64)).shape == (1, 3, 224, 224)

    def test_a_grey_image_becomes_three_channels(self) -> None:
        held = io.BytesIO()
        Image.new("L", (80, 80), 120).save(held, format="JPEG")

        assert IMAGENET.array_of(held.getvalue()).shape == (1, 3, 224, 224)

    def test_the_normalization_is_the_one_the_model_was_trained_with(self) -> None:
        grey = image_bytes(colour=(128, 128, 128))

        under_halves = HALVES.array_of(grey)
        under_imagenet = IMAGENET.array_of(grey)

        assert under_halves.mean() == pytest.approx(0.0, abs=0.02)
        assert abs(float(under_imagenet.mean())) > 0.1

    def test_the_same_bytes_give_the_same_array(self) -> None:
        held = image_bytes()

        assert np.array_equal(IMAGENET.array_of(held), IMAGENET.array_of(held))

    def test_bytes_that_are_not_an_image_are_refused(self) -> None:
        from eemls_inference_service.backends.backend import BackendError

        with pytest.raises(BackendError, match="image"):
            IMAGENET.array_of(b"not a picture at all")


class TestTurningLogitsIntoConfidence:
    def test_the_confidences_add_up_to_one(self) -> None:
        held = softmax(np.array([2.0, 1.0, 0.1], dtype=np.float32))

        assert float(held.sum()) == pytest.approx(1.0)

    def test_the_largest_logit_keeps_the_largest_confidence(self) -> None:
        held = softmax(np.array([0.5, 9.0, 1.0], dtype=np.float32))

        assert int(held.argmax()) == 1

    def test_a_very_large_logit_does_not_overflow(self) -> None:
        held = softmax(np.array([1000.0, 0.0], dtype=np.float32))

        assert float(held[0]) == pytest.approx(1.0)
        assert not np.isnan(held).any()


class TestRankingTheClasses:
    def test_it_ranks_the_classes_by_confidence(self) -> None:
        held = ranked(np.array([0.1, 0.7, 0.2]), ("car", "truck", "bus"), top_k=3)

        assert [name for name, _ in held] == ["truck", "bus", "car"]

    def test_it_returns_only_as_many_as_asked_for(self) -> None:
        held = ranked(np.array([0.1, 0.7, 0.2]), ("car", "truck", "bus"), top_k=2)

        assert [name for name, _ in held] == ["truck", "bus"]

    def test_a_tie_is_broken_by_the_class_name(self) -> None:
        held = ranked(np.array([0.5, 0.5]), ("truck", "bus"), top_k=2)

        assert [name for name, _ in held] == ["bus", "truck"]

    def test_as_many_classes_as_the_model_answers_is_required(self) -> None:
        from eemls_inference_service.backends.backend import BackendError

        with pytest.raises(BackendError, match="3"):
            ranked(np.array([0.1, 0.7, 0.2]), ("car", "truck"), top_k=2)
