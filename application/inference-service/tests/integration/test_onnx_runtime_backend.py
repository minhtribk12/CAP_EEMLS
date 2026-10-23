from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from eemls_inference_service.backends.backend import BackendError, Prediction, Preprocessing
from eemls_inference_service.backends.onnx_runtime import OnnxBackend

pytestmark = pytest.mark.integration

# The fixture model answers 1001 scores, named by index.
CLASSES = tuple(f"class_{index}" for index in range(1001))
IMAGENET = Preprocessing(
    input_height=224,
    input_width=224,
    normalization_mean=(0.485, 0.456, 0.406),
    normalization_std=(0.229, 0.224, 0.225),
)


def picture(colour: tuple[int, int, int]) -> bytes:
    held = io.BytesIO()
    Image.new("RGB", (256, 256), colour).save(held, format="JPEG")
    return held.getvalue()


def opened(path: Path, **changes: object) -> OnnxBackend:
    fields: dict[str, object] = {
        "path": path,
        "preprocessing": IMAGENET,
        "classes": CLASSES,
        "providers": ("CPUExecutionProvider",),
        "session_threads": 2,
        "warmup_requests": 1,
    }
    return OnnxBackend(**(fields | changes))  # type: ignore[arg-type]


class TestLoadingIt:
    def test_it_serves_on_a_provider_from_the_chain(self, onnx_model: Path) -> None:
        held = opened(onnx_model)

        assert held.provider() == "CPUExecutionProvider"
        held.close()

    def test_asking_for_a_gpu_this_machine_has_not_falls_back_to_the_cpu(
        self, onnx_model: Path
    ) -> None:
        held = opened(onnx_model, providers=("CUDAExecutionProvider", "CPUExecutionProvider"))

        assert held.provider() == "CPUExecutionProvider"
        held.close()

    def test_a_model_answering_other_classes_than_are_declared_is_refused(
        self, onnx_model: Path
    ) -> None:
        with pytest.raises(BackendError, match="1001"):
            opened(onnx_model, classes=("car", "truck", "bus"))

    def test_a_preprocessing_of_another_size_than_the_model_takes_is_refused(
        self, onnx_model: Path
    ) -> None:
        wrong = IMAGENET.model_copy(update={"input_height": 300, "input_width": 300})

        with pytest.raises(BackendError, match="300"):
            opened(onnx_model, preprocessing=wrong)

    def test_a_file_that_is_not_a_model_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "not-a-model.onnx"
        path.write_bytes(b"certainly not an onnx graph")

        with pytest.raises(BackendError, match="not-a-model"):
            opened(path)


class TestWhatItAnswers:
    def test_a_real_prediction_names_a_declared_class(self, onnx_model: Path) -> None:
        held = opened(onnx_model)

        answer = held.predict(picture((120, 80, 40)))

        assert isinstance(answer, Prediction)
        assert answer.label in CLASSES
        assert 0.0 < answer.confidence <= 1.0
        held.close()

    def test_the_ranking_is_ordered_and_as_long_as_asked_for(self, onnx_model: Path) -> None:
        held = opened(onnx_model, top_k=5)

        answer = held.predict(picture((10, 200, 90)))

        assert len(answer.ranked) == 5
        assert [one for _, one in answer.ranked] == sorted(
            (one for _, one in answer.ranked), reverse=True
        )
        assert answer.label == answer.ranked[0][0]
        held.close()

    def test_the_same_image_answers_the_same_way_twice(self, onnx_model: Path) -> None:
        held = opened(onnx_model)
        image = picture((200, 30, 30))

        first, second = held.predict(image), held.predict(image)

        assert (first.label, first.confidence) == (second.label, second.confidence)
        held.close()

    def test_two_different_images_are_not_answered_identically(self, onnx_model: Path) -> None:
        held = opened(onnx_model)

        one = held.predict(picture((250, 250, 250)))
        other = held.predict(picture((5, 5, 60)))

        assert (one.label, one.confidence) != (other.label, other.confidence)
        held.close()

    def test_bytes_that_are_not_an_image_are_refused(self, onnx_model: Path) -> None:
        held = opened(onnx_model)

        with pytest.raises(BackendError, match="image"):
            held.predict(b"not a picture")
        held.close()
