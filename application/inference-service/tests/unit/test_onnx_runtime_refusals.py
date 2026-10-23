from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from eemls_inference_service.backends.backend import BackendError, Preprocessing
from eemls_inference_service.backends.onnx_runtime import (
    OnnxBackend,
    gpu_libraries_loaded,
    providers_that_did_not_start,
)

CLASSES = ("car", "bus", "cyclist")


def model(
    path: Path, *, size: int = 8, classes: int = 3, weights: np.ndarray | None = None
) -> Path:
    """A stand-in model: the mean of each channel, turned into one score per class."""
    held = np.eye(3, classes, dtype=np.float32) if weights is None else weights
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["image", "axes"], ["channels"], keepdims=0),
            helper.make_node("MatMul", ["channels", "head"], ["scores"]),
        ],
        "stand_in",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, classes])],
        [
            numpy_helper.from_array(held, name="head"),
            numpy_helper.from_array(np.asarray([2, 3], dtype=np.int64), name="axes"),
        ],
    )
    whole = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    whole.ir_version = 10
    path.write_bytes(whole.SerializeToString())
    return path


def prepared(size: int = 8) -> Preprocessing:
    return Preprocessing(
        input_height=size,
        input_width=size,
        normalization_mean=(0.5, 0.5, 0.5),
        normalization_std=(0.5, 0.5, 0.5),
    )


def backend(
    path: Path, *, size: int = 8, classes: tuple[str, ...] = CLASSES, **changes: object
) -> OnnxBackend:
    return OnnxBackend(
        path=path,
        preprocessing=prepared(size),
        classes=classes,
        providers=changes.pop("providers", ("CPUExecutionProvider",)),  # type: ignore[arg-type]
        session_threads=1,
        warmup_requests=changes.pop("warmup_requests", 0),  # type: ignore[arg-type]
        **changes,  # type: ignore[arg-type]
    )


def an_image(colour: tuple[int, int, int] = (255, 0, 0), size: int = 8) -> bytes:
    import io

    from PIL import Image

    held = io.BytesIO()
    Image.new("RGB", (size, size), colour).save(held, format="PNG")
    return held.getvalue()


class TestTheRankingItReports:
    def test_the_class_with_the_highest_score_is_the_answer(self, tmp_path: Path) -> None:
        held = backend(model(tmp_path / "m.onnx"))

        answer = held.predict(an_image((255, 0, 0)))
        held.close()

        assert answer.label == "car"

    def test_the_confidences_are_shares_that_add_up(self, tmp_path: Path) -> None:
        held = backend(model(tmp_path / "m.onnx"))

        answer = held.predict(an_image())
        held.close()

        assert sum(one for _, one in answer.ranked) == pytest.approx(1.0)
        assert answer.confidence == answer.ranked[0][1]

    def test_it_ranks_no_more_than_it_was_asked_for(self, tmp_path: Path) -> None:
        held = backend(model(tmp_path / "m.onnx"), top_k=2)

        answer = held.predict(an_image())
        held.close()

        assert len(answer.ranked) == 2


class TestWhatItRefuses:
    def test_a_model_taking_another_image_size_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(BackendError, match="takes"):
            backend(model(tmp_path / "m.onnx", size=16), size=8)

    def test_the_refusal_names_both_sizes(self, tmp_path: Path) -> None:
        with pytest.raises(BackendError, match="16.*8|8.*16"):
            backend(model(tmp_path / "m.onnx", size=16), size=8)

    def test_a_model_answering_another_number_of_classes_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(BackendError, match="answers 5"):
            backend(model(tmp_path / "m.onnx", classes=5))

    def test_a_provider_nothing_here_has_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(BackendError, match="is available here"):
            backend(model(tmp_path / "m.onnx"), providers=("MoonLanderExecutionProvider",))


class TestTheProvidersItSettlesFor:
    def test_a_provider_this_machine_lacks_is_skipped_and_said_so(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            held = backend(
                model(tmp_path / "m.onnx"),
                providers=("MoonLanderExecutionProvider", "CPUExecutionProvider"),
            )

        assert held.provider() == "CPUExecutionProvider"
        assert "provider_unavailable" in caplog.text
        assert "MoonLander" in caplog.text
        held.close()

    def test_a_provider_that_is_built_in_but_did_not_start_is_said_so(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            providers_that_did_not_start(
                ["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]
            )

        assert "provider_unavailable" in caplog.text
        assert "CUDAExecutionProvider" in caplog.text

    def test_nothing_is_said_when_every_provider_started(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            providers_that_did_not_start(["CPUExecutionProvider"], ["CPUExecutionProvider"])

        assert caplog.text == ""


class TestWarmingUpBeforeItServes:
    def test_it_answers_the_same_whether_it_was_warmed_or_not(self, tmp_path: Path) -> None:
        cold = backend(model(tmp_path / "m.onnx"), warmup_requests=0)
        warm = backend(model(tmp_path / "m.onnx"), warmup_requests=3)
        image = an_image()

        one, other = cold.predict(image), warm.predict(image)
        cold.close()
        warm.close()

        assert (one.label, one.confidence) == (other.label, other.confidence)


class TestTheGpuLibraries:
    @pytest.mark.parametrize(
        "usable",
        [["CUDAExecutionProvider", "CPUExecutionProvider"], ["CUDAExecutionProvider"]],
    )
    def test_they_are_preloaded_when_a_gpu_provider_is_usable(
        self, usable: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[bool] = []
        monkeypatch.setattr("onnxruntime.preload_dlls", lambda **_: calls.append(True))

        gpu_libraries_loaded(usable)

        assert calls == [True]

    def test_they_are_not_preloaded_for_the_cpu_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[bool] = []
        monkeypatch.setattr("onnxruntime.preload_dlls", lambda **_: calls.append(True))

        gpu_libraries_loaded(["CPUExecutionProvider"])

        assert calls == []
