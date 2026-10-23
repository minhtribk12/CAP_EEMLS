from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from eemls.config.models import ModelEntry
from eemls_inference_service.backends.backend import BackendError
from eemls_inference_service.model_loading import LoadError, preprocessing_of, verified

BODY = b"a stand-in for an exported model"
DIGEST = hashlib.sha256(BODY).hexdigest()


def entry(**changes: object) -> ModelEntry:
    fields: dict[str, object] = {
        "hugging_face_id": "owner/b5",
        "base_model": "b5",
        "onnx_file": "model.onnx",
        "sha256": DIGEST,
        "input_height": 224,
        "input_width": 224,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "classes": ["pedestrian", "cyclist", "motorcycle", "car", "truck", "bus"],
        "memory_mb": 46.8,
        "variant_of": None,
    }
    return ModelEntry.model_validate(fields | changes)


def on_disk(root: Path, body: bytes = BODY, name: str = "b5") -> Path:
    path = root / name / "model.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


class TestWhereItLooks:
    def test_a_model_is_under_its_own_name(self, tmp_path: Path) -> None:
        on_disk(tmp_path)

        assert verified("b5", entry(), under=tmp_path) == (tmp_path / "b5" / "model.onnx")


class TestWhatItProves:
    def test_a_file_matching_its_checksum_is_accepted(self, tmp_path: Path) -> None:
        on_disk(tmp_path)

        assert verified("b5", entry(), under=tmp_path).name == "model.onnx"

    def test_a_file_that_is_not_there_is_refused_by_name(self, tmp_path: Path) -> None:
        with pytest.raises(LoadError, match="b5"):
            verified("b5", entry(), under=tmp_path)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file whatever its mode")
    def test_a_folder_it_may_not_open_is_refused_by_name(self, tmp_path: Path) -> None:
        folder = on_disk(tmp_path).parent
        folder.chmod(0o000)
        try:
            with pytest.raises(LoadError, match="b5: .* could not be read"):
                verified("b5", entry(), under=tmp_path)
        finally:
            folder.chmod(0o755)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file whatever its mode")
    def test_a_file_it_may_not_read_is_refused_by_name(self, tmp_path: Path) -> None:
        on_disk(tmp_path).chmod(0o000)

        with pytest.raises(LoadError, match="b5: .* could not be read"):
            verified("b5", entry(), under=tmp_path)

    def test_a_file_that_does_not_match_its_checksum_is_refused(self, tmp_path: Path) -> None:
        on_disk(tmp_path, body=b"something else entirely")

        with pytest.raises(LoadError, match="sha256|checksum|declares"):
            verified("b5", entry(), under=tmp_path)


class TestWhatItReadsFromTheDeclaration:
    def test_the_preprocessing_is_the_one_the_model_declares(self) -> None:
        held = preprocessing_of(entry(input_height=300, input_width=300))

        assert (held.input_height, held.input_width) == (300, 300)
        assert held.normalization_mean == (0.485, 0.456, 0.406)

    def test_a_backend_error_is_a_load_error_so_one_refusal_reaches_the_caller(self) -> None:
        assert issubclass(LoadError, BackendError)
