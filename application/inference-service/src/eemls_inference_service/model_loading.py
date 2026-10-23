from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from eemls.config.models import ModelEntry
from eemls_inference_service.backends.backend import BackendError, Preprocessing
from eemls_inference_service.backends.onnx_runtime import OnnxBackend


class LoadError(BackendError): ...


def verified(name: str, entry: ModelEntry, *, under: Path) -> Path:
    if entry.onnx_file is None or entry.sha256 is None:
        raise LoadError(
            f"{name}: config/local/models.yaml gives no file and digest for it, "
            "so no file can be proved to be it"
        )
    path = under / name / entry.onnx_file
    try:
        if not path.is_file():
            raise LoadError(f"{name}: {path} is not there, so nothing can be served")
        found = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as refused:
        raise LoadError(f"{name}: {path} could not be read: {refused}") from refused
    if found != entry.sha256:
        raise LoadError(f"{name}: {path} is {found}, and the repository declares {entry.sha256}")
    return path


def preprocessing_of(entry: ModelEntry) -> Preprocessing:
    return Preprocessing(
        input_height=entry.input_height,
        input_width=entry.input_width,
        normalization_mean=entry.normalization_mean,
        normalization_std=entry.normalization_std,
    )


def load(
    name: str,
    entry: ModelEntry,
    *,
    under: Path,
    providers: Sequence[str],
    session_threads: int,
    warmup_requests: int,
) -> OnnxBackend:
    return OnnxBackend(
        path=verified(name, entry, under=under),
        preprocessing=preprocessing_of(entry),
        classes=entry.classes,
        providers=providers,
        session_threads=session_threads,
        warmup_requests=warmup_requests,
    )
