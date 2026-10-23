from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort

from eemls.config.services import PROVIDERS
from eemls_inference_service.backends.backend import (
    BackendError,
    Prediction,
    Preprocessing,
    ranked,
    softmax,
)

logger = logging.getLogger(__name__)

TOP_K = 3


class OnnxBackend:
    def __init__(
        self,
        *,
        path: Path,
        preprocessing: Preprocessing,
        classes: Sequence[str],
        providers: Sequence[str],
        session_threads: int,
        warmup_requests: int,
        top_k: int = TOP_K,
    ) -> None:
        self._preprocessing = preprocessing
        self._classes = tuple(classes)
        self._top_k = top_k
        usable = self._narrowed(providers)
        options = ort.SessionOptions()
        options.intra_op_num_threads = session_threads
        options.log_severity_level = 3
        try:
            gpu_libraries_loaded(usable)
            self._session = ort.InferenceSession(str(path), sess_options=options, providers=usable)
        except Exception as refused:
            raise BackendError(f"{path.name} did not load: {refused}") from refused
        providers_that_did_not_start(usable, self._session.get_providers())
        self._input = self._session.get_inputs()[0].name
        self._check_the_model_is_the_one_declared(path)
        self._warm_up(warmup_requests)

    @staticmethod
    def _narrowed(providers: Sequence[str]) -> list[str]:
        available = set(ort.get_available_providers())
        usable = [one for one in providers if one in available]
        skipped = [one for one in providers if one not in available]
        if skipped:
            logger.warning(
                f"error_type=provider_unavailable skipped={skipped} "
                f"recovery_action=serving_on={usable}"
            )
        if not usable:
            raise BackendError(
                f"none of {list(providers)} is available here, only {sorted(available)}"
            )
        return usable

    def predict(self, image: bytes) -> Prediction:
        array = self._preprocessing.array_of(image)
        logits = self._session.run(None, {self._input: array})[0][0]
        order = ranked(softmax(np.asarray(logits)), self._classes, top_k=self._top_k)
        return Prediction(label=order[0][0], confidence=order[0][1], ranked=order)

    def provider(self) -> str:
        return str(self._session.get_providers()[0])

    def close(self) -> None:
        del self._session

    def _check_the_model_is_the_one_declared(self, path: Path) -> None:
        shape = self._session.get_inputs()[0].shape
        wanted = (self._preprocessing.input_height, self._preprocessing.input_width)
        found = tuple(one for one in shape[-2:] if isinstance(one, int))
        if found and found != wanted:
            raise BackendError(f"{path.name} takes {found} and the declaration says {wanted}")
        width = self._session.get_outputs()[0].shape[-1]
        if isinstance(width, int) and width != len(self._classes):
            raise BackendError(
                f"{path.name} answers {width} classes and {len(self._classes)} are declared"
            )

    def _warm_up(self, requests: int) -> None:
        blank = np.zeros(
            (1, 3, self._preprocessing.input_height, self._preprocessing.input_width),
            dtype=np.float32,
        )
        for _ in range(requests):
            self._session.run(None, {self._input: blank})


GPU_PROVIDERS = frozenset(PROVIDERS) - {"CPUExecutionProvider"}


def gpu_libraries_loaded(usable: Sequence[str]) -> None:
    if GPU_PROVIDERS.intersection(usable):
        ort.preload_dlls()


def providers_that_did_not_start(usable: Sequence[str], started: Sequence[str]) -> None:
    stopped = [one for one in usable if one not in started]
    if stopped:
        logger.warning(
            f"error_type=provider_unavailable skipped={stopped} "
            f"recovery_action=serving_on={list(started)}"
        )
