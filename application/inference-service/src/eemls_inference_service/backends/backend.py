from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from eemls.domain.errors import EemlsError
from eemls.domain.types import FrozenModel, PositiveCount, Ratio


class BackendError(EemlsError): ...


class Prediction(FrozenModel):
    label: str
    confidence: Ratio
    ranked: tuple[tuple[str, Ratio], ...]


class Preprocessing(FrozenModel):
    input_height: PositiveCount
    input_width: PositiveCount
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]

    def array_of(self, image: bytes) -> NDArray[np.float32]:
        """Image bytes as a float NCHW array."""
        try:
            opened = Image.open(io.BytesIO(image))
            opened.load()
        except (UnidentifiedImageError, OSError, ValueError) as refused:
            raise BackendError(
                f"those bytes are not an image this backend reads: {refused}"
            ) from refused
        held: Image.Image = opened.convert("RGB").resize(
            (self.input_width, self.input_height), Image.Resampling.BILINEAR
        )
        pixels: NDArray[np.float32] = np.asarray(held, dtype=np.float32)
        pixels /= 255.0
        pixels -= np.asarray(self.normalization_mean, dtype=np.float32)
        pixels /= np.asarray(self.normalization_std, dtype=np.float32)
        return np.ascontiguousarray(pixels.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)


def softmax(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    shifted = np.asarray(logits, dtype=np.float64) - float(np.max(logits))
    held: NDArray[np.float64] = np.exp(shifted)
    return np.asarray(held / held.sum(), dtype=np.float32)


def ranked(
    confidences: NDArray[np.float32] | NDArray[np.float64],
    classes: Sequence[str],
    *,
    top_k: int,
) -> tuple[tuple[str, float], ...]:
    if len(classes) != len(confidences):
        raise BackendError(
            f"the model answers {len(confidences)} classes and {len(classes)} are declared"
        )
    order = sorted(
        zip(classes, (float(one) for one in confidences), strict=True),
        key=lambda held: (-held[1], held[0]),
    )
    return tuple(order[:top_k])


@runtime_checkable
class Backend(Protocol):
    def predict(self, image: bytes) -> Prediction: ...

    def provider(self) -> str: ...

    def close(self) -> None: ...
