from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Draw = Callable[[int], NDArray[np.float64]]


def standard_normal(seed: int) -> Draw:
    generator = np.random.default_rng(seed)

    def draw(size: int) -> NDArray[np.float64]:
        return generator.standard_normal(size)

    return draw


@dataclass(frozen=True, slots=True)
class IncludeExcludeScores:
    include: float
    exclude: float

    @property
    def difference(self) -> float:
        return self.include - self.exclude


class SampledWeights:
    def __init__(self, include: NDArray[np.float64], exclude: NDArray[np.float64]) -> None:
        self._include = include
        self._exclude = exclude

    def scores(self, signals: Sequence[float]) -> IncludeExcludeScores:
        held = np.asarray(signals, dtype=np.float64)
        return IncludeExcludeScores(
            include=float(self._include @ held), exclude=float(self._exclude @ held)
        )


class _Head:
    def __init__(self, dimension: int, *, prior_precision: float) -> None:
        self._precision: NDArray[np.float64] = np.eye(dimension, dtype=np.float64) * (
            prior_precision
        )
        self._weighted: NDArray[np.float64] = np.zeros(dimension, dtype=np.float64)

    def mean(self) -> NDArray[np.float64]:
        return np.linalg.solve(self._precision, self._weighted).astype(np.float64)

    def sample(self, draw: Draw) -> NDArray[np.float64]:
        lower = np.linalg.cholesky(self._precision)
        shift = np.linalg.solve(lower.T, draw(len(self._weighted))).astype(np.float64)
        return self.mean() + shift

    def learn(self, signals: NDArray[np.float64], target: float, noise_variance: float) -> None:
        self._precision = self._precision + np.outer(signals, signals) / noise_variance
        self._weighted = self._weighted + signals * target / noise_variance


class Weights:
    def __init__(self, *, dimension: int, prior_precision: float, draw: Draw) -> None:
        self._include = _Head(dimension, prior_precision=prior_precision)
        self._exclude = _Head(dimension, prior_precision=prior_precision)
        self._draw = draw

    def sampled(self) -> SampledWeights:
        return SampledWeights(self._include.sample(self._draw), self._exclude.sample(self._draw))

    def expected(self, signals: Sequence[float]) -> float:
        return float(self._include.mean() @ np.asarray(signals, dtype=np.float64))

    def learn(
        self, signals: Sequence[float], contribution: float, *, noise_variance: float
    ) -> None:
        held = np.asarray(signals, dtype=np.float64)
        self._include.learn(held, contribution, noise_variance)
        self._exclude.learn(held, -contribution, noise_variance)
