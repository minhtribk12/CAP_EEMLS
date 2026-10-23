"""Build model variations: a frozen backbone with a linear head grafted on.

Usage:
    python3 tools/build_variations.py --probe        two variations, and what they reach
    python3 tools/build_variations.py --build        every variation of the recipe
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import yaml
from eemls_inference_service.backends.backend import Preprocessing
from numpy.typing import NDArray
from onnx import TensorProto, helper, numpy_helper

from eemls_tools.config_tree import ConfigTree, load_config
from eemls_tools.datasets import (
    TRAINED,
    ImageQuality,
    image_id_of,
    manifest_rows,
    pick_ids,
)
from eemls_tools.model_config import BaseModelEntry

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
LOCAL = CONFIG / "local" / "models.yaml"
HUB = ROOT / "data" / "hub"
BACKBONES = ROOT / "models" / "base"
INTO = ROOT / "models"
STEPS = 400
LEARNING_RATE = 0.5

Row = dict[str, Any]


@dataclass(frozen=True)
class Head:
    weights: NDArray[np.float64]
    bias: NDArray[np.float64]
    classes: tuple[str, ...]


def split_of(
    image_ids: Sequence[str], *, train_fraction: float, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    held = sorted(set(image_ids))
    scored = sorted(held, key=lambda one: _digest_of(f"split:{seed}:{one}"))
    cut = min(round(len(scored) * train_fraction), max(len(scored) - 1, 0))
    return tuple(sorted(scored[:cut])), tuple(sorted(scored[cut:]))


def split_by_condition(
    rows: Sequence[Row], *, train_fraction: float, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    groups: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        groups.setdefault((row["label"], row["condition"]), []).append(row["image_id"])
    train: list[str] = []
    test: list[str] = []
    for held in groups.values():
        one, other = split_of(held, train_fraction=train_fraction, seed=seed)
        train.extend(one)
        test.extend(other)
    return tuple(sorted(train)), tuple(sorted(test))


def fitted(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    classes: Sequence[str],
    seed: int,
    class_weights: Mapping[str, float] | None = None,
    steps: int = STEPS,
    learning_rate: float = LEARNING_RATE,
) -> Head:
    held = np.asarray(features, dtype=np.float64)
    middle = held.mean(axis=0)
    deviation = held.std(axis=0)
    deviation = np.where(deviation > 0.0, deviation, 1.0)
    standard = (held - middle) / deviation
    weighed = _weighed(labels, classes, class_weights)
    weights, bias = _descended(
        standard, labels, len(classes), weighed, seed=seed, steps=steps, rate=learning_rate
    )
    # Fold the standardisation into the weights.
    folded = weights / deviation[:, None]
    return Head(weights=folded, bias=bias - middle @ folded, classes=tuple(classes))


def scores_of(head: Head, features: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(features, dtype=np.float64) @ head.weights + head.bias


def accuracy_by_class(
    head: Head, features: NDArray[np.float64], labels: NDArray[np.int64]
) -> dict[str, float]:
    chosen = scores_of(head, features).argmax(axis=1)
    held: dict[str, float] = {}
    for index, name in enumerate(head.classes):
        mine = labels == index
        if not mine.any():
            continue
        held[name] = float((chosen[mine] == index).mean())
    return held


def grafted(model: bytes, head: Head) -> bytes:
    held = onnx.load_from_string(model)
    output = held.graph.output[0]
    weights = numpy_helper.from_array(head.weights.astype(np.float32), name="eemls_head_weights")
    bias = numpy_helper.from_array(head.bias.astype(np.float32), name="eemls_head_bias")
    held.graph.initializer.extend([weights, bias])
    held.graph.node.extend(
        [
            helper.make_node("Flatten", [output.name], ["eemls_head_input"], axis=1),
            helper.make_node(
                "MatMul", ["eemls_head_input", "eemls_head_weights"], ["eemls_head_scores"]
            ),
            helper.make_node("Add", ["eemls_head_scores", "eemls_head_bias"], ["eemls_classes"]),
        ]
    )
    named = helper.make_tensor_value_info(
        "eemls_classes", TensorProto.FLOAT, [1, len(head.classes)]
    )
    held.graph.output.remove(output)
    held.graph.output.extend([named])
    if held.ir_version < 4:
        held.graph.input.extend(
            [
                helper.make_tensor_value_info(one.name, one.data_type, list(one.dims))
                for one in (weights, bias)
            ]
        )
    onnx.checker.check_model(held)
    return bytes(held.SerializeToString())


def features_of(
    session: ort.InferenceSession, images: Sequence[bytes], prepared: Preprocessing
) -> NDArray[np.float64]:
    name = session.get_inputs()[0].name
    held = [
        np.asarray(session.run(None, {name: prepared.array_of(image)})[0]).reshape(-1)
        for image in images
    ]
    return np.vstack(held).astype(np.float64)


def _weighed(
    labels: NDArray[np.int64], classes: Sequence[str], class_weights: Mapping[str, float] | None
) -> NDArray[np.float64]:
    asked = dict(class_weights or {})
    per_class = np.asarray([asked.get(name, 1.0) for name in classes], dtype=np.float64)
    return per_class[labels]


def _descended(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    classes: int,
    weighed: NDArray[np.float64],
    *,
    seed: int,
    steps: int,
    rate: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    generator = np.random.default_rng(seed)
    weights = generator.standard_normal((features.shape[1], classes)) * 0.01
    bias = np.zeros(classes, dtype=np.float64)
    target = np.zeros((len(labels), classes), dtype=np.float64)
    target[np.arange(len(labels)), labels] = 1.0
    share = weighed[:, None] / weighed.sum()
    for _ in range(steps):
        held = _softmax(features @ weights + bias)
        error = (held - target) * share
        weights -= rate * (features.T @ error)
        bias -= rate * error.sum(axis=0)
    return weights, bias


def _softmax(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    held = np.exp(scores - scores.max(axis=1, keepdims=True))
    return np.asarray(held / held.sum(axis=1, keepdims=True))


def _digest_of(held: str) -> str:
    return hashlib.sha256(held.encode("utf-8")).hexdigest()


def labelled(
    rows: Sequence[Row], *, classes: Sequence[str], per_class: int, seed: int
) -> list[Row]:
    held: dict[str, list[Row]] = {name: [] for name in classes}
    for row in rows:
        if row["image_quality"] == ImageQuality.clear and row["label"] in held:
            held[row["label"]].append(row)
    chosen: list[Row] = []
    for name in classes:
        wanted = set(pick_ids([one["image_id"] for one in held[name]], count=per_class, seed=seed))
        chosen.extend(one for one in held[name] if one["image_id"] in wanted)
    return chosen


def _backbone(name: str, base: BaseModelEntry) -> tuple[ort.InferenceSession, Preprocessing]:
    prepared = Preprocessing(
        input_height=base.input_height,
        input_width=base.input_width,
        normalization_mean=base.normalization_mean,
        normalization_std=base.normalization_std,
    )
    session = ort.InferenceSession(
        str(BACKBONES / f"{name}.onnx"), providers=["CPUExecutionProvider"]
    )
    return session, prepared


def _probe(arguments: argparse.Namespace) -> int:
    config = load_config(CONFIG)
    classes = config.datasets.classes
    rows = labelled(
        manifest_rows(HUB), classes=classes, per_class=arguments.per_class, seed=arguments.seed
    )
    images = [(HUB / row["path"]).read_bytes() for row in rows]
    labels = _labels_of(rows, classes)
    train, _ = split_by_condition(
        rows, train_fraction=config.datasets.train_fraction, seed=arguments.seed
    )
    trains = np.asarray([row["image_id"] in set(train) for row in rows])
    print(f"{len(images)} images over {len(classes)} classes, {int(trains.sum())} of them training")
    print(f"{'backbone':24s} {'ms/image':>9s} {'general':>8s} {'special':>8s} {'spread':>7s}")
    for name in arguments.backbones:
        session, prepared = _backbone(name, config.models.variations.base_models[name])
        started = time.perf_counter()
        features = features_of(session, images, prepared)
        per_image = (time.perf_counter() - started) * 1000.0 / len(images)
        general = fitted(features[trains], labels[trains], classes=classes, seed=arguments.seed)
        special = fitted(
            features[trains],
            labels[trains],
            classes=classes,
            seed=arguments.seed,
            class_weights=dict.fromkeys(arguments.specialism, arguments.weight),
        )
        held = accuracy_by_class(general, features[~trains], labels[~trains])
        mine = accuracy_by_class(special, features[~trains], labels[~trains])
        on = [one for one in arguments.specialism if one in held]
        print(
            f"{name:24s} {per_image:9.1f} "
            f"{sum(held.values()) / len(held):8.3f} {sum(mine.values()) / len(mine):8.3f} "
            f"{sum(mine[one] - held[one] for one in on) / max(len(on), 1):7.3f}"
        )
        print(f"  {'class':16s} {'general':>8s} {'special':>8s}")
        for one in sorted(held):
            mark = " *" if one in arguments.specialism else "  "
            print(f"  {one:16s} {held[one]:8.3f} {mine.get(one, 0.0):8.3f}{mark}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_variations")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--backbones", nargs="*", default=["b3", "b6"])
    parser.add_argument("--specialism", nargs="*", default=["cyclist", "motorcycle"])
    parser.add_argument("--weight", type=float, default=8.0)
    parser.add_argument("--per-class", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(argv)
    if arguments.build:
        return _build()
    if arguments.probe:
        return _probe(arguments)
    parser.error("nothing to do; pass --build or --probe")


def blurred_twin(rows: Sequence[Row], manifest: Mapping[str, Row]) -> list[Row]:
    held = []
    for row in rows:
        twin = manifest.get(
            image_id_of(
                row["dataset"], row["condition"], ImageQuality.blurred, row["original_label"]
            )
        )
        if twin is not None:
            held.append(twin)
    return held


def write_trained_ids(path: Path, rows: Sequence[Row], manifest: Mapping[str, Row]) -> None:
    held = {row["image_id"] for row in rows} | {
        row["image_id"] for row in blurred_twin(rows, manifest)
    }
    path.write_text("".join(f"{one}\n" for one in sorted(held)), encoding="utf-8")


def recorded(
    name: str,
    base: BaseModelEntry,
    classes: Sequence[str],
    path: Path,
    digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    entry = {
        "base_model": name.rsplit("_", 1)[0],
        "input_height": base.input_height,
        "input_width": base.input_width,
        "normalization_mean": list(base.normalization_mean),
        "normalization_std": list(base.normalization_std),
        "classes": list(classes),
        "memory_mb": round(path.stat().st_size / 1_000_000, 1),
    }
    source = {
        "hugging_face_id": base.hugging_face_id,
        "onnx_file": path.name,
        "sha256": digest,
    }
    return entry, source


def write_sources(path: Path, models: Mapping[str, Any]) -> None:
    held = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
    body = held if isinstance(held, dict) else {}
    body["models"] = dict(models)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body, sort_keys=True), encoding="utf-8")


@dataclass(frozen=True)
class Split:
    trains: list[Row]
    tests: list[Row]
    blurs: list[Row]


@dataclass
class BuiltVariations:
    models: dict[str, Any]
    sources: dict[str, Any]
    variation_profiles: dict[str, Any]


def _split(config: ConfigTree) -> Split:
    recipe = config.models.variations
    every = manifest_rows(HUB)
    manifest = {row["image_id"]: row for row in every}
    rows = labelled(
        every,
        classes=config.datasets.classes,
        per_class=recipe.train_images_per_class,
        seed=recipe.seed,
    )
    train, _ = split_by_condition(
        rows, train_fraction=config.datasets.train_fraction, seed=recipe.seed
    )
    held = set(train)
    trains = [row for row in rows if row["image_id"] in held]
    tests = [row for row in rows if row["image_id"] not in held]
    blurs = blurred_twin(tests, manifest)
    print(f"{len(trains)} training images, {len(tests)} held back, {len(blurs)} blurred copies")
    write_trained_ids(HUB / TRAINED, trains, manifest)
    return Split(trains=trains, tests=tests, blurs=blurs)


def _variations_of(
    backbone: str, base: BaseModelEntry, split: Split, config: ConfigTree, into: BuiltVariations
) -> None:
    recipe = config.models.variations
    classes = config.datasets.classes
    session, prepared = _backbone(backbone, base)
    graph = (BACKBONES / f"{backbone}.onnx").read_bytes()
    started = time.perf_counter()
    sets = {"train": split.trains, "test": split.tests, "blur": split.blurs}
    features = {key: _features_of_rows(session, rows, prepared) for key, rows in sets.items()}
    print(f"{backbone}: features in {time.perf_counter() - started:.0f}s")
    labels = {key: _labels_of(rows, classes) for key, rows in sets.items()}
    specialism = recipe.specialisms[backbone]
    weight = recipe.specialist_class_weight
    for suffix, weights in (
        ("general", None),
        (specialism[0], dict.fromkeys(specialism, weight)),
    ):
        name = f"{backbone}_{suffix}"
        head = fitted(
            features["train"],
            labels["train"],
            classes=classes,
            seed=recipe.seed,
            class_weights=weights,
        )
        path = INTO / name / f"{name}.onnx"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(grafted(graph, head))
        into.models[name], into.sources[name] = recorded(
            name, base, classes, path, hashlib.sha256(path.read_bytes()).hexdigest()
        )
        into.models[name]["variant_of"] = backbone
        reached = _rounded(accuracy_by_class(head, features["test"], labels["test"]))
        fell = _rounded(accuracy_by_class(head, features["blur"], labels["blur"]))
        into.variation_profiles[name] = {
            "variant_of": backbone,
            "specialism": list(specialism) if weights else [],
            "class_weight": weight,
            "train_images_per_class": recipe.train_images_per_class,
            "seed": recipe.seed,
            "accuracy_by_class": reached,
            "accuracy_by_class_blurred": fell,
        }
        clear = sum(reached.values()) / len(reached)
        blurred = sum(fell.values()) / len(fell)
        print(f"  {name:34s} {clear:.3f} clear, {blurred:.3f} blurred")


def _build() -> int:
    config = load_config(CONFIG)
    split = _split(config)
    INTO.mkdir(parents=True, exist_ok=True)
    built = BuiltVariations(models={}, sources={}, variation_profiles={})
    for backbone, base in config.models.variations.base_models.items():
        _variations_of(backbone, base, split, config, built)
    (CONFIG / "models" / "model_repository.yaml").write_text(
        yaml.safe_dump({"models": built.models}, sort_keys=True), encoding="utf-8"
    )
    write_sources(LOCAL, built.sources)
    (CONFIG / "models" / "variation_profiles.yaml").write_text(
        yaml.safe_dump({"variations": built.variation_profiles}, sort_keys=True), encoding="utf-8"
    )
    print(f"{len(built.models)} variations built into {INTO}")
    return 0


def _features_of_rows(
    session: ort.InferenceSession, rows: Sequence[Row], prepared: Preprocessing
) -> NDArray[np.float64]:
    return features_of(session, [(HUB / row["path"]).read_bytes() for row in rows], prepared)


def _labels_of(rows: Sequence[Row], classes: Sequence[str]) -> NDArray[np.int64]:
    return np.asarray([classes.index(row["label"]) for row in rows], dtype=np.int64)


def _rounded(held: Mapping[str, float]) -> dict[str, float]:
    return {name: round(one, 3) for name, one in sorted(held.items())}


if __name__ == "__main__":
    sys.exit(main())
