from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eemls_tools.datasets import center_crop
from prepare_multispectral import main

WIDTH, HEIGHT = 64, 48
CAR, PERSON = 1, 2


def dataset(root: Path, masks: dict[str, np.ndarray], *, black: tuple[str, ...] = ()) -> Path:
    source = root / "multispectral"
    (source / "images").mkdir(parents=True)
    (source / "labels").mkdir()
    for name, mask in masks.items():
        bands = np.full((HEIGHT, WIDTH, 4), 120, dtype=np.uint8)
        Image.fromarray(bands, "RGBA").save(source / "images" / f"{name}.png")
        Image.fromarray(mask.astype(np.uint8)).save(source / "labels" / f"{name}.png")
    names = sorted(masks)
    (source / "train.txt").write_text("".join(f"{n}\n{n}_flip\n" for n in names[:-1]))
    (source / "val.txt").write_text("")
    (source / "test.txt").write_text(f"{names[-1]}\n")
    (source / "black_list.txt").write_text("".join(f"{n}\n" for n in black))
    return source


def mask(**covered: int) -> np.ndarray:
    held = np.zeros(HEIGHT * WIDTH, dtype=np.uint8)
    at = 0
    for label, count in covered.items():
        held[at : at + count] = {"car": CAR, "person": PERSON}[label]
        at += count
    return held.reshape(HEIGHT, WIDTH)


def built(root: Path, source: Path) -> list[dict[str, object]]:
    into = root / "mfnet_224"
    assert main(["--multispectral", str(source), "--into", str(into), "--size", "16"]) == 0
    index: list[dict[str, object]] = json.loads((into / "index.json").read_text(encoding="utf-8"))
    return index


def test_each_image_takes_the_class_covering_most_of_it(tmp_path: Path) -> None:
    source = dataset(tmp_path, {"00001D": mask(car=300, person=100), "00002N": mask(person=200)})

    index = built(tmp_path, source)

    assert [(one["name"], one["label"], one["condition"], one["split"]) for one in index] == [
        ("00001D", "car", "day", "train"),
        ("00002N", "pedestrian", "night", "test"),
    ]
    assert index[0]["share"] == round(300 / (WIDTH * HEIGHT), 4)
    assert (tmp_path / "mfnet_224" / "day" / "car" / "00001D.jpg").is_file()


def test_an_image_whose_class_covers_too_little_is_left_out(tmp_path: Path) -> None:
    source = dataset(tmp_path, {"00001D": mask(car=15), "00002D": mask(car=16)})

    assert [one["name"] for one in built(tmp_path, source)] == ["00002D"]


def test_an_image_listed_black_unlabelled_or_without_a_mask_is_left_out(tmp_path: Path) -> None:
    source = dataset(
        tmp_path,
        {"00001D": mask(car=300), "00002D": mask(), "00003D": mask(car=300)},
        black=("00003D",),
    )
    (source / "images" / "00004D.png").write_bytes((source / "images" / "00001D.png").read_bytes())

    assert [one["name"] for one in built(tmp_path, source)] == ["00001D"]


def test_the_crop_is_the_centre_square_at_the_size_asked() -> None:
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:, (WIDTH - HEIGHT) // 2 : (WIDTH + HEIGHT) // 2] = 255

    held = center_crop(Image.fromarray(image), 16)

    assert held.size == (16, 16)
    assert np.asarray(held).min() == 255


def test_the_sources_given_are_the_ones_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = dataset(tmp_path, {"00001D": mask(car=300)})
    sources = tmp_path / "sources.yaml"
    sources.write_text("dominant_classes: []\n", encoding="utf-8")

    assert main(["--sources", str(sources), "--multispectral", str(source)]) == 1
    assert f"refused: {sources}" in capsys.readouterr().err


def test_a_folder_without_images_is_refused_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--multispectral", str(tmp_path / "nowhere")]) == 1
    assert "refused: no images under" in capsys.readouterr().err
