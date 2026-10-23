from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from eemls_data_hub.image_store import ImageStore
from PIL import Image

from support.repository import ROOT

sys.path.insert(0, str(ROOT / "tools"))

from eemls_tools.datasets import center_crop, image_id_of
from prepare_dataset import PrepareError, blurred, gaussian_kernel

SIZE = 64


def picture(width: int = 200, height: int = 120) -> Image.Image:
    pixels = np.zeros((height, width, 3), np.uint8)
    pixels[:, ::4] = 255
    pixels[::4, :] = 200
    return Image.fromarray(pixels)


def written(root: Path, *, images: int = 2) -> Path:
    for condition in ("day", "night"):
        folder = root / "data" / "mfnet_224" / condition / "car"
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(images):
            picture().save(folder / f"{condition}-{index}.jpg", format="JPEG", quality=92)
    return root


def downloaded(root: Path, dataset: str, label: str, names: tuple[str, ...]) -> None:
    folder = root / "data" / dataset / label
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        picture().save(folder / f"{name}.jpg", format="JPEG", quality=92)


def rows_of(root: Path) -> list[dict[str, str]]:
    manifest = (root / "hub" / "manifest.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in manifest.split("\n") if line]


def source(conditions: list[str], contributes: list[str]) -> dict[str, object]:
    return {
        "citation": "c",
        "url": "u",
        "license": "l",
        "conditions": conditions,
        "contributes": contributes,
        "label_map": {name: name for name in contributes},
    }


def sources(root: Path, **changes: object) -> Path:
    body: dict[str, object] = {
        "dominant_classes": ["car"],
        "tail_classes": ["dog"],
        "conditions": ["day", "night"],
        "image_size": SIZE,
        "images_per_dominant_class": 10,
        "images_per_tail_class": 5,
        "train_fraction": 0.8,
        "blur": {"sigma": 1.6, "kernel": 7},
        "sources": {
            "open_images": source(["day"], ["car", "dog"]),
            "imagenet": source(["day"], ["car", "dog"]),
            "multispectral": source(["day", "night"], ["car"]),
        },
    }
    path = root / "sources.yaml"
    path.write_text(yaml.safe_dump(body | changes), encoding="utf-8")
    return path


def prepare(root: Path, *extra: str, **changes: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(ROOT / "tools" / "prepare_dataset.py"),
            "--sources",
            str(sources(root, **changes)),
            "--into",
            str(root / "hub"),
            "--open-images",
            str(root / "data" / "open_images"),
            "--multispectral",
            str(root / "data" / "mfnet_224"),
            "--imagenet",
            str(root / "data" / "imagenet"),
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )


class TestTheBlurUsesBothDeclaredNumbers:
    def test_the_kernel_adds_up_to_one(self) -> None:
        assert gaussian_kernel(1.6, 7).sum() == pytest.approx(1.0)

    def test_the_kernel_is_symmetric_about_its_centre(self) -> None:
        held = gaussian_kernel(1.6, 7)

        assert held == pytest.approx(held[::-1])

    def test_a_wider_kernel_holds_more_of_the_curve(self) -> None:
        narrow, wide = gaussian_kernel(1.6, 3), gaussian_kernel(1.6, 9)

        assert narrow[len(narrow) // 2] > wide[len(wide) // 2]

    def test_an_even_width_is_refused(self) -> None:
        with pytest.raises(PrepareError, match="odd"):
            gaussian_kernel(1.6, 6)

    def test_a_blurred_image_holds_less_detail_and_the_same_size(self) -> None:
        clear = center_crop(picture(), SIZE)

        held = blurred(clear, sigma=1.6, kernel=7)

        assert held.size == clear.size
        assert np.asarray(held, float).std() < np.asarray(clear, float).std()


class TestOneSizeForEveryModel:
    def test_an_image_is_centre_cropped_to_a_square_and_resized(self) -> None:
        held = center_crop(picture(200, 120), SIZE)

        assert held.size == (SIZE, SIZE)

    def test_a_tall_image_is_cropped_the_same_way(self) -> None:
        assert center_crop(picture(120, 200), SIZE).size == (SIZE, SIZE)

    def test_the_crop_is_the_middle_of_the_image(self) -> None:
        pixels = np.zeros((100, 300, 3), np.uint8)
        pixels[:, 100:200] = 255
        held = center_crop(Image.fromarray(pixels), SIZE)

        assert np.asarray(held).min() == 255


class TestWhatAnIdSays:
    def test_an_id_names_the_source_the_condition_and_the_copy(self) -> None:
        assert image_id_of("multispectral", "night", "blurred", "00003N") == (
            "multispectral-night-blurred-00003N"
        )

    def test_the_two_copies_of_one_image_are_two_ids(self) -> None:
        clear = image_id_of("open_images", "day", "clear", "abc")
        blurry = image_id_of("open_images", "day", "blurred", "abc")

        assert clear != blurry


class TestThePreparedStore:
    def test_it_writes_a_clear_and_a_blurred_copy_of_every_image(self, tmp_path: Path) -> None:
        written(tmp_path)

        answer = prepare(tmp_path)

        assert answer.returncode == 0, answer.stderr
        manifest = (tmp_path / "hub" / "manifest.jsonl").read_text(encoding="utf-8")
        rows = [json.loads(line) for line in manifest.split("\n") if line]
        assert len(rows) == 8
        assert {row["image_quality"] for row in rows} == {"clear", "blurred"}

    def test_the_store_opens_what_it_wrote(self, tmp_path: Path) -> None:
        written(tmp_path)
        prepare(tmp_path)

        store = ImageStore.opened(tmp_path / "hub", max_image_bytes=1_000_000)

        assert store.conditions() == ("day", "night")
        assert len(store.ids(label="car", condition="night", image_quality="clear")) == 2

    def test_a_night_image_is_stored_as_a_night_image(self, tmp_path: Path) -> None:
        written(tmp_path)
        prepare(tmp_path)
        store = ImageStore.opened(tmp_path / "hub", max_image_bytes=1_000_000)

        held = store.meta(store.ids(label="car", condition="night", image_quality="clear")[0])

        assert (held.condition, held.dataset) == ("night", "multispectral")

    def test_every_prepared_image_carries_the_declared_size(self, tmp_path: Path) -> None:
        written(tmp_path)
        prepare(tmp_path)

        for path in (tmp_path / "hub").rglob("*.jpg"):
            with Image.open(path) as held:
                assert held.size == (SIZE, SIZE), path

    def test_two_runs_leave_the_same_store_byte_for_byte(self, tmp_path: Path) -> None:
        written(tmp_path)
        prepare(tmp_path)
        first = {
            str(path.relative_to(tmp_path / "hub")): path.read_bytes()
            for path in sorted((tmp_path / "hub").rglob("*"))
            if path.is_file()
        }

        prepare(tmp_path)
        second = {
            str(path.relative_to(tmp_path / "hub")): path.read_bytes()
            for path in sorted((tmp_path / "hub").rglob("*"))
            if path.is_file()
        }

        assert first == second

    def test_a_root_with_nothing_downloaded_is_refused(self, tmp_path: Path) -> None:
        answer = prepare(tmp_path)

        assert answer.returncode == 1
        assert "nothing is downloaded" in answer.stderr

    def test_checking_writes_nothing(self, tmp_path: Path) -> None:
        written(tmp_path)

        answer = prepare(tmp_path, "--check")

        assert answer.returncode == 0
        assert not (tmp_path / "hub").exists()

    def test_it_names_each_class_the_sources_give_that_has_no_image(self, tmp_path: Path) -> None:
        written(tmp_path)
        given = {
            "open_images": source(["day"], ["car", "dog"]),
            "multispectral": source(["day", "night"], ["car"]),
        }

        answer = prepare(tmp_path, "--check", sources=given)

        assert "missing: day dog" in answer.stdout


class TestImageNetIsASource:
    def test_an_imagenet_image_is_a_day_image_of_its_class(self, tmp_path: Path) -> None:
        downloaded(tmp_path, "imagenet", "dog", ("n02106662_7",))

        answer = prepare(tmp_path)

        assert answer.returncode == 0, answer.stderr
        assert {
            (row["dataset"], row["condition"], row["label"], row["original_label"])
            for row in rows_of(tmp_path)
        } == {("imagenet", "day", "dog", "n02106662_7")}

    def test_an_imagenet_id_names_its_source(self, tmp_path: Path) -> None:
        downloaded(tmp_path, "imagenet", "dog", ("n02106662_7",))

        prepare(tmp_path)

        assert {row["image_id"] for row in rows_of(tmp_path)} == {
            "imagenet-day-clear-n02106662_7",
            "imagenet-day-blurred-n02106662_7",
        }

    def test_an_imagenet_image_is_cropped_to_the_declared_size(self, tmp_path: Path) -> None:
        downloaded(tmp_path, "imagenet", "dog", ("n02106662_7",))

        prepare(tmp_path)

        for path in (tmp_path / "hub").rglob("*.jpg"):
            with Image.open(path) as held:
                assert held.size == (SIZE, SIZE), path

    def test_a_class_from_both_sources_keeps_the_images_of_each(self, tmp_path: Path) -> None:
        downloaded(tmp_path, "imagenet", "car", ("n02958343_1",))
        downloaded(tmp_path, "open_images", "car", ("abc",))

        prepare(tmp_path)

        assert sorted(
            row["dataset"] for row in rows_of(tmp_path) if row["image_quality"] == "clear"
        ) == ["imagenet", "open_images"]

    def test_an_imagenet_folder_outside_our_classes_is_left_out(self, tmp_path: Path) -> None:
        downloaded(tmp_path, "imagenet", "dog", ("n02106662_7",))
        downloaded(tmp_path, "imagenet", "tram", ("n04335435_1",))

        prepare(tmp_path)

        assert {row["label"] for row in rows_of(tmp_path)} == {"dog"}

    def test_checking_counts_the_imagenet_images(self, tmp_path: Path) -> None:
        downloaded(tmp_path, "imagenet", "dog", ("n02106662_7", "n02106662_8"))

        answer = prepare(tmp_path, "--check")

        assert answer.stdout.split("\n")[1].split() == ["dog", "2", "0"]

    def test_a_class_imagenet_gives_with_no_image_is_named_missing(self, tmp_path: Path) -> None:
        downloaded(tmp_path, "imagenet", "dog", ("n02106662_7",))
        given = {"imagenet": source(["day"], ["car", "dog"])}

        answer = prepare(tmp_path, "--check", sources=given, conditions=["day"])

        assert "missing: day car" in answer.stdout


def test_the_blur_of_the_committed_declaration_is_the_one_the_tool_uses() -> None:
    declared = yaml.safe_load((ROOT / "config" / "datasets" / "sources.yaml").read_text("utf-8"))

    held = gaussian_kernel(float(declared["blur"]["sigma"]), int(declared["blur"]["kernel"]))

    assert len(held) == declared["blur"]["kernel"]
    assert held.sum() == pytest.approx(1.0)


def test_a_blurred_copy_is_not_the_clear_one() -> None:
    clear = center_crop(picture(), SIZE)
    held = io.BytesIO()
    blurred(clear, sigma=1.6, kernel=7).save(held, format="JPEG", quality=92)
    original = io.BytesIO()
    clear.save(original, format="JPEG", quality=92)

    assert held.getvalue() != original.getvalue()
