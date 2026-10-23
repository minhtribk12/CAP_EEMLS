from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from prepare_imagenet import main
from support.repository import ROOT


def picture() -> Image.Image:
    pixels = np.zeros((40, 60, 3), np.uint8)
    pixels[:, ::3] = 255
    return Image.fromarray(pixels)


def ilsvrc(root: Path, held: dict[str, int], *, form: str = "JPEG") -> Path:
    folder = root / "ilsvrc"
    for synset, count in held.items():
        (folder / synset).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            picture().save(folder / synset / f"{synset}_{index}.JPEG", format=form)
    return folder


def sources(root: Path, **changes: object) -> Path:
    body: dict[str, object] = {
        "dominant_classes": ["car"],
        "tail_classes": ["dog", "cat"],
        "conditions": ["day"],
        "image_size": 8,
        "images_per_dominant_class": 4,
        "images_per_tail_class": 2,
        "train_fraction": 0.8,
        "blur": {"sigma": 1.6, "kernel": 7},
        "sources": {
            "imagenet": {
                "citation": "c",
                "url": "u",
                "license": "l",
                "conditions": ["day"],
                "contributes": ["car", "dog"],
                "label_map": {"n1": "car", "n2": "car", "n3": "dog", "n4": "cat"},
            },
            "open_images": {
                "citation": "c",
                "url": "u",
                "license": "l",
                "conditions": ["day"],
                "contributes": ["cat"],
                "label_map": {"Cat": "cat"},
            },
        },
    }
    path = root / "sources.yaml"
    path.write_text(yaml.safe_dump(body | changes), encoding="utf-8")
    return path


def run(root: Path, source: Path, *extra: str) -> int:
    return main(
        [
            "--sources",
            str(sources(root)),
            "--imagenet",
            str(source),
            "--into",
            str(root / "imagenet"),
            *extra,
        ]
    )


def taken(root: Path, label: str) -> list[str]:
    return sorted(path.name for path in (root / "imagenet" / label).glob("*"))


class TestWhatItTakes:
    def test_an_image_is_written_into_its_class_named_by_its_file_id(self, tmp_path: Path) -> None:
        source = ilsvrc(tmp_path, {"n3": 1})

        assert run(tmp_path, source) == 0

        assert taken(tmp_path, "dog") == ["n3_0.jpg"]

    def test_the_synsets_of_one_class_are_pooled(self, tmp_path: Path) -> None:
        source = ilsvrc(tmp_path, {"n1": 1, "n2": 1})

        run(tmp_path, source)

        assert taken(tmp_path, "car") == ["n1_0.jpg", "n2_0.jpg"]

    def test_a_synset_outside_the_label_map_is_left_out(self, tmp_path: Path) -> None:
        source = ilsvrc(tmp_path, {"n3": 1, "n9": 1})

        run(tmp_path, source)

        assert sorted(path.name for path in (tmp_path / "imagenet").iterdir()) == ["dog"]

    def test_a_class_imagenet_does_not_contribute_is_left_out(self, tmp_path: Path) -> None:
        source = ilsvrc(tmp_path, {"n4": 1})

        run(tmp_path, source)

        assert not (tmp_path / "imagenet" / "cat").exists()

    def test_only_the_named_classes_are_taken(self, tmp_path: Path) -> None:
        source = ilsvrc(tmp_path, {"n1": 1, "n3": 1})

        run(tmp_path, source, "--classes", "dog")

        assert sorted(path.name for path in (tmp_path / "imagenet").iterdir()) == ["dog"]

    def test_an_image_that_is_not_a_jpeg_is_written_as_one(self, tmp_path: Path) -> None:
        source = ilsvrc(tmp_path, {"n3": 1}, form="PNG")

        run(tmp_path, source)

        with Image.open(tmp_path / "imagenet" / "dog" / "n3_0.jpg") as held:
            assert held.format == "JPEG"


class TestHowManyItTakes:
    def test_a_dominant_class_takes_the_dominant_count(self, tmp_path: Path) -> None:
        source = ilsvrc(tmp_path, {"n1": 5, "n2": 5})

        run(tmp_path, source)

        assert len(taken(tmp_path, "car")) == 4

    def test_a_tail_class_takes_the_tail_count(self, tmp_path: Path) -> None:
        source = ilsvrc(tmp_path, {"n3": 5})

        run(tmp_path, source)

        assert len(taken(tmp_path, "dog")) == 2

    def test_the_seed_chooses_which_images_are_taken(self, tmp_path: Path) -> None:
        source = ilsvrc(tmp_path, {"n1": 6, "n2": 6})
        picks = []
        for run_of, seed in (("a", "1"), ("b", "1"), ("c", "2")):
            (tmp_path / run_of).mkdir()
            run(tmp_path / run_of, source, "--seed", seed)
            picks.append(taken(tmp_path / run_of, "car"))

        assert picks[0] == picks[1]
        assert picks[0] != picks[2]

    def test_counting_writes_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = ilsvrc(tmp_path, {"n3": 3})

        assert run(tmp_path, source, "--count") == 0

        assert not (tmp_path / "imagenet").exists()
        assert capsys.readouterr().out.split("\n")[1].split() == ["dog", "2", "3", "2"]


class TestARerun:
    def test_an_image_already_there_is_kept(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = ilsvrc(tmp_path, {"n3": 1})
        run(tmp_path, source)
        kept = tmp_path / "imagenet" / "dog" / "n3_0.jpg"
        kept.write_bytes(b"kept")

        run(tmp_path, source)

        assert kept.read_bytes() == b"kept"
        assert "0 written, 1 already there" in capsys.readouterr().out

    def test_an_image_that_cannot_be_read_is_counted_as_unreadable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = ilsvrc(tmp_path, {"n3": 1})
        (source / "n3" / "n3_0.JPEG").write_bytes(b"not an image")

        run(tmp_path, source)

        assert "0 written, 0 already there, 1 unreadable" in capsys.readouterr().out


def test_a_folder_that_is_not_there_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(tmp_path, tmp_path / "nowhere") == 1
    assert "refused: no images under" in capsys.readouterr().err


def test_every_class_imagenet_contributes_has_a_synset_in_the_committed_map() -> None:
    declared = yaml.safe_load((ROOT / "config" / "datasets" / "sources.yaml").read_text("utf-8"))
    held = declared["sources"]["imagenet"]

    assert set(held["contributes"]) <= set(held["label_map"].values())
    assert all(len(one) == 9 and one[0] == "n" and one[1:].isdigit() for one in held["label_map"])
