from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

import fetch_dataset
from eemls_tools.datasets import DatasetsConfig


def sources(root: Path) -> Path:
    body = {
        "dominant_classes": ["car"],
        "tail_classes": [],
        "conditions": ["day"],
        "image_size": 8,
        "images_per_dominant_class": 2,
        "images_per_tail_class": 1,
        "train_fraction": 0.8,
        "blur": {"sigma": 1.6, "kernel": 7},
        "sources": {
            "open_images": {
                "citation": "c",
                "url": "u",
                "license": "l",
                "contributes": ["car"],
                "conditions": ["day"],
                "label_map": {"Car": "car"},
            }
        },
    }
    path = root / "sources.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def offered(monkeypatch: pytest.MonkeyPatch, ids: list[str]) -> None:
    def classes(_: DatasetsConfig) -> dict[str, str]:
        return {"/m/car": "car"}

    def found(_: object, __: Sequence[str]) -> dict[str, list[str]]:
        return {"car": ids}

    def refused(*_: object, **__: object) -> None:
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(fetch_dataset, "wanted_classes", classes)
    monkeypatch.setattr(fetch_dataset, "available", found)
    monkeypatch.setattr(urllib.request, "urlopen", refused)


def fetched(root: Path) -> int:
    return fetch_dataset.main(["--sources", str(sources(root)), "--into", str(root / "images")])


class TestFailedDownloads:
    def test_a_failed_download_is_counted_apart_and_fails_the_fetch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        offered(monkeypatch, ["train/held", "train/lost"])
        (tmp_path / "images" / "car").mkdir(parents=True)
        (tmp_path / "images" / "car" / "held.jpg").write_bytes(b"image")

        code = fetched(tmp_path)

        assert "0 downloaded, 1 already there, 1 failed" in capsys.readouterr().out
        assert code == 1


class TestImagesNotPicked:
    def test_an_image_not_picked_moves_out_of_its_class_folder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        offered(monkeypatch, ["train/held"])
        folder = tmp_path / "images" / "car"
        folder.mkdir(parents=True)
        (folder / "held.jpg").write_bytes(b"image")
        (folder / "old.jpg").write_bytes(b"old image")

        fetched(tmp_path)

        assert sorted(one.name for one in folder.iterdir()) == ["held.jpg"]
        moved = tmp_path / "images" / "unpicked" / "car" / "old.jpg"
        assert moved.read_bytes() == b"old image"
        assert "1 not picked, moved to" in capsys.readouterr().out
