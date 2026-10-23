from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from eemls.config.models import ModelEntry, ModelRepositoryConfig
from eemls_tools.model_config import BaseModelEntry
from fetch_models import (
    Fetched,
    FetchError,
    declared_backbones,
    declared_models,
    fetch_backbone,
    fetch_one,
    url_of,
)

BODY = b"a tiny stand-in for an onnx file"
DIGEST = hashlib.sha256(BODY).hexdigest()


def fields(**changes: object) -> dict[str, object]:
    return {
        "hugging_face_id": "owner/b3",
        "base_model": "b3",
        "onnx_file": "model.onnx",
        "sha256": DIGEST,
        "input_height": 224,
        "input_width": 224,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "classes": ["car"],
        "memory_mb": 14.0,
        "variant_of": None,
        **changes,
    }


def entry(**changes: object) -> ModelEntry:
    return ModelEntry.model_validate(fields(**changes))


class Serving:
    def __init__(self, body: bytes = BODY) -> None:
        self.body = body
        self.calls = 0

    def __call__(self, url: str) -> bytes:
        self.calls += 1
        return self.body


def serves(body: bytes = BODY) -> Serving:
    return Serving(body)


class TestWhereItLooks:
    def test_the_url_names_the_repository_and_the_file(self) -> None:
        assert url_of(entry()) == ("https://huggingface.co/owner/b3/resolve/main/model.onnx")


class TestWhatItKeeps:
    def test_a_file_matching_its_checksum_is_kept(self, tmp_path: Path) -> None:
        download = serves()

        held = fetch_one("b3", entry(), into=tmp_path, download=download)

        assert held == Fetched.downloaded
        assert (tmp_path / "b3" / "model.onnx").read_bytes() == BODY
        assert download.calls == 1

    def test_a_file_already_there_and_right_is_not_downloaded_again(self, tmp_path: Path) -> None:
        download = serves()
        fetch_one("b3", entry(), into=tmp_path, download=download)

        again = fetch_one("b3", entry(), into=tmp_path, download=download)

        assert again == Fetched.already_there
        assert download.calls == 1


class TestWhatIsBuiltHereRatherThanFetched:
    def test_a_variation_is_not_downloaded(self, tmp_path: Path) -> None:
        def refuse(url: str) -> bytes:
            raise AssertionError(f"a variation must not be fetched, and this asked for {url}")

        held = fetch_one(
            "b5_general",
            entry(variant_of="b5"),
            into=tmp_path,
            download=refuse,
        )

        assert held is Fetched.built_here

    def test_a_model_that_is_nobody_s_variation_is_downloaded(self, tmp_path: Path) -> None:
        held = fetch_one("b5", entry(variant_of=None), into=tmp_path, download=lambda _: BODY)

        assert held is Fetched.downloaded


class TestWhatItRefuses:
    def test_a_file_whose_checksum_does_not_match_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(FetchError, match="b3"):
            fetch_one("b3", entry(), into=tmp_path, download=serves(b"something else"))

    def test_a_refused_file_is_not_left_behind(self, tmp_path: Path) -> None:
        with pytest.raises(FetchError):
            fetch_one("b3", entry(), into=tmp_path, download=serves(b"something else"))

        assert not (tmp_path / "b3" / "model.onnx").exists()

    def test_a_file_on_disk_that_no_longer_matches_is_downloaded_again(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "b3" / "model.onnx"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"half a file")
        download = serves()

        held = fetch_one("b3", entry(), into=tmp_path, download=download)

        assert held == Fetched.downloaded
        assert path.read_bytes() == BODY

    def test_a_download_that_fails_says_which_model_it_was(self, tmp_path: Path) -> None:
        def refuses(url: str) -> bytes:
            raise OSError("no route to host")

        with pytest.raises(FetchError, match="b3"):
            fetch_one("b3", entry(), into=tmp_path, download=refuses)


SOURCE_FIELDS = ("hugging_face_id", "onnx_file", "sha256")


def pool(**models: dict[str, object]) -> ModelRepositoryConfig:
    return ModelRepositoryConfig.model_validate({"models": models})


def committed(**changes: object) -> dict[str, object]:
    return {key: value for key, value in fields(**changes).items() if key not in SOURCE_FIELDS}


class TestWhereEachModelComesFrom:
    def test_a_sourced_model_of_the_pool_is_declared(self) -> None:
        assert declared_models(pool(m1=fields()))["m1"] == entry()

    def test_a_model_without_a_source_is_refused_by_name(self) -> None:
        with pytest.raises(FetchError, match=r"\['m2'\].*local/models.yaml"):
            declared_models(pool(m1=fields(), m2=committed()))

    def test_a_variation_needs_no_source_to_be_listed(self) -> None:
        held = declared_models(pool(v1=committed(variant_of="m1")))

        assert held["v1"].variant_of == "m1"


class TestTheBackbones:
    def base_models(self) -> dict[str, BaseModelEntry]:
        shape = {
            "family": "cnn",
            "input_height": 224,
            "input_width": 224,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "weights_mb": 14.0,
        }
        return {
            "b1": BaseModelEntry.model_validate(
                {
                    **shape,
                    "name": "real",
                    "hugging_face_id": "owner/repo",
                    "onnx_file": "model.onnx",
                    "sha256": DIGEST,
                }
            ),
            "b2": BaseModelEntry.model_validate(shape),
        }

    def test_every_backbone_with_a_source_is_declared(self) -> None:
        held = declared_backbones(self.base_models())

        assert set(held) == {"b1"}
        assert held["b1"].onnx_file == "model.onnx"

    def test_a_backbone_lands_where_tuning_reads_it(self, tmp_path: Path) -> None:
        held = declared_backbones(self.base_models())["b1"]

        fetched = fetch_backbone("b1", held, into=tmp_path, download=serves())

        assert fetched is Fetched.downloaded
        assert (tmp_path / "base" / "b1.onnx").read_bytes() == BODY

    def test_a_backbone_that_does_not_match_its_digest_is_refused(self, tmp_path: Path) -> None:
        held = declared_backbones(self.base_models())["b1"]

        with pytest.raises(FetchError, match="b1"):
            fetch_backbone("b1", held, into=tmp_path, download=serves(b"something else"))
        assert not (tmp_path / "base" / "b1.onnx").exists()
