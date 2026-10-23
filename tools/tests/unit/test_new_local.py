from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eemls.testing.trees import committed_tree
from new_local import main


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    return committed_tree(tmp_path / "config")


def test_it_lists_every_model_and_backbone_of_the_tree_with_empty_values(tree: Path) -> None:
    assert main(["--config", str(tree)]) == 0

    held = yaml.safe_load((tree / "local" / "models.yaml").read_text(encoding="utf-8"))
    repository = yaml.safe_load((tree / "models" / "model_repository.yaml").read_text())
    recipe = yaml.safe_load((tree / "models" / "variations.yaml").read_text())
    assert set(held["models"]) == set(repository["models"])
    assert set(held["base_models"]) == set(recipe["base_models"])
    assert held["models"][next(iter(held["models"]))] == {
        "hugging_face_id": "",
        "onnx_file": "",
        "sha256": "",
    }
    assert set(held["base_models"][next(iter(held["base_models"]))]) == {
        "name",
        "hugging_face_id",
        "onnx_file",
        "sha256",
    }


def test_it_writes_the_image_registry_and_version_to_fill_in(tree: Path) -> None:
    main(["--config", str(tree)])

    held = yaml.safe_load((tree / "local" / "images.yaml").read_text(encoding="utf-8"))

    assert held == {"registry": "", "version": "latest"}


def test_a_local_file_already_there_is_never_overwritten(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tree / "local").mkdir()
    (tree / "local" / "models.yaml").write_text("models: {}\n", encoding="utf-8")

    assert main(["--config", str(tree)]) == 1
    assert (tree / "local" / "models.yaml").read_text(encoding="utf-8") == "models: {}\n"
    assert "models.yaml" in capsys.readouterr().err
