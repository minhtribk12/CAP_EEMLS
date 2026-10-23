from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from eemls.testing.trees import committed_tree
from eemls_inference_service.model_loading import LoadError, verified
from eemls_tools.config_tree import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[2]
TREE = ROOT / "config"
DIGEST = "ab" * 32


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    return committed_tree(tmp_path / "config")


def local(tree: Path, body: dict[str, object]) -> None:
    path = tree / "local" / "models.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body), encoding="utf-8")


def a_model(tree: Path) -> str:
    return next(iter(load_config(tree).models.model_repository.models))


def a_backbone(tree: Path) -> str:
    return next(iter(load_config(tree).models.variations.base_models))


def given(**fields: str) -> dict[str, str]:
    return {"hugging_face_id": "owner/name", "onnx_file": "model.onnx", "sha256": DIGEST} | fields


class TestTheCommittedTree:
    def test_it_loads_without_the_local_file(self, tree: Path) -> None:
        entry = load_config(tree).models.model_repository.models[a_model(tree)]

        assert (entry.hugging_face_id, entry.onnx_file, entry.sha256) == (None, None, None)

    @pytest.mark.parametrize("name", ["model_repository.yaml", "variations.yaml"])
    @pytest.mark.parametrize("field", ["hugging_face_id", "onnx_file", "sha256"])
    def test_it_names_no_source_file_or_digest_of_any_model(self, name: str, field: str) -> None:
        assert f"{field}:" not in (TREE / "models" / name).read_text(encoding="utf-8")


class TestTheLocalFile:
    def test_it_gives_every_model_its_source_file_and_digest(self, tree: Path) -> None:
        name = a_model(tree)
        local(tree, {"models": {name: given()}})

        entry = load_config(tree).models.model_repository.models[name]

        assert (entry.hugging_face_id, entry.onnx_file, entry.sha256) == (
            "owner/name",
            "model.onnx",
            DIGEST,
        )

    def test_it_gives_every_backbone_its_source_file_and_digest(self, tree: Path) -> None:
        name = a_backbone(tree)
        local(tree, {"base_models": {name: given()}})

        assert load_config(tree).models.variations.base_models[name].sha256 == DIGEST

    def test_a_model_the_tree_does_not_hold_is_refused_by_name(self, tree: Path) -> None:
        local(tree, {"models": {"m_nowhere": given()}})

        with pytest.raises(ConfigError, match="local/models.yaml: .*m_nowhere"):
            load_config(tree)

    def test_a_field_it_does_not_know_is_refused(self, tree: Path) -> None:
        local(tree, {"models": {a_model(tree): given(weights="x")}})

        with pytest.raises(ConfigError, match="local/models.yaml: .*weights"):
            load_config(tree)

    def test_a_digest_that_is_not_one_is_refused(self, tree: Path) -> None:
        local(tree, {"models": {a_model(tree): given(sha256="short")}})

        with pytest.raises(ConfigError, match="local/models.yaml: .*sha256"):
            load_config(tree)


def test_the_file_may_be_read_from_where_eemls_model_sources_names(
    tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = a_model(tree)
    held = tmp_path / "models" / "models.yaml"
    held.parent.mkdir()
    held.write_text(yaml.safe_dump({"models": {name: given()}}), encoding="utf-8")
    monkeypatch.setenv("EEMLS_MODEL_SOURCES", str(held))

    entry = load_config(tree).models.model_repository.models[name]

    assert entry.sha256 == DIGEST


def test_the_named_file_is_read_in_place_of_the_tree_s_own(
    tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = a_model(tree)
    local(tree, {"models": {name: given(sha256="cd" * 32)}})
    held = tmp_path / "other.yaml"
    held.write_text(yaml.safe_dump({"models": {name: given()}}), encoding="utf-8")
    monkeypatch.setenv("EEMLS_MODEL_SOURCES", str(held))

    entry = load_config(tree).models.model_repository.models[name]

    assert entry.sha256 == DIGEST


def test_a_model_file_is_not_touched_until_the_local_file_names_it(
    tree: Path, tmp_path: Path
) -> None:
    name = a_model(tree)
    entry = load_config(tree).models.model_repository.models[name]

    with pytest.raises(LoadError, match=f"{name}: .*config/local/models.yaml"):
        verified(name, entry, under=tmp_path)


def test_the_local_file_may_name_the_real_backbone_behind_each_id(tree: Path) -> None:
    name = a_backbone(tree)
    local(tree, {"base_models": {name: given(name="a real name")}})

    assert load_config(tree).models.variations.base_models[name].name == "a real name"


def real_names() -> list[str]:
    held = TREE / "local" / "models.yaml"
    if not held.is_file():
        pytest.skip("config/local/models.yaml is not here, so there is no real name to look for")
    body = yaml.safe_load(held.read_text(encoding="utf-8"))
    return [entry["name"] for entry in body["base_models"].values() if entry.get("name")]


COMMITTED = (
    "application",
    "config",
    "deploy",
    "docs",
    "examples",
    "experiments",
    "src",
    "tests",
    "tools",
)


def test_no_committed_file_names_a_real_backbone() -> None:
    names = real_names()
    assert names
    stems = {re.match(r"[a-z]+", name.lower()).group(0) for name in names}  # type: ignore[union-attr]
    words = re.compile(rf"(?<![a-z])({'|'.join(sorted(stems))})(?![a-z])", flags=re.IGNORECASE)
    found = sorted(
        f"{path.relative_to(ROOT)}: {hit.group(0)}"
        for path in [
            *(path for path in ROOT.iterdir() if path.is_file()),
            *(path for top in COMMITTED for path in (ROOT / top).rglob("*")),
        ]
        if path.is_file()
        and path.relative_to(ROOT).parts[:2] not in {("config", "local"), ("examples", "local")}
        and "__pycache__" not in path.parts
        and path.suffix
        in {".py", ".yaml", ".yml", ".md", ".sh", ".hcl", ".json", ".jsonl", ".toml", ".txt", ""}
        for hit in words.finditer(path.read_text(encoding="utf-8", errors="ignore"))
    )

    assert found == []


class TestTheLocalImagesFile:
    def test_a_tree_without_it_names_no_registry(self, tree: Path) -> None:
        assert load_config(tree).images is None

    def test_it_names_the_registry_and_the_version_of_the_images(self, tree: Path) -> None:
        path = tree / "local" / "images.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"registry": "docker.io/someone", "version": "1.0"}))

        images = load_config(tree).images

        assert images is not None
        assert (images.registry, images.version) == ("docker.io/someone", "1.0")

    def test_an_empty_registry_is_refused_by_file(self, tree: Path) -> None:
        path = tree / "local" / "images.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"registry": "", "version": "latest"}))

        with pytest.raises(ConfigError, match="local/images.yaml: registry"):
            load_config(tree)


def test_the_committed_examples_load_as_the_local_files(tree: Path) -> None:
    (tree / "local").mkdir()
    for name in ("models.yaml", "images.yaml"):
        (tree / "local" / name).write_text(
            (ROOT / "examples" / "local" / name).read_text(encoding="utf-8"), encoding="utf-8"
        )

    config = load_config(tree)

    assert config.images is not None
    assert config.models.variations.base_models["b1"].hugging_face_id


def test_the_example_names_a_source_for_every_backbone_of_the_recipe() -> None:
    example = yaml.safe_load((ROOT / "examples" / "local" / "models.yaml").read_text())
    recipe = yaml.safe_load((ROOT / "config" / "models" / "variations.yaml").read_text())

    assert set(example["base_models"]) == set(recipe["base_models"])
