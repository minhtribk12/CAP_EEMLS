from __future__ import annotations

from pathlib import Path

import pytest

from eemls.testing.trees import TREE, NoConfigurationTreeError, tree_above


def test_the_tree_is_the_config_folder_of_the_repository() -> None:
    assert TREE.name == "config"
    assert (TREE.parent / "pyproject.toml").is_file()


def test_a_folder_with_no_tree_above_it_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(NoConfigurationTreeError, match=str(tmp_path)):
        tree_above(tmp_path)


def test_a_config_folder_outside_a_project_is_not_the_tree(tmp_path: Path) -> None:
    (tmp_path / "config" / "services").mkdir(parents=True)

    with pytest.raises(NoConfigurationTreeError):
        tree_above(tmp_path / "config")
