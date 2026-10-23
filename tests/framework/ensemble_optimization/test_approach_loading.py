from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from eemls.config.datasets import DatasetClasses
from eemls.config.reading import ConfigError, Reader
from eemls.ensemble_optimization.approach_loading import (
    StartupError,
    approaches,
    load_approaches,
    policy_of,
)
from eemls.ensemble_optimization.testing.approaches import FAKE, installed
from eemls.testing.trees import TREE, committed_tree

ROOT = Path(__file__).resolve().parents[3]
FRAMEWORK = ROOT / "src" / "eemls"
CONTROL_PLANE_MODULES = (
    *sorted((FRAMEWORK / "orchestration").rglob("*.py")),
    *sorted((FRAMEWORK / "observation_service").rglob("*.py")),
    *(
        FRAMEWORK / "ensemble_optimization" / name
        for name in ("decision_loop.py", "approach_loading.py")
    ),
    FRAMEWORK / "evaluation_engine" / "consumer_evaluation.py",
)
CLASSES = Reader(TREE).build(DatasetClasses, "datasets/sources.yaml").classes
BUILT_IN = (
    "cap",
    "scale_reduction",
    "sneakpeek",
    "axiomvision",
    "cocktail",
    "frugalml",
    "keep_the_ensemble",
)


@pytest.fixture
def site(tmp_path: Path) -> Path:
    held = tmp_path / "site"
    held.mkdir()
    return held


def tree_with(tmp_path: Path, name: str, body: dict[str, object]) -> Path:
    target = committed_tree(tmp_path / "config")
    (target / "approaches" / f"{name}.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")
    return target


class TestWhichApproachesAreFound:
    def test_every_built_in_approach_is_registered(self) -> None:
        assert set(BUILT_IN) <= set(approaches())

    def test_an_installed_approach_is_found_by_its_name(
        self, site: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        installed(site, monkeypatch)

        assert approaches()["fake"].name == "fake"

    def test_a_policy_file_is_validated_through_the_settings_of_its_approach(
        self, tmp_path: Path, site: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        installed(site, monkeypatch)
        tree = tree_with(tmp_path, "fake", {"kind": "fake", "max_ensemble_size": 2, "patience": 4})

        held = load_approaches(Reader(tree))["fake"]

        assert type(held) is approaches()["fake"].Settings
        assert held.model_dump()["patience"] == 4

    def test_each_committed_policy_file_is_validated_through_its_approach(self) -> None:
        found = load_approaches(Reader(ROOT / "config"))

        for name, held in found.items():
            assert type(held) is approaches()[held.kind].Settings, name


class TestWhatIsRefusedAtStart:
    def test_two_approaches_of_one_name_are_refused_by_name(
        self, site: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        installed(site, monkeypatch, name="cap")

        with pytest.raises(ConfigError, match="approach cap is installed twice"):
            approaches()

    def test_an_approach_installed_under_another_name_is_refused_by_name(
        self, site: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = FAKE.replace('Approach(name="{name}"', 'Approach(name="other"')
        installed(site, monkeypatch, source=source)

        with pytest.raises(ConfigError, match="approach fake is installed under the name other"):
            approaches()

    def test_another_api_version_is_refused_by_name(
        self, site: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        installed(site, monkeypatch, api_version="API_VERSION + 1")

        with pytest.raises(ConfigError, match="approach fake states api_version"):
            approaches()

    def test_an_approach_that_cannot_be_imported_is_refused_by_name(
        self, site: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        installed(site, monkeypatch, source="import eemls.no_such_module\n")

        with pytest.raises(ConfigError, match="approach fake could not be loaded"):
            approaches()

    def test_an_approach_that_fails_while_importing_is_refused_by_name(
        self, site: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        installed(site, monkeypatch, source="raise RuntimeError('broken')\n")

        with pytest.raises(ConfigError, match="approach fake could not be loaded: broken"):
            approaches()

    def test_an_entry_point_that_loads_no_approach_is_refused_by_name(
        self, site: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        installed(site, monkeypatch, source="approach = 3\n")

        with pytest.raises(ConfigError, match="approach fake is not an Approach"):
            approaches()

    def test_settings_that_are_not_approach_settings_are_refused_by_name(
        self, site: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        installed(site, monkeypatch, source=FAKE.replace("Settings=FakeSettings", "Settings=dict"))

        with pytest.raises(ConfigError, match="approach fake states Settings that are not"):
            approaches()

    def test_finding_no_approach_at_all_says_to_install_them(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "eemls.ensemble_optimization.approach_loading.entry_points",
            lambda group: [],
        )

        with pytest.raises(ConfigError, match="run `poetry install`"):
            approaches()

    def test_bad_settings_are_refused_with_the_name_of_the_approach(
        self, tmp_path: Path, site: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        installed(site, monkeypatch)
        tree = tree_with(tmp_path, "slow", {"kind": "fake", "max_ensemble_size": 2, "rush": 1})

        with pytest.raises(ConfigError, match=r"approaches/slow.yaml: approach fake: .*rush"):
            load_approaches(Reader(tree))

    def test_an_unknown_kind_is_refused_by_name(self, tmp_path: Path) -> None:
        tree = tree_with(tmp_path, "slow", {"kind": "nobody", "max_ensemble_size": 2})

        with pytest.raises(ConfigError, match="kind 'nobody' is not one of"):
            load_approaches(Reader(tree))


class TestThePolicyEachFileBuilds:
    def test_a_policy_named_other_than_its_approach_is_refused_by_name(
        self, site: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        installed(
            site, monkeypatch, source=FAKE.replace('    name = "{name}"', '    name = "other"')
        )
        settings = approaches()["fake"].Settings.model_validate(
            {"kind": "fake", "max_ensemble_size": 2, "patience": 1}
        )

        with pytest.raises(StartupError, match="approach fake builds a policy named other"):
            policy_of(settings, classes=CLASSES)

    def test_the_committed_cap_file_builds_cap(self) -> None:
        held = policy_of(load_approaches(Reader(TREE))["cap"], classes=CLASSES)

        assert held.name == "cap"

    def test_every_approach_the_article_compares_has_an_implementation(self) -> None:
        found = load_approaches(Reader(TREE))

        held = {name: policy_of(settings, classes=CLASSES).name for name, settings in found.items()}

        assert held == {name: name for name in found}


def test_no_approach_is_named_in_the_control_plane_modules() -> None:
    names = [re.escape(one) for one in BUILT_IN]
    named = re.compile(
        rf"\b({'|'.join(names + [one.replace('_', '') for one in names])})\b", re.IGNORECASE
    )

    found = [
        f"{path.relative_to(ROOT)}:{line}: {match.group(0)}"
        for path in CONTROL_PLANE_MODULES
        for line, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        for match in named.finditer(text)
    ]

    assert not found
