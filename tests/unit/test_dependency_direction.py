from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path
from typing import Any

import pytest

from tests.support.imports import imported_names

ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK = ROOT / "src" / "eemls"
APPLICATIONS = sorted(path.parent.name for path in (ROOT / "application").glob("*/pyproject.toml"))
PROJECTS = (*(f"application/{folder}" for folder in APPLICATIONS), "tools")

CORE = frozenset({"domain", "config", "monitoring", "messaging", "image_catalogue", "interfaces"})
APPROACHES = frozenset(
    f"ensemble_optimization.approaches.{path.parent.name}"
    for path in (FRAMEWORK / "ensemble_optimization" / "approaches").glob("*/__init__.py")
)
# A framework package may import the packages named here, and no other package of the framework.
ALLOWED: dict[str, frozenset[str]] = {
    **dict.fromkeys(CORE, CORE),
    "domain": frozenset(),
    "testing": CORE,
    "evaluation_engine": CORE,
    "metrics_database": CORE,
    "observation_service": CORE | {"evaluation_engine", "metrics_database", "testing"},
    "orchestration": CORE
    | {"evaluation_engine", "metrics_database", "observation_service", "testing"},
    "ensemble_optimization": CORE
    | {"evaluation_engine", "metrics_database", "observation_service", "orchestration"},
    **dict.fromkeys(APPROACHES, CORE | {"ensemble_optimization", "evaluation_engine"}),
    "ensemble_optimization.testing": CORE
    | {"ensemble_optimization", "evaluation_engine", "testing"},
}
# A third-party package the framework imports in this one module only.
ONLY_IN: dict[str, str] = {
    "psycopg": "metrics_database/postgres.py",
    "psycopg_pool": "metrics_database/postgres.py",
    "numpy": "ensemble_optimization/approaches/cap",
    "PIL": "testing",
    "prometheus_client": "monitoring/declared_metrics.py",
}


def package_of(module: str) -> str:
    parts = module.split(".")
    nested = ".".join(parts[1:4])
    if nested in APPROACHES:
        return nested
    if ".".join(parts[1:3]) == "ensemble_optimization.testing":
        return "ensemble_optimization.testing"
    return parts[1]


def relative_imports(path: Path) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and node.level > 0
    ]


def framework_modules() -> list[Path]:
    return sorted(FRAMEWORK.rglob("*.py"))


def module_name(path: Path) -> str:
    parts = path.relative_to(FRAMEWORK.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def framework_packages(path: Path) -> set[str]:
    names = imported_names(path)
    return {package_of(name) for name in names if name.startswith("eemls.")} | ({"eemls"} & names)


def normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def distribution(requirement: str) -> str:
    return normalized(re.split(r"[\s(\[<>=!~;]", requirement.strip(), maxsplit=1)[0])


def manifest(project: str) -> dict[str, Any]:
    with (ROOT / project / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def declared(project: str, *, with_dev: bool) -> set[str]:
    read = manifest(project)
    names = {distribution(one) for one in read["project"]["dependencies"]}
    if with_dev:
        group = read.get("tool", {}).get("poetry", {}).get("group", {})
        names |= {normalized(one) for one in group.get("dev", {}).get("dependencies", {})}
    return names


def own_package(project: str) -> str:
    return "eemls_" + Path(project).name.replace("-", "_")


DISTRIBUTIONS = {
    module: {normalized(one) for one in names} for module, names in packages_distributions().items()
}


def undeclared(project: str, folder: str, *, with_dev: bool) -> list[str]:
    return undeclared_in(
        project, sorted((ROOT / project / folder).rglob("*.py")), with_dev=with_dev
    )


def undeclared_in(project: str, paths: list[Path], *, with_dev: bool) -> list[str]:
    allowed = declared(project, with_dev=with_dev)
    local = {path.stem for path in (ROOT / project).rglob("*.py")} | {
        path.name for path in (ROOT / project).rglob("*") if path.is_dir()
    }
    found: list[str] = []
    for path in paths:
        for name in sorted(imported_names(path)):
            top = name.split(".")[0]
            if top in sys.stdlib_module_names or top in local or top == own_package(project):
                continue
            if top == "eemls" or top.startswith("eemls_"):
                giver = {normalized(top)}
            else:
                giver = DISTRIBUTIONS.get(top, {f"no distribution gives {top}"})
            if not giver & allowed:
                found.append(f"{path.relative_to(ROOT)} imports {name}")
    return sorted(set(found))


@pytest.mark.parametrize("path", framework_modules(), ids=lambda path: module_name(path))
def test_every_framework_module_imports_only_what_the_direction_allows(path: Path) -> None:
    package = package_of(module_name(path)) if path.parent != FRAMEWORK else None
    if package is None:
        assert not framework_packages(path)
        return
    crossed = framework_packages(path) - ALLOWED[package] - {package}

    assert not crossed


def test_every_framework_package_is_named_in_the_direction() -> None:
    packages = {
        package_of(module_name(path)) for path in framework_modules() if path.parent != FRAMEWORK
    }

    assert packages == set(ALLOWED)


def test_the_framework_never_imports_an_application_or_the_tools() -> None:
    crossed = sorted(
        f"{path.relative_to(ROOT)} imports {name}"
        for path in framework_modules()
        for name in imported_names(path)
        if name.split(".")[0].startswith("eemls_")
    )

    assert crossed == []


@pytest.mark.parametrize("module", sorted(ONLY_IN))
def test_a_framework_extra_is_imported_in_one_place(module: str) -> None:
    importers = sorted(
        path.relative_to(FRAMEWORK).as_posix()
        for path in framework_modules()
        if any(name.split(".")[0] == module for name in imported_names(path))
    )

    assert importers
    assert all(one.startswith(ONLY_IN[module]) for one in importers), importers


@pytest.mark.parametrize("folder", APPLICATIONS)
def test_an_application_owns_one_package_named_after_its_folder(folder: str) -> None:
    packages = sorted(path.name for path in (ROOT / "application" / folder / "src").iterdir())

    assert [one for one in packages if not one.startswith(".")] == [
        own_package(f"application/{folder}")
    ]


@pytest.mark.parametrize("folder", APPLICATIONS)
@pytest.mark.parametrize("part", ["src", "tests"])
def test_an_application_imports_no_other_application_or_the_tools(folder: str, part: str) -> None:
    project = f"application/{folder}"
    crossed = sorted(
        f"{path.relative_to(ROOT)} imports {name}"
        for path in (ROOT / project / part).rglob("*.py")
        for name in imported_names(path)
        if name.split(".")[0].startswith("eemls_") and name.split(".")[0] != own_package(project)
    )

    assert crossed == []


@pytest.mark.parametrize("folder", APPLICATIONS)
def test_an_application_depends_on_the_framework_and_no_other_project(folder: str) -> None:
    project = f"application/{folder}"
    named = {one for one in declared(project, with_dev=True) if one.startswith("eemls")}

    assert named == {"eemls"}


@pytest.mark.parametrize("project", PROJECTS)
def test_a_project_imports_only_what_it_declares(project: str) -> None:
    assert undeclared(project, "src", with_dev=False) == []


def test_the_tools_scripts_import_only_what_the_tools_declare() -> None:
    assert undeclared_in("tools", sorted((ROOT / "tools").glob("*.py")), with_dev=False) == []


@pytest.mark.parametrize("tree", ["src", "application", "tools", "tests"])
def test_no_module_imports_relatively(tree: str) -> None:
    found = sorted(
        f"{path.relative_to(ROOT)}:{line}"
        for path in (ROOT / tree).rglob("*.py")
        for line in relative_imports(path)
    )

    assert found == []


def test_a_relative_import_is_found(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("from .types import FrozenModel\n", encoding="utf-8")

    assert relative_imports(module) == [1]


def test_a_bare_framework_import_crosses_the_direction(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("import eemls\n", encoding="utf-8")

    assert framework_packages(module) == {"eemls"}


@pytest.mark.parametrize("project", PROJECTS)
def test_the_tests_of_a_project_import_only_its_dependencies_and_dev_dependencies(
    project: str,
) -> None:
    assert undeclared(project, "tests", with_dev=True) == []


def test_the_control_plane_never_names_an_approach() -> None:
    named = sorted(
        f"{path.relative_to(ROOT)} imports {name}"
        for path in (ROOT / "application/control-plane/src").rglob("*.py")
        for name in imported_names(path)
        if name.startswith("eemls.ensemble_optimization.approaches")
    )

    assert named == []


def test_the_prometheus_client_is_imported_by_no_application() -> None:
    importers = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "application").glob("*/src/**/*.py")
        if "prometheus_client" in path.read_text(encoding="utf-8")
    )

    assert importers == []


@pytest.mark.parametrize("service", ("consumer", "inference_service", "data_hub"))
def test_starting_a_service_loads_no_metrics_database_code(service: str) -> None:
    probe = (
        "import sys; "
        f"import eemls_{service}.main; "
        "print(sorted(name for name in sys.modules if name.startswith('eemls.metrics_database')))"
    )
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    assert done.stdout.strip() == "[]"
