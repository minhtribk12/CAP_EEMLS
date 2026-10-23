from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
DOCS = ROOT / "docs"

SERVICES = ("control plane", "inference service", "data hub", "consumer", "broker")


def workflow() -> dict[Any, Any]:
    body: dict[Any, Any] = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return body


def triggers() -> dict[Any, Any]:
    body = workflow()
    events: dict[Any, Any] = body["on"] if "on" in body else body[True]
    return events


def steps() -> list[dict[Any, Any]]:
    jobs = workflow()["jobs"]
    return [step for job in jobs.values() for step in job["steps"]]


def test_the_workflow_exists_where_github_reads_it() -> None:
    assert WORKFLOW.is_file()


def test_the_workflow_runs_on_a_push_and_on_a_pull_request() -> None:
    events = triggers()

    assert "push" in events
    assert "pull_request" in events


@pytest.mark.parametrize("gate", ["make check", "make coverage"])
def test_the_workflow_runs_the_same_gate_as_a_developer(gate: str) -> None:
    commands = " ".join(str(step.get("run", "")) for step in steps())

    assert gate in commands


def test_the_workflow_scans_for_credentials() -> None:
    body = WORKFLOW.read_text(encoding="utf-8")

    assert "gitleaks" in body


def test_the_workflow_pins_every_action_it_uses() -> None:
    used = [str(step["uses"]) for step in steps() if "uses" in step]

    assert used
    for action in used:
        assert "@" in action, action
        assert not action.endswith(("@main", "@master")), action


@pytest.mark.parametrize("name", ["run.md"])
def test_every_document_exists_and_says_something(name: str) -> None:
    body = (DOCS / name).read_text(encoding="utf-8")

    assert body.startswith("# ")
    assert len(body.splitlines()) > 10


@pytest.mark.parametrize("service", SERVICES)
def test_the_readme_names_every_service(service: str) -> None:
    body = (ROOT / "README.md").read_text(encoding="utf-8").lower()

    assert service in body


@pytest.mark.parametrize("target", ["make up", "minikube", "k3s", "experiment.sh", "runs/"])
def test_the_run_guide_covers_each_way_to_run_and_where_results_go(target: str) -> None:
    assert target in (DOCS / "run.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "name", ["architecture.md", "deployment.md", "experiments.md", "artifact.md"]
)
def test_no_document_repeats_the_run_guide(name: str) -> None:
    assert not (DOCS / name).exists()
