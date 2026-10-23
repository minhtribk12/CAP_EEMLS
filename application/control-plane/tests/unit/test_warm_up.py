from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from eemls.image_catalogue import Catalogue
from eemls.testing.trees import committed_tree
from eemls_control_plane.config import load_control_plane_config
from eemls_control_plane.warm_up import main, warm_up

TREE = Path(__file__).resolve().parents[4] / "config"


def recording(asked: list[tuple[str, str]]) -> httpx.MockTransport:
    def answer(request: httpx.Request) -> httpx.Response:
        asked.append((request.url.host, json.loads(request.content)["image_id"]))
        return httpx.Response(200, json={})

    return httpx.MockTransport(answer)


def test_every_service_of_the_pool_is_asked_about_every_image() -> None:
    asked: list[tuple[str, str]] = []
    config = load_control_plane_config(TREE)

    warm_up(config, ("img-1", "img-2"), transport=recording(asked))

    pool = set(config.models.inference_services.services)
    assert {host for host, _ in asked} == pool
    assert len(asked) == 2 * len(pool)


def test_every_service_is_asked_the_same_images_in_the_same_order() -> None:
    asked: list[tuple[str, str]] = []

    warm_up(load_control_plane_config(TREE), ("img-1", "img-2"), transport=recording(asked))

    by_service: dict[str, list[str]] = {}
    for host, image in asked:
        by_service.setdefault(host, []).append(image)
    assert {tuple(images) for images in by_service.values()} == {("img-1", "img-2")}


def test_it_counts_what_each_service_answered() -> None:
    def refusing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503 if request.url.host.endswith("bt3") else 200, json={})

    held = warm_up(
        load_control_plane_config(TREE), ("img-1",), transport=httpx.MockTransport(refusing)
    )

    assert held["infer-b1-pedestrian-bt3"] == 0
    assert held["infer-b7-general-p5820"] == 1


def test_it_warms_from_the_files_the_control_plane_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = committed_tree(tmp_path / "config")
    (tree / "services" / "data_hub.yaml").unlink()
    (tree / "services" / "consumer.yaml").unlink()
    monkeypatch.setattr("eemls_control_plane.warm_up.Catalogue.read", lambda _: Catalogue({}))
    monkeypatch.setattr("eemls_control_plane.warm_up.warm_up", lambda *_, **__: {})

    assert main(["--config", str(tree), "--images", "0"]) == 0


def test_a_service_that_does_not_answer_is_logged_by_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nothing is listening")

    held = warm_up(
        load_control_plane_config(TREE), ("img-1",), transport=httpx.MockTransport(unreachable)
    )

    assert set(held.values()) == {0}
    assert "error_type=warm_request_failed service=infer-b1-pedestrian-bt3" in caplog.text
