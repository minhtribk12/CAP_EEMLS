from __future__ import annotations

import itertools
import json
import os
import subprocess
from pathlib import Path

import pytest

from check_device_answers import SERVE, TOLERANCE, compare_answers, spread_by_label, store_of


def row(image_id: str, label: str) -> dict[str, str]:
    return {"image_id": image_id, "label": label, "path": f"day/clear/{label}/{image_id}.jpg"}


def answer(label: str, confidence: float, provider: str = "CPUExecutionProvider") -> str:
    return json.dumps(
        {"predicted_label": label, "confidence": confidence, "execution_provider": provider}
    )


class TestTheImagesItSends:
    def test_it_draws_the_same_images_every_time_spread_over_the_labels(self) -> None:
        rows = [row(f"i{n}", label) for n in range(30) for label in ("car", "bus")]

        once = spread_by_label(rows, 4)

        assert once == spread_by_label(rows, 4)
        assert sorted(one["label"] for one in once) == ["bus", "bus", "car", "car"]

    def test_the_store_holds_the_drawn_images_and_their_manifest(self, tmp_path: Path) -> None:
        source = tmp_path / "hub"
        chosen = [row("i1", "car")]
        (source / "day/clear/car").mkdir(parents=True)
        (source / "day/clear/car/i1.jpg").write_bytes(b"jpg")

        store = store_of(chosen, source, tmp_path / "store")

        assert (store / "day/clear/car/i1.jpg").read_bytes() == b"jpg"
        assert [
            json.loads(line) for line in (store / "manifest.jsonl").read_text().splitlines()
        ] == chosen


class TestHowAnswersAreCompared:
    def test_the_same_label_and_a_close_confidence_agree(self) -> None:
        [held] = compare_answers([answer("car", 0.9)], [answer("car", 0.9 + TOLERANCE / 2)])

        assert held.same

    def test_another_label_disagrees(self) -> None:
        [held] = compare_answers([answer("car", 0.9)], [answer("bus", 0.9)])

        assert not held.same

    def test_a_confidence_off_by_more_than_the_tolerance_disagrees(self) -> None:
        [held] = compare_answers([answer("car", 0.9)], [answer("car", 0.9 + TOLERANCE * 2)])

        assert not held.same

    def test_it_records_the_provider_the_device_ran_on(self) -> None:
        [held] = compare_answers(
            [answer("car", 0.9)], [answer("car", 0.9, "CUDAExecutionProvider")]
        )

        assert held.provider == "CUDAExecutionProvider"

    def test_a_device_that_answered_fewer_images_is_refused(self) -> None:
        with pytest.raises(ValueError, match="answered 1 of 2 images"):
            compare_answers([answer("car", 0.9), answer("bus", 0.8)], [answer("car", 0.9)])


SERVED = ["inference", "hub", "infer-a", "/store", "/models", "/here/models.yaml", "/here/config"]


def started_containers(tmp_path: Path) -> list[list[str]]:
    for tool in ("docker", "curl"):
        (tmp_path / tool).write_text(
            f'#!/bin/sh\nprintf "%s\\n" "$@" >> "{tmp_path}/{tool}.args"\n', encoding="utf-8"
        )
        (tmp_path / tool).chmod(0o755)

    subprocess.run(
        ["sh", "-s", "--", *SERVED],
        input=SERVE,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=True,
    )

    docker = (tmp_path / "docker.args").read_text(encoding="utf-8").splitlines()
    starts = [at for at, one in enumerate(docker) if one == "run"] + [len(docker)]
    return [docker[a:b] for a, b in itertools.pairwise(starts)]


def test_the_service_reads_the_model_sources_it_is_given(tmp_path: Path) -> None:
    [_, inference] = started_containers(tmp_path)

    assert "/here/models.yaml:/opt/eemls/model_sources.yaml:ro" in inference
    assert "EEMLS_MODEL_SOURCES=/opt/eemls/model_sources.yaml" in inference


def test_every_container_mounts_the_config_it_is_given_read_only(tmp_path: Path) -> None:
    started = started_containers(tmp_path)

    assert len(started) == 2
    for one in started:
        assert "/here/config:/opt/eemls/config:ro" in one, one
