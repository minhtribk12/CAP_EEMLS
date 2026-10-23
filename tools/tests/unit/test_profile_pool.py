from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from profile_pool import ProfileError, answered_by, main, sampled, service_urls


def service(times_ms: dict[str, float], label: str = "car") -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        asked = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "predicted_label": label,
                "confidence": 0.9,
                "total_ms": times_ms[asked["image_id"]],
                "node": "pi-7",
                "instance_id": "infer-a-7f-x1",
            },
        )

    return httpx.MockTransport(handle)


class TestARunningService:
    def test_each_answer_carries_the_time_the_service_reports(self) -> None:
        held = answered_by(
            "http://infer-a:8082",
            [("img-1", "car"), ("img-2", "truck")],
            transport=service({"img-1": 120.0, "img-2": 140.0}),
        )

        assert held.answers == [("car", "car", 0.9, 120.0), ("car", "truck", 0.9, 140.0)]

    def test_the_node_and_copy_are_the_ones_that_answered(self) -> None:
        held = answered_by(
            "http://infer-a:8082", [("img-1", "car")], transport=service({"img-1": 5.0})
        )

        assert (held.node, held.instance_id) == ("pi-7", "infer-a-7f-x1")

    def test_a_service_that_refuses_is_named(self) -> None:
        refusing = httpx.MockTransport(lambda _: httpx.Response(503))

        with pytest.raises(ProfileError, match="http://infer-a:8082 answered 503"):
            answered_by("http://infer-a:8082", [("img-1", "car")], transport=refusing)


class TestWhereEachServiceIsMeasured:
    def test_a_service_is_placed_at_its_url(self) -> None:
        assert service_urls(["infer-a=http://127.0.0.1:18001"]) == {
            "infer-a": "http://127.0.0.1:18001"
        }

    def test_a_place_without_a_url_is_refused(self) -> None:
        with pytest.raises(ProfileError, match="infer-a: give it as <service>=<url>"):
            service_urls(["infer-a"])


def manifest(tmp_path: Path, counts: dict[str, int]) -> Path:
    rows = [
        {"image_id": f"{name}-{i:03d}", "label": name, "image_quality": "clear"}
        for name, count in counts.items()
        for i in range(count)
    ]
    rows.append({"image_id": "car-blurred", "label": "car", "image_quality": "blurred"})
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return tmp_path


class TestTheImagesMeasured:
    def test_the_total_is_spread_evenly_over_the_classes(self, tmp_path: Path) -> None:
        rows = sampled(
            manifest(tmp_path, {"car": 9, "bus": 9}), classes=["car", "bus"], images=6, seed=0
        )

        assert [row["label"] for row in rows].count("car") == 3
        assert len(rows) == 6

    def test_a_small_class_gives_all_it_has_and_the_others_fill_the_total(
        self, tmp_path: Path
    ) -> None:
        rows = sampled(
            manifest(tmp_path, {"guardrail": 1, "car": 9}),
            classes=["guardrail", "car"],
            images=5,
            seed=0,
        )

        assert [row["label"] for row in rows].count("guardrail") == 1
        assert len(rows) == 5

    def test_a_store_smaller_than_the_total_gives_every_clear_image(self, tmp_path: Path) -> None:
        rows = sampled(
            manifest(tmp_path, {"car": 2, "bus": 1}), classes=["car", "bus"], images=10, seed=0
        )

        assert len(rows) == 3
        assert all(row["image_quality"] == "clear" for row in rows)

    def test_a_class_from_two_datasets_is_sampled_from_both(self, tmp_path: Path) -> None:
        rows = [
            {"image_id": f"{source}-{i:03d}", "label": "car", "image_quality": "clear"}
            for source in ("bdd", "multispectral")
            for i in range(50)
        ]
        path = tmp_path / "manifest.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

        first = sampled(tmp_path, classes=["car"], images=20, seed=7)
        again = sampled(tmp_path, classes=["car"], images=20, seed=7)

        assert {row["image_id"].split("-")[0] for row in first} == {"bdd", "multispectral"}
        assert first == again

    def test_an_image_the_heads_trained_on_is_not_measured(self, tmp_path: Path) -> None:
        hub = manifest(tmp_path, {"car": 3})
        (hub / "trained.txt").write_text("car-000\ncar-002\n", encoding="utf-8")

        rows = sampled(hub, classes=["car"], images=10, seed=0)

        assert [row["image_id"] for row in rows] == ["car-001"]


class AskedError(Exception): ...


def test_ten_thousand_images_are_measured_unless_told_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: dict[str, int] = {}

    def capture(_: Path, *, classes: object, images: int, seed: int) -> list[object]:
        asked["images"] = images
        raise AskedError

    monkeypatch.setattr("profile_pool.sampled", capture)

    with pytest.raises(AskedError):
        main([])

    assert asked["images"] == 10000


def test_a_tree_that_does_not_load_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--config", str(tmp_path / "nowhere")]) == 1
    assert capsys.readouterr().err.startswith("refused: ")
