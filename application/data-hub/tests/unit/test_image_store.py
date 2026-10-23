from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from eemls_data_hub.image_store import ImageStore, NoSuchImageError, StoreError

LABELS = ("pedestrian", "car")


def picture(colour: tuple[int, int, int] = (90, 90, 90)) -> bytes:
    held = io.BytesIO()
    Image.new("RGB", (32, 32), colour).save(held, format="JPEG")
    return held.getvalue()


def prepared(root: Path, *, rows: list[dict[str, object]] | None = None) -> Path:
    written = rows if rows is not None else _default_rows()
    for row in written:
        path = root / str(row["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(picture())
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in written), encoding="utf-8"
    )
    return root


def _default_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in LABELS:
        for condition in ("day", "night"):
            for quality in ("clear", "blurred"):
                for index in range(2):
                    image_id = f"{label}-{condition}-{quality}-{index}"
                    rows.append(
                        {
                            "image_id": image_id,
                            "label": label,
                            "dataset": "open_images",
                            "original_label": f"/m/{label}",
                            "condition": condition,
                            "image_quality": quality,
                            "path": f"{condition}/{quality}/{label}/{image_id}.jpg",
                        }
                    )
    return rows


def opened(root: Path) -> ImageStore:
    return ImageStore.opened(prepared(root), max_image_bytes=1_000_000)


class TestWhatItHolds:
    def test_it_holds_every_row_of_the_manifest(self, tmp_path: Path) -> None:
        assert len(opened(tmp_path)) == 16

    def test_it_answers_what_it_knows_about_one_image(self, tmp_path: Path) -> None:
        held = opened(tmp_path).meta("car-night-clear-0")

        assert held.label == "car"
        assert held.dataset == "open_images"
        assert held.original_label == "/m/car"
        assert held.condition == "night"
        assert held.image_quality == "clear"

    def test_it_answers_the_file_behind_one_image(self, tmp_path: Path) -> None:
        assert opened(tmp_path).bytes_of("car-day-clear-0")[:2] == b"\xff\xd8"

    def test_it_lists_one_class_in_one_condition_and_quality_in_a_stable_order(
        self, tmp_path: Path
    ) -> None:
        store = opened(tmp_path)

        assert store.ids(label="car", condition="day", image_quality="clear") == (
            "car-day-clear-0",
            "car-day-clear-1",
        )
        assert store.ids(label="car", condition="day", image_quality="blurred") == (
            "car-day-blurred-0",
            "car-day-blurred-1",
        )

    def test_a_night_image_is_never_listed_for_the_day(self, tmp_path: Path) -> None:
        store = opened(tmp_path)

        day = store.ids(label="car", condition="day", image_quality="clear")
        night = store.ids(label="car", condition="night", image_quality="clear")

        assert set(day) & set(night) == set()
        assert all("night" in one for one in night)

    def test_it_lists_the_conditions_it_holds(self, tmp_path: Path) -> None:
        assert opened(tmp_path).conditions() == ("day", "night")

    def test_it_lists_the_classes_it_holds(self, tmp_path: Path) -> None:
        assert opened(tmp_path).labels() == ("car", "pedestrian")

    def test_an_image_the_heads_trained_on_is_not_listed(self, tmp_path: Path) -> None:
        root = prepared(tmp_path)
        (root / "trained.txt").write_text("car-day-clear-0\ncar-day-blurred-0\n", encoding="utf-8")
        store = ImageStore.opened(root, max_image_bytes=1_000_000)

        assert store.ids(label="car", condition="day", image_quality="clear") == (
            "car-day-clear-1",
        )
        assert store.ids(label="car", condition="day", image_quality="blurred") == (
            "car-day-blurred-1",
        )

    def test_with_no_list_of_trained_images_every_image_is_listed(self, tmp_path: Path) -> None:
        assert len(opened(tmp_path).ids(label="car", condition="day", image_quality="clear")) == 2

    def test_a_class_it_does_not_hold_lists_nothing(self, tmp_path: Path) -> None:
        store = opened(tmp_path)

        assert store.ids(label="unicorn", condition="day", image_quality="clear") == ()
        assert store.ids(label="car", condition="twilight", image_quality="clear") == ()


class TestWhatItRefuses:
    def test_an_id_it_does_not_hold_is_a_typed_refusal(self, tmp_path: Path) -> None:
        with pytest.raises(NoSuchImageError, match="img-nowhere"):
            opened(tmp_path).meta("img-nowhere")

    def test_a_root_with_no_manifest_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(StoreError, match="manifest.jsonl"):
            ImageStore.opened(tmp_path, max_image_bytes=1_000)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file whatever its mode")
    def test_a_root_it_may_not_read_is_refused(self, tmp_path: Path) -> None:
        root = prepared(tmp_path / "hub")
        root.chmod(0o000)
        try:
            with pytest.raises(StoreError, match="could not be read"):
                ImageStore.opened(root, max_image_bytes=1_000_000)
        finally:
            root.chmod(0o755)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file whatever its mode")
    def test_a_manifest_it_may_not_read_is_refused(self, tmp_path: Path) -> None:
        root = prepared(tmp_path / "hub")
        (root / "manifest.jsonl").chmod(0o000)
        with pytest.raises(StoreError, match="could not be read"):
            ImageStore.opened(root, max_image_bytes=1_000_000)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file whatever its mode")
    def test_an_image_folder_it_may_not_open_is_refused_by_line(self, tmp_path: Path) -> None:
        root = prepared(tmp_path / "hub")
        folder = next(path for path in root.iterdir() if path.is_dir())
        folder.chmod(0o000)
        try:
            with pytest.raises(StoreError, match=r"manifest.jsonl:\d+: .* could not be read"):
                ImageStore.opened(root, max_image_bytes=1_000_000)
        finally:
            folder.chmod(0o755)

    def test_a_line_that_is_not_json_is_refused_by_line(self, tmp_path: Path) -> None:
        root = prepared(tmp_path / "hub")
        with (root / "manifest.jsonl").open("a", encoding="utf-8") as manifest:
            manifest.write("{not json\n")
        with pytest.raises(StoreError, match=r"manifest.jsonl:\d+: "):
            ImageStore.opened(root, max_image_bytes=1_000_000)

    def test_a_manifest_that_is_not_text_is_refused(self, tmp_path: Path) -> None:
        root = prepared(tmp_path / "hub")
        (root / "manifest.jsonl").write_bytes(b"\xff\xfe\x00broken")
        with pytest.raises(StoreError, match="could not be read"):
            ImageStore.opened(root, max_image_bytes=1_000_000)

    def test_a_row_whose_file_is_missing_is_refused_by_line(self, tmp_path: Path) -> None:
        prepared(tmp_path)
        (tmp_path / "day" / "clear" / "car" / "car-day-clear-0.jpg").unlink()

        with pytest.raises(StoreError, match="car-day-clear-0"):
            ImageStore.opened(tmp_path, max_image_bytes=1_000_000)

    def test_one_id_held_twice_is_refused(self, tmp_path: Path) -> None:
        rows = _default_rows()
        rows.append(dict(rows[0]))

        with pytest.raises(StoreError, match="held twice"):
            ImageStore.opened(prepared(tmp_path, rows=rows), max_image_bytes=1_000_000)

    def test_a_row_missing_what_it_must_say_is_refused_by_line(self, tmp_path: Path) -> None:
        rows = _default_rows()
        del rows[1]["label"]

        with pytest.raises(StoreError, match="manifest.jsonl:2"):
            ImageStore.opened(prepared(tmp_path, rows=rows), max_image_bytes=1_000_000)

    def test_a_quality_that_is_not_one_of_the_two_is_refused(self, tmp_path: Path) -> None:
        rows = _default_rows()
        rows[0]["image_quality"] = "upside_down"

        with pytest.raises(StoreError, match="manifest.jsonl:1"):
            ImageStore.opened(prepared(tmp_path, rows=rows), max_image_bytes=1_000_000)

    def test_a_file_larger_than_allowed_is_refused(self, tmp_path: Path) -> None:
        store = ImageStore.opened(prepared(tmp_path), max_image_bytes=4)

        with pytest.raises(StoreError, match="over the 4"):
            store.bytes_of("car-day-clear-0")
