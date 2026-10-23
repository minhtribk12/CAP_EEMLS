from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from numpy.typing import NDArray
from onnx import helper

from build_variations import (
    Head,
    accuracy_by_class,
    fitted,
    grafted,
    recorded,
    scores_of,
    split_by_condition,
    split_of,
    write_sources,
    write_trained_ids,
)
from eemls_tools.datasets import pick_ids
from eemls_tools.model_config import BaseModelEntry

CLASSES = ("car", "bus", "cyclist")


def separable(per_class: int = 40) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    held = []
    labels = []
    for index in range(len(CLASSES)):
        middle = np.zeros((per_class, 6), dtype=np.float64)
        middle[:, index] = 8.0
        middle[:, index + 3] = np.linspace(-1.0, 1.0, per_class)
        held.append(middle)
        labels.extend([index] * per_class)
    return np.vstack(held), np.asarray(labels, dtype=np.int64)


class TestFittingAHead:
    def test_it_separates_what_a_line_can_separate(self) -> None:
        features, labels = separable()

        held = fitted(features, labels, classes=CLASSES, seed=0)

        chosen = scores_of(held, features).argmax(axis=1)
        assert (chosen == labels).mean() == pytest.approx(1.0)

    def test_the_head_answers_one_score_per_class(self) -> None:
        features, labels = separable()

        held = fitted(features, labels, classes=CLASSES, seed=0)

        assert held.classes == CLASSES
        assert scores_of(held, features).shape == (len(labels), len(CLASSES))

    def test_the_same_data_and_seed_give_the_same_head(self) -> None:
        features, labels = separable()

        one = fitted(features, labels, classes=CLASSES, seed=0)
        other = fitted(features, labels, classes=CLASSES, seed=0)

        assert np.array_equal(one.weights, other.weights)
        assert np.array_equal(one.bias, other.bias)

    def test_a_feature_that_never_varies_does_not_break_the_fitting(self) -> None:
        features, labels = separable()
        features[:, 5] = 3.0

        held = fitted(features, labels, classes=CLASSES, seed=0)

        assert np.isfinite(held.weights).all()
        assert np.isfinite(held.bias).all()

    def test_the_head_is_given_the_raw_features_and_folds_the_rest_in(self) -> None:
        features, labels = separable()
        middle = features.mean(axis=0)
        deviation = np.where(features.std(axis=0) > 0.0, features.std(axis=0), 1.0)
        standard = (features - middle) / deviation

        raw = fitted(features, labels, classes=CLASSES, seed=0)
        already = fitted(standard, labels, classes=CLASSES, seed=0)

        assert scores_of(raw, features) == pytest.approx(scores_of(already, standard), abs=1e-9)

    def test_weighing_one_class_up_makes_it_the_one_answered(self) -> None:
        features, labels = separable(per_class=40)
        features[:, :] = 0.0

        held = fitted(features, labels, classes=CLASSES, seed=0, class_weights={"bus": 50.0})

        assert held.classes[int(scores_of(held, features)[0].argmax())] == "bus"


class TestWhatItReached:
    def test_the_accuracy_is_counted_for_each_class_on_its_own(self) -> None:
        features, labels = separable()
        held = fitted(features, labels, classes=CLASSES, seed=0)

        reached = accuracy_by_class(held, features, labels)

        assert set(reached) == set(CLASSES)
        assert all(one == pytest.approx(1.0) for one in reached.values())

    def test_a_class_with_no_example_is_not_reported(self) -> None:
        features, labels = separable()
        kept = labels != 2
        held = fitted(features, labels, classes=CLASSES, seed=0)

        reached = accuracy_by_class(held, features[kept], labels[kept])

        assert set(reached) == {"car", "bus"}

    def test_a_head_that_answers_one_class_is_right_only_about_that_class(self) -> None:
        features, labels = separable()
        head = Head(
            weights=np.zeros((6, 3), dtype=np.float64),
            bias=np.asarray([0.0, 5.0, 0.0], dtype=np.float64),
            classes=CLASSES,
        )

        reached = accuracy_by_class(head, features, labels)

        assert reached["bus"] == pytest.approx(1.0)
        assert reached["car"] == pytest.approx(0.0)


class TestHowTheImagesAreSplit:
    def test_the_share_asked_for_is_the_share_that_trains(self) -> None:
        held = [f"image-{index}" for index in range(100)]

        train, test = split_of(held, train_fraction=0.8, seed=0)

        assert (len(train), len(test)) == (80, 20)

    def test_no_image_is_in_both_halves(self) -> None:
        held = [f"image-{index}" for index in range(100)]

        train, test = split_of(held, train_fraction=0.8, seed=0)

        assert set(train).isdisjoint(test)
        assert set(train) | set(test) == set(held)

    def test_the_same_seed_splits_the_same_way(self) -> None:
        held = [f"image-{index}" for index in range(100)]

        assert split_of(held, train_fraction=0.8, seed=0) == split_of(
            held, train_fraction=0.8, seed=0
        )

    def test_another_seed_splits_another_way(self) -> None:
        held = [f"image-{index}" for index in range(100)]

        one, _ = split_of(held, train_fraction=0.8, seed=0)
        other, _ = split_of(held, train_fraction=0.8, seed=1)

        assert one != other

    def test_the_order_the_images_arrive_in_does_not_change_the_split(self) -> None:
        held = [f"image-{index}" for index in range(100)]

        one, _ = split_of(held, train_fraction=0.8, seed=0)
        other, _ = split_of(list(reversed(held)), train_fraction=0.8, seed=0)

        assert set(one) == set(other)


class TestSplittingEachClassByCondition:
    def rows(self, label: str, condition: str, count: int) -> list[dict[str, str]]:
        return [
            {"image_id": f"{label}-{condition}-{index}", "label": label, "condition": condition}
            for index in range(count)
        ]

    def test_a_class_with_three_night_images_keeps_one_out_of_training(self) -> None:
        rows = self.rows("guardrail", "day", 50) + self.rows("guardrail", "night", 3)

        train, test = split_by_condition(rows, train_fraction=0.8, seed=0)

        assert len([one for one in test if "-night-" in one]) >= 1
        assert len([one for one in train if "-night-" in one]) == 2

    def test_a_condition_with_one_image_keeps_it_out_of_training(self) -> None:
        rows = self.rows("guardrail", "day", 10) + self.rows("guardrail", "night", 1)

        train, test = split_by_condition(rows, train_fraction=0.8, seed=0)

        assert "guardrail-night-0" in test
        assert "guardrail-night-0" not in train

    def test_each_condition_trains_on_the_share_asked_for(self) -> None:
        rows = (
            self.rows("car", "day", 100)
            + self.rows("car", "night", 20)
            + self.rows("bus", "day", 40)
            + self.rows("bus", "night", 10)
        )

        train, test = split_by_condition(rows, train_fraction=0.8, seed=0)

        counts = {
            (label, condition): len(
                [one for one in train if one.startswith(f"{label}-{condition}-")]
            )
            for label, condition in (
                ("car", "day"),
                ("car", "night"),
                ("bus", "day"),
                ("bus", "night"),
            )
        }
        assert counts == {
            ("car", "day"): 80,
            ("car", "night"): 16,
            ("bus", "day"): 32,
            ("bus", "night"): 8,
        }
        assert set(train).isdisjoint(test)
        assert set(train) | set(test) == {row["image_id"] for row in rows}

    def test_the_same_seed_splits_each_condition_the_same_way(self) -> None:
        rows = self.rows("car", "day", 30) + self.rows("car", "night", 7)

        assert split_by_condition(rows, train_fraction=0.8, seed=0) == split_by_condition(
            list(reversed(rows)), train_fraction=0.8, seed=0
        )


class TestChoosingWhichImagesToMeasureOn:
    def test_the_count_asked_for_is_the_count_chosen(self) -> None:
        held = [f"image-{index}" for index in range(1000)]

        assert len(pick_ids(held, count=100, seed=0)) == 100

    def test_fewer_candidates_than_asked_for_gives_all_of_them(self) -> None:
        held = [f"image-{index}" for index in range(8)]

        assert len(pick_ids(held, count=100, seed=0)) == 8

    def test_the_same_seed_chooses_the_same_images(self) -> None:
        held = [f"image-{index}" for index in range(1000)]

        assert pick_ids(held, count=50, seed=0) == pick_ids(held, count=50, seed=0)

    def test_what_was_chosen_is_still_split_in_the_share_asked_for(self) -> None:
        candidates = [f"image-{index}" for index in range(2000)]

        chosen = pick_ids(candidates, count=100, seed=0)
        train, test = split_of(chosen, train_fraction=0.8, seed=0)

        assert len(test) == 20
        assert 10 <= len(set(test) & set(chosen)) <= 30
        assert len(train) == 80

    def test_a_small_class_and_a_large_one_are_split_alike(self) -> None:
        small = pick_ids([f"small-{index}" for index in range(30)], count=30, seed=0)
        large = pick_ids([f"large-{index}" for index in range(3000)], count=30, seed=0)

        _, held = split_of(small, train_fraction=0.8, seed=0)
        _, mine = split_of(large, train_fraction=0.8, seed=0)

        assert len(held) == len(mine) == 6


class TestGraftingTheHeadOntoTheGraph:
    def backbone(self) -> bytes:
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        weights = numpy_helper.from_array(
            np.eye(4, 6, dtype=np.float32) * 2.0, name="backbone_weights"
        )
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["input", "backbone_weights"], ["logits"])],
            "backbone",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 6])],
            [weights],
        )
        held = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        # The highest IR version the serving runtime reads.
        held.ir_version = 10
        onnx.checker.check_model(held)
        return bytes(held.SerializeToString())

    def head(self) -> Head:
        return Head(
            weights=np.arange(18, dtype=np.float64).reshape(6, 3) / 10.0,
            bias=np.asarray([0.5, -0.5, 1.0], dtype=np.float64),
            classes=CLASSES,
        )

    def test_the_grafted_model_answers_our_classes(self) -> None:
        import onnxruntime as ort

        held = grafted(self.backbone(), self.head())
        session = ort.InferenceSession(held, providers=["CPUExecutionProvider"])

        answer = session.run(None, {"input": np.ones((1, 4), dtype=np.float32)})[0]

        assert np.asarray(answer).shape == (1, len(CLASSES))

    def test_it_answers_what_the_head_answers_about_the_same_logits(self) -> None:
        import onnxruntime as ort

        image = np.ones((1, 4), dtype=np.float32)
        logits = np.asarray(
            ort.InferenceSession(self.backbone(), providers=["CPUExecutionProvider"]).run(
                None, {"input": image}
            )[0]
        )
        held = self.head()

        session = ort.InferenceSession(
            grafted(self.backbone(), held), providers=["CPUExecutionProvider"]
        )
        answer = np.asarray(session.run(None, {"input": image})[0])

        assert answer == pytest.approx(scores_of(held, logits.astype(np.float64)), abs=1e-5)

    def test_the_grafted_model_is_a_model_onnx_accepts(self) -> None:
        import onnx

        onnx.checker.check_model(onnx.load_from_string(grafted(self.backbone(), self.head())))

    def test_a_backbone_from_before_initializers_were_separate_is_still_grafted(self) -> None:
        import onnx
        import onnxruntime as ort

        old = onnx.load_from_string(self.backbone())
        old.ir_version = 3
        old.graph.input.extend(
            [
                helper.make_tensor_value_info(one.name, one.data_type, list(one.dims))
                for one in old.graph.initializer
            ]
        )

        held = grafted(old.SerializeToString(), self.head())

        onnx.checker.check_model(onnx.load_from_string(held))
        session = ort.InferenceSession(held, providers=["CPUExecutionProvider"])
        answer = session.run(None, {"input": np.ones((1, 4), dtype=np.float32)})[0]
        assert np.asarray(answer).shape == (1, len(CLASSES))

    def test_grafting_does_not_raise_the_version_the_runtime_has_to_read(self) -> None:
        import onnx

        before = onnx.load_from_string(self.backbone()).ir_version
        after = onnx.load_from_string(grafted(self.backbone(), self.head())).ir_version

        assert after == before

    def test_the_head_is_the_only_thing_added_to_the_graph(self) -> None:
        import onnx

        before = onnx.load_from_string(self.backbone())
        after = onnx.load_from_string(grafted(self.backbone(), self.head()))

        assert len(after.graph.node) == len(before.graph.node) + 3
        assert [one.name for one in after.graph.output] == ["eemls_classes"]


BASE = BaseModelEntry(
    hugging_face_id="owner/b1",
    family="cnn",
    input_height=224,
    input_width=224,
    normalization_mean=(0.5, 0.5, 0.5),
    normalization_std=(0.5, 0.5, 0.5),
    weights_mb=14.0,
)


class TestWhatABuiltVariationRecords:
    def test_the_committed_entry_names_no_source_file_or_digest(self, tmp_path: Path) -> None:
        path = tmp_path / "b1_car" / "b1_car.onnx"
        path.parent.mkdir()
        path.write_bytes(b"x" * 2_000_000)

        entry, _ = recorded("b1_car", BASE, CLASSES, path, "ab" * 32)

        assert not {"hugging_face_id", "onnx_file", "sha256"} & set(entry)
        assert (entry["base_model"], entry["memory_mb"]) == ("b1", 2.0)

    def test_the_local_entry_names_the_backbone_source_the_file_and_its_digest(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "b1_car.onnx"
        path.write_bytes(b"x")

        _, source = recorded("b1_car", BASE, CLASSES, path, "ab" * 32)

        assert source == {
            "hugging_face_id": "owner/b1",
            "onnx_file": "b1_car.onnx",
            "sha256": "ab" * 32,
        }

    def test_writing_the_variations_keeps_the_backbones_of_the_local_file(
        self, tmp_path: Path
    ) -> None:
        held = tmp_path / "models.yaml"
        held.write_text(yaml.safe_dump({"base_models": {"b1": {"sha256": "cd" * 32}}}))

        write_sources(held, {"b1_car": {"sha256": "ab" * 32}})

        assert yaml.safe_load(held.read_text()) == {
            "base_models": {"b1": {"sha256": "cd" * 32}},
            "models": {"b1_car": {"sha256": "ab" * 32}},
        }


class TestTheImagesTheHeadsTrainedOn:
    def test_each_training_image_and_its_blurred_copy_are_written_once(
        self, tmp_path: Path
    ) -> None:
        rows = [
            {
                "image_id": "bdd-day-clear-b",
                "dataset": "bdd",
                "condition": "day",
                "original_label": "b",
            },
            {
                "image_id": "bdd-day-clear-a",
                "dataset": "bdd",
                "condition": "day",
                "original_label": "a",
            },
        ]
        manifest = {"bdd-day-blurred-a": {"image_id": "bdd-day-blurred-a"}}
        path = tmp_path / "trained.txt"

        write_trained_ids(path, rows + rows[:1], manifest)

        assert path.read_text(encoding="utf-8").splitlines() == [
            "bdd-day-blurred-a",
            "bdd-day-clear-a",
            "bdd-day-clear-b",
        ]
