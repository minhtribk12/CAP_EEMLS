from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

FIXTURE_SIZE = 224
FIXTURE_CLASSES = 1001


@pytest.fixture(scope="session")
def onnx_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    head = np.random.default_rng(0).standard_normal((3, FIXTURE_CLASSES)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["image", "axes"], ["channels"], keepdims=0),
            helper.make_node("MatMul", ["channels", "head"], ["scores"]),
        ],
        "fixture",
        [
            helper.make_tensor_value_info(
                "image", TensorProto.FLOAT, [1, 3, FIXTURE_SIZE, FIXTURE_SIZE]
            )
        ],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, FIXTURE_CLASSES])],
        [
            numpy_helper.from_array(head, name="head"),
            numpy_helper.from_array(np.asarray([2, 3], dtype=np.int64), name="axes"),
        ],
    )
    whole = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    whole.ir_version = 10
    path = tmp_path_factory.mktemp("fixture") / "model.onnx"
    path.write_bytes(whole.SerializeToString())
    return path
