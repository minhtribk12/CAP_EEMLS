from __future__ import annotations

from pathlib import Path

import pytest

from eemls.testing.environment import no_ambient_environment
from eemls.testing.trees import committed_tree

__all__ = ["no_ambient_environment"]

OBSERVATION_SERVICE = """\
url: http://observation-service:8083
host: 0.0.0.0
port: 8083
request_timeout_ms: 2000
profile_window_s: 150
profile_min_labeled_per_class: 20
profile_min_labeled_overall: 50
"""


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    held = committed_tree(tmp_path / "config")
    (held / "services" / "observation_service.yaml").write_text(
        OBSERVATION_SERVICE, encoding="utf-8"
    )
    return held
