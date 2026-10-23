from __future__ import annotations

from pathlib import Path

import pytest

from eemls.config.models import InferenceServicesConfig
from eemls.config.reading import ConfigError, Reader, check_the_device_types_are_priced

SERVICES = InferenceServicesConfig.model_validate(
    {
        "services": {"infer-b1-pi4": {"model": "b1", "device_type": "pi4", "replicas": 1}},
        "autoscale": {"enabled": False, "max_replicas": 1, "cpu_percent": 80, "cooldown_s": 0},
    }
)


class TestAFile:
    def test_a_key_given_twice_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "twice.yaml").write_text("a: 1\na: 2\n", encoding="utf-8")

        with pytest.raises(ConfigError, match="appears twice"):
            Reader(tmp_path).read_file("twice.yaml")

    def test_a_mapping_is_read_as_it_is_written(self, tmp_path: Path) -> None:
        (tmp_path / "once.yaml").write_text("a: 1\nb: [x, y]\n", encoding="utf-8")

        assert Reader(tmp_path).read_file("once.yaml") == {"a": 1, "b": ["x", "y"]}

    def test_an_empty_file_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "empty.yaml").write_text("", encoding="utf-8")

        with pytest.raises(ConfigError, match="file is empty"):
            Reader(tmp_path).read_file("empty.yaml")


class TestTheDevicePrices:
    def test_an_unpriced_device_type_is_refused_by_the_file_that_misses_it(self) -> None:
        with pytest.raises(ConfigError, match="which cost/elsewhere.yaml does not hold"):
            check_the_device_types_are_priced(SERVICES, {}, "cost/elsewhere.yaml")
