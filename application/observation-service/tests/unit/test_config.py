from __future__ import annotations

from pathlib import Path

import pytest

from eemls.config.reading import ConfigError
from eemls_observation_service.config import FILES_READ, load_observation_service_config


class TestItsViewOfTheTree:
    def test_it_reads_its_own_service_file(self, tree: Path) -> None:
        settings = load_observation_service_config(tree).services.observation_service

        assert (str(settings.url), settings.port) == ("http://observation-service:8083/", 8083)
        assert settings.request_timeout_ms == 2000

    def test_it_reads_the_profile_keys_from_its_own_service_file(self, tree: Path) -> None:
        settings = load_observation_service_config(tree).services.observation_service

        assert (
            settings.profile_window_s,
            settings.profile_min_labeled_per_class,
            settings.profile_min_labeled_overall,
        ) == (150, 20, 50)

    def test_it_prices_the_devices(self, tree: Path) -> None:
        assert load_observation_service_config(tree).cost.rates()

    def test_a_tree_without_its_service_file_is_refused(self, tree: Path) -> None:
        (tree / "services" / "observation_service.yaml").unlink()

        with pytest.raises(ConfigError, match="observation_service.yaml"):
            load_observation_service_config(tree)

    def test_an_unknown_key_in_its_service_file_is_refused(self, tree: Path) -> None:
        path = tree / "services" / "observation_service.yaml"
        path.write_text(path.read_text(encoding="utf-8") + "unknown: 1\n", encoding="utf-8")

        with pytest.raises(ConfigError, match="unknown"):
            load_observation_service_config(tree)

    def test_the_files_it_reads_are_named(self) -> None:
        assert sorted(FILES_READ) == [
            "cost/prices.yaml",
            "services/broker.yaml",
            "services/metrics_database.yaml",
            "services/monitoring.yaml",
            "services/observation_service.yaml",
        ]
