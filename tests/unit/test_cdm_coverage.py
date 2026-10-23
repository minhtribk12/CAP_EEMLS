from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eemls.domain.consumer_defined_metrics import CdmSet
from eemls.metrics_database.schema import CoverageError, check_coverage, readable_columns

ROOT = Path(__file__).resolve().parents[2]
CONSUMERS = ROOT / "config" / "consumers"


def declared(folder: str) -> CdmSet:
    body = yaml.safe_load((CONSUMERS / folder / "cdm.yaml").read_text(encoding="utf-8"))
    return CdmSet.model_validate({"definitions": body["cdms"]})


class TestTheCommittedConsumers:
    @pytest.mark.parametrize("folder", sorted(path.name for path in CONSUMERS.iterdir()))
    def test_every_field_every_consumer_reads_is_a_stored_column(self, folder: str) -> None:
        check_coverage(declared(folder))

    def test_the_committed_configuration_reads_the_five_the_article_needs(self) -> None:
        reads: set[str] = set()
        for folder in sorted(path.name for path in CONSUMERS.iterdir()):
            reads |= declared(folder).attributes()

        assert reads == {
            "cost_per_1k",
            "predicted_label",
            "response_time_ms",
            "throughput_rps",
            "true_label",
        }

    def test_the_readable_fields_span_the_tables_a_window_read_joins(self) -> None:
        readable = readable_columns()

        assert {
            "true_label",
            "predicted_label",
            "response_time_ms",
            "cost_per_1k",
        } <= readable
        assert "throughput_rps" in readable


class TestWhatIsRefused:
    def metric(self, attribute: str) -> CdmSet:
        return CdmSet.model_validate(
            {
                "definitions": {
                    "invented": {
                        "measured": "per_window",
                        "window": 500,
                        "unit": "ratio",
                        "description": "Reads an attribute nothing stores",
                        "specification": {"function": "average", "attribute": attribute},
                    }
                }
            }
        )

    def test_a_cdm_naming_an_unknown_field_is_refused_with_that_name(self) -> None:
        with pytest.raises(CoverageError, match="unicorn_count"):
            check_coverage(self.metric("unicorn_count"))

    def test_the_refusal_names_the_metric_that_reads_it(self) -> None:
        with pytest.raises(CoverageError, match="invented"):
            check_coverage(self.metric("unicorn_count"))

    def test_a_field_hidden_inside_a_filter_is_checked_too(self) -> None:
        asked = CdmSet.model_validate(
            {
                "definitions": {
                    "filtered": {
                        "measured": "per_window",
                        "window": 500,
                        "unit": "count",
                        "description": "Contexts on an attribute nothing stores",
                        "specification": {
                            "function": "count",
                            "scopes": [{"attribute": "unicorn_color", "value": "white"}],
                        },
                    }
                }
            }
        )

        with pytest.raises(CoverageError, match="unicorn_color"):
            check_coverage(asked)

    def test_a_field_hidden_inside_a_group_by_is_checked_too(self) -> None:
        asked = CdmSet.model_validate(
            {
                "definitions": {
                    "grouped": {
                        "measured": "per_window",
                        "window": 500,
                        "unit": "count",
                        "description": "Groups by an attribute nothing stores",
                        "specification": {
                            "function": "count",
                            "group_by": "unicorn_breed",
                        },
                    }
                }
            }
        )

        with pytest.raises(CoverageError, match="unicorn_breed"):
            check_coverage(asked)

    def test_a_metric_that_reads_nothing_is_covered(self) -> None:
        asked = CdmSet.model_validate(
            {
                "definitions": {
                    "queries": {
                        "measured": "per_window",
                        "window": 500,
                        "unit": "count",
                        "description": "How many queries the window holds",
                        "specification": {"function": "count"},
                    }
                }
            }
        )

        check_coverage(asked)
