from __future__ import annotations

from collections.abc import Iterator

import pytest

from eemls.metrics_database.database import MetricsDatabase
from eemls.metrics_database.testing import MemoryMetricsDatabase
from tests.framework.metrics_database.conftest import WRITING

RUN = "contract"


@pytest.fixture(
    params=[
        pytest.param("memory", id="memory"),
        pytest.param("postgres", id="postgres", marks=pytest.mark.integration),
        pytest.param("recent", id="recent", marks=pytest.mark.integration),
    ]
)
def database(request: pytest.FixtureRequest) -> Iterator[MetricsDatabase]:
    if request.param == "memory":
        held: MetricsDatabase = MemoryMetricsDatabase(run_id=RUN)
        held.apply_schema()
        yield held
        return
    from eemls.metrics_database.postgres import PostgresDatabase

    dsn = request.getfixturevalue("dsn")
    run_id = request.getfixturevalue("run_id")
    postgres = PostgresDatabase(dsn=dsn, run_id=run_id, **WRITING)
    postgres.apply_schema()
    try:
        if request.param == "recent":
            from eemls.metrics_database.recent_windows import RecentWindows

            yield RecentWindows(postgres, keep=500)
        else:
            yield postgres
    finally:
        postgres.close()
