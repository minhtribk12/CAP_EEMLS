from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import psycopg
import pytest
from psycopg_pool import PoolTimeout

from eemls.metrics_database import postgres
from eemls.metrics_database.buffered_writer import WriteError
from eemls.metrics_database.postgres import PostgresDatabase
from eemls.testing.records import served
from tests.framework.metrics_database.conftest import WRITING


class DownPool:
    failure: Exception = psycopg.OperationalError()

    def __init__(self, dsn: str, **_: Any) -> None: ...

    @staticmethod
    def check_connection(held: Any) -> None: ...

    def connection(self) -> AbstractContextManager[Any]:
        raise self.failure

    def close(self) -> None: ...


@pytest.mark.parametrize(
    "failure",
    [psycopg.OperationalError("connection refused"), PoolTimeout("no connection in time")],
)
def test_a_write_with_the_connection_down_raises_and_keeps_every_row(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    monkeypatch.setattr(postgres, "reachable", lambda dsn, **_: None)
    monkeypatch.setattr(DownPool, "failure", failure)
    monkeypatch.setattr(postgres, "ConnectionPool", DownPool)
    held = PostgresDatabase(dsn="postgresql://down", run_id="r1", **WRITING)
    for index in range(3):
        held.write_query(served(query_id=f"q-{index}"))

    with pytest.raises(WriteError):
        held.flush()

    assert held._writer.depth == 3
