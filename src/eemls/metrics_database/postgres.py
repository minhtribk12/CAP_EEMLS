from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

import psycopg
from psycopg import Connection, sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from eemls.config.services import MetricsDatabaseConfig
from eemls.domain.errors import EemlsError
from eemls.domain.measuring import Measuring, NoMeasuring
from eemls.domain.records import (
    Decision,
    Feedback,
    InferenceServiceProfile,
    Instance,
    RunOwner,
    ServedQuery,
)
from eemls.domain.types import FrozenModel, MemberPrediction, Row, Timestamp
from eemls.metrics_database.buffered_writer import Batch, BufferedWriter, WriteError
from eemls.metrics_database.schema import (
    FROM_FEEDBACK,
    FROM_THE_SLOWEST_MEMBER,
    exported_order,
    json_columns,
    run_row,
    schema_statements,
    table_columns,
)

logger = logging.getLogger(__name__)

SCHEMA_LOCK = 0x45454D4C

SELECTED: tuple[str, ...] = (
    *(name for name in table_columns("queries") if name != "run_id"),
    *FROM_FEEDBACK,
)


def _window_selection() -> sql.Composed:
    return sql.SQL(", ").join(
        sql.SQL("{}.{}").format(
            sql.Identifier("f" if name in FROM_FEEDBACK else "q"), sql.Identifier(name)
        )
        for name in SELECTED
    )


WINDOW_READ: sql.Composed = sql.SQL(
    """
WITH newest AS (
    SELECT * FROM queries
    WHERE run_id = %(run_id)s AND consumer_id = %(consumer_id)s
    ORDER BY t_arrive DESC, query_id DESC
    LIMIT %(limit)s
)
SELECT {selection}
FROM newest AS q
LEFT JOIN feedback AS f
    ON f.run_id = q.run_id AND f.query_id = q.query_id AND f.consumer_id = q.consumer_id
ORDER BY q.t_arrive ASC, q.query_id ASC
"""
).format(selection=_window_selection())

SLOWEST_MEMBER_READ = """
SELECT DISTINCT ON (query_id) query_id, total_ms, fetch_ms, inference_ms
FROM member_predictions
WHERE run_id = %(run_id)s AND query_id = ANY(%(query_ids)s)
  AND in_ensemble AND status = 'ok'
ORDER BY query_id, total_ms DESC, service_name DESC
"""

MEMBER_WINDOW = """
WITH newest AS (
    SELECT run_id, query_id, t_arrive FROM queries
    WHERE run_id = %(run_id)s AND consumer_id = %(consumer_id)s
    ORDER BY t_arrive DESC, query_id DESC
    LIMIT %(limit)s
)
SELECT m.*, f.true_label
FROM newest AS q
JOIN member_predictions AS m ON m.run_id = q.run_id AND m.query_id = q.query_id
LEFT JOIN feedback AS f ON f.run_id = q.run_id AND f.query_id = q.query_id
ORDER BY q.t_arrive ASC, q.query_id ASC, m.service_name ASC, m.in_ensemble DESC
"""

SERVED = """
SELECT * FROM queries
WHERE run_id = %(run_id)s AND consumer_id = %(consumer_id)s
  AND t_arrive >= %(since)s AND t_arrive < %(until)s
ORDER BY t_arrive ASC, query_id ASC
"""

MEMBER_ANSWERS = """
SELECT m.*, f.true_label
FROM member_predictions AS m
JOIN queries AS q ON q.run_id = m.run_id AND q.query_id = m.query_id
LEFT JOIN feedback AS f ON f.run_id = m.run_id AND f.query_id = m.query_id
WHERE m.run_id = %(run_id)s AND q.t_arrive >= %(since)s AND q.t_arrive < %(until)s
ORDER BY m.query_id, m.service_name
"""

STOP_INSTANCE = """
UPDATE instances SET stopped_at = %(stopped_at)s
WHERE run_id = %(run_id)s AND instance_id = %(instance_id)s AND stopped_at IS NULL
"""

CLAIM_RUN = """
WITH claimed AS (
    INSERT INTO run_owners (run_id, instance_id, claimed_at)
    VALUES (%(run_id)s, %(instance_id)s, %(claimed_at)s)
    ON CONFLICT (run_id) DO NOTHING
    RETURNING instance_id
)
SELECT instance_id FROM claimed
UNION ALL
SELECT instance_id FROM run_owners WHERE run_id = %(run_id)s
LIMIT 1
"""

RECENT_DECISIONS = """
SELECT *
FROM decisions
WHERE run_id = %(run_id)s
ORDER BY t DESC, decision_id DESC
LIMIT %(limit)s
"""

LATEST_PROFILES = """
SELECT DISTINCT ON (instance_id) *
FROM inference_service_profiles
WHERE run_id = %(run_id)s
ORDER BY instance_id, window_end DESC
"""


RETRY_S = 0.5


class DatabaseUnavailableError(EemlsError): ...


class PostgresDatabase:
    def __init__(
        self,
        *,
        dsn: str,
        run_id: str,
        monitoring: Measuring | None = None,
        buffer_rows: int,
        flush_interval_ms: float,
        statement_timeout_ms: int,
        max_connections: int,
        connect_timeout_s: float = 5.0,
    ) -> None:
        self.run_id = run_id
        self._closed = False
        self._lock = threading.Lock()
        reachable(dsn, connect_timeout_s=connect_timeout_s)
        self._pool = ConnectionPool(
            dsn,
            min_size=1,
            max_size=max_connections,
            open=True,
            timeout=connect_timeout_s,
            check=ConnectionPool.check_connection,
            kwargs={"options": f"-c statement_timeout={statement_timeout_ms}"},
        )
        reported: Measuring = NoMeasuring() if monitoring is None else monitoring
        self._monitoring = reported
        self._writer = BufferedWriter(
            self._write,
            monitoring=reported,
            buffer_rows=buffer_rows,
            flush_interval_ms=flush_interval_ms,
        )

    @classmethod
    def from_config(
        cls,
        settings: MetricsDatabaseConfig,
        *,
        password: str,
        run_id: str,
        monitoring: Measuring | None = None,
        statement_timeout_ms: int | None = None,
    ) -> PostgresDatabase:
        return cls(
            dsn=settings.dsn(password),
            run_id=run_id,
            monitoring=monitoring,
            buffer_rows=settings.buffer_rows,
            flush_interval_ms=float(settings.flush_interval_ms),
            statement_timeout_ms=(
                settings.statement_timeout_ms
                if statement_timeout_ms is None
                else statement_timeout_ms
            ),
            max_connections=settings.max_database_connections,
        )

    def apply_schema(self) -> None:
        with self._pool.connection() as held:
            held.execute("SELECT pg_advisory_xact_lock(%s)", (SCHEMA_LOCK,))
            for statement in schema_statements():
                held.execute(statement)

    def register_instance(self, instance: Instance) -> None:
        self._hold("instances", instance)

    def write_instance_stopped(self, instance_id: str, stopped_at: Timestamp) -> None:
        self.flush()
        with self._pool.connection() as held:
            held.execute(
                STOP_INSTANCE,
                {"run_id": self.run_id, "instance_id": instance_id, "stopped_at": stopped_at},
            )

    def write_query(self, query: ServedQuery) -> None:
        self._hold("queries", query)

    def write_member_predictions(self, predictions: Sequence[MemberPrediction]) -> None:
        for one in predictions:
            self._hold("member_predictions", one)

    def write_feedback(self, feedback: Feedback) -> None:
        self._hold("feedback", feedback)

    def write_decision(self, decision: Decision) -> None:
        self._hold("decisions", decision)

    def write_profile(self, profile: InferenceServiceProfile) -> None:
        self._hold("inference_service_profiles", profile)

    def read_window(self, consumer_id: str, limit: int) -> Sequence[Row]:
        wanted = max(0, int(limit))
        self._flush_for_a_reader()
        with self._pool.connection() as held, held.cursor(binary=True) as cursor:
            cursor.execute(
                WINDOW_READ,
                {"run_id": self.run_id, "consumer_id": consumer_id, "limit": wanted},
            )
            served = cursor.fetchall()
            rows = [dict(zip(SELECTED, row, strict=True)) for row in served]
            cursor.execute(
                SLOWEST_MEMBER_READ,
                {"run_id": self.run_id, "query_ids": [str(row["query_id"]) for row in rows]},
            )
            slowest = {
                held_id: {"total_ms": total, "fetch_ms": fetch, "inference_ms": inference}
                for held_id, total, fetch, inference in cursor.fetchall()
            }
        for row in rows:
            row.update(slowest.get(str(row["query_id"]), dict.fromkeys(FROM_THE_SLOWEST_MEMBER)))
        return rows

    def read_member_window(self, consumer_id: str, limit: int) -> Sequence[Row]:
        wanted = max(0, int(limit))
        self._flush_for_a_reader()
        with self._pool.connection() as held, held.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                MEMBER_WINDOW,
                {"run_id": self.run_id, "consumer_id": consumer_id, "limit": wanted},
            )
            return [_without_run(row) for row in cursor.fetchall()]

    def read_member_answers(self, *, since: Timestamp, until: Timestamp) -> Sequence[Row]:
        self._flush_for_a_reader()
        with self._pool.connection() as held, held.cursor(row_factory=dict_row) as cursor:
            cursor.execute(MEMBER_ANSWERS, {"run_id": self.run_id, "since": since, "until": until})
            return [_without_run(row) for row in cursor.fetchall()]

    def read_served(self, *, consumer_id: str, since: Timestamp, until: Timestamp) -> Sequence[Row]:
        self._flush_for_a_reader()
        with self._pool.connection() as held, held.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                SERVED,
                {"run_id": self.run_id, "consumer_id": consumer_id, "since": since, "until": until},
            )
            return [_without_run(row) for row in cursor.fetchall()]

    def read_rows(self, table: str) -> Sequence[Row]:
        order = exported_order(table)
        self._flush_for_a_reader()
        statement = sql.SQL("SELECT * FROM {} WHERE run_id = %(run_id)s ORDER BY {}").format(
            sql.Identifier(table), sql.SQL(", ").join(sql.Identifier(name) for name in order)
        )
        with self._pool.connection() as held, held.cursor(row_factory=dict_row) as cursor:
            cursor.execute(statement, {"run_id": self.run_id})
            return [_without_run(row) for row in cursor.fetchall()]

    def claim_run(self, owner: RunOwner) -> str:
        with self._pool.connection() as held, held.cursor() as cursor:
            cursor.execute(
                CLAIM_RUN,
                {
                    "run_id": self.run_id,
                    "instance_id": owner.instance_id,
                    "claimed_at": owner.claimed_at,
                },
            )
            row = cursor.fetchone()
        if row is None:
            raise WriteError(f"the run {self.run_id} could not be claimed")
        return str(row[0])

    def read_decisions(self, limit: int) -> Sequence[Decision]:
        wanted = max(0, int(limit))
        self._flush_for_a_reader()
        with self._pool.connection() as held, held.cursor(row_factory=dict_row) as cursor:
            cursor.execute(RECENT_DECISIONS, {"run_id": self.run_id, "limit": wanted})
            return [Decision.model_validate(_without_run(row)) for row in cursor]

    def read_profiles(self) -> Sequence[InferenceServiceProfile]:
        self._flush_for_a_reader()
        with self._pool.connection() as held, held.cursor(row_factory=dict_row) as cursor:
            cursor.execute(LATEST_PROFILES, {"run_id": self.run_id})
            return [InferenceServiceProfile.model_validate(_without_run(row)) for row in cursor]

    def flush(self) -> None:
        self._writer.flush()

    def _flush_for_a_reader(self) -> None:
        self._writer.flush_quietly()

    def close(self) -> None:
        with self._lock:
            self._closed = True
        try:
            self._writer.close_quietly()
        finally:
            self._pool.close()

    def _hold(self, table: str, value: FrozenModel) -> None:
        with self._lock:
            if self._closed:
                raise WriteError(f"the database is closed, so {table} takes no more rows")
        self._writer.add(table, run_row(table, value, run_id=self.run_id))

    def _write(self, batch: Batch) -> None:
        try:
            with self._pool.connection() as held:
                for table, rows in batch.items():
                    if rows:
                        _insert(held, table, rows)
            return
        except psycopg.OperationalError:
            raise
        except (psycopg.Error, ValueError) as refused:
            logger.warning(
                f"error_type=batch_refused rows={sum(len(r) for r in batch.values())} "
                f"recovery_action=retry_each_row detail={refused}"
            )
        self._write_each(batch)

    def _write_each(self, batch: Batch) -> None:
        for table, rows in batch.items():
            for row in rows:
                try:
                    with self._pool.connection() as held:
                        _insert(held, table, (row,))
                except (psycopg.DataError, psycopg.IntegrityError, ValueError) as refused:
                    self._monitoring.count("eemls_metrics_database_rows_refused_total", table=table)
                    logger.error(
                        f"error_type=row_refused table={table} "
                        f"recovery_action=dropped detail={refused}"
                    )


def _insert(held: Connection[Any], table: str, rows: Sequence[Row]) -> None:
    names = [name for name in table_columns(table) if name in rows[0]]
    for row in rows:
        if set(row) != set(names):
            raise ValueError(f"a {table} row carries {sorted(set(row) ^ set(names))}")
    as_json = json_columns(table)
    statement = sql.SQL("INSERT INTO {table} ({columns}) VALUES ({values})").format(
        table=sql.Identifier(table),
        columns=sql.SQL(", ").join(sql.Identifier(name) for name in names),
        values=sql.SQL(", ").join(sql.Placeholder(name) for name in names),
    ) + sql.SQL(" ON CONFLICT DO NOTHING")
    with held.cursor() as cursor:
        cursor.executemany(
            statement,
            [
                {name: Jsonb(row[name]) if name in as_json else row[name] for name in names}
                for row in rows
            ],
        )


def _without_run(row: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in row.items() if name != "run_id"}


def reachable(dsn: str, *, connect_timeout_s: float, retry_s: float = RETRY_S) -> None:
    deadline = time.monotonic() + connect_timeout_s
    while True:
        try:
            psycopg.connect(dsn, connect_timeout=max(1, round(connect_timeout_s))).close()
        except psycopg.OperationalError as refused:
            if time.monotonic() >= deadline:
                raise DatabaseUnavailableError(
                    f"the metrics database refused the connection: {refused}"
                ) from refused
            time.sleep(retry_s)
        else:
            return
