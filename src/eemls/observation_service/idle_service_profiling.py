from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable, Sequence

from eemls.domain.types import EnsembleMember, MemberPrediction, Query
from eemls.domain.waiting import all_finished
from eemls.metrics_database.buffered_writer import WriteError
from eemls.metrics_database.database import MetricsDatabase

logger = logging.getLogger(__name__)

Asking = Callable[[Query, Sequence[EnsembleMember]], Awaitable[Sequence[MemberPrediction]]]


class IdleServiceProfiling:
    def __init__(self, *, database: MetricsDatabase, fraction: float, asking: Asking) -> None:
        self._database = database
        self._fraction = fraction
        self._asking = asking
        self._asked: dict[str, int] = {}
        self._running: set[asyncio.Task[None]] = set()

    def due(self, consumer_id: str) -> bool:
        asked = self._asked.get(consumer_id, 0) + 1
        self._asked[consumer_id] = asked
        return math.floor(asked * self._fraction) > math.floor((asked - 1) * self._fraction)

    def start(self, query: Query, idle: Sequence[EnsembleMember]) -> None:
        if not idle:
            return
        probing = asyncio.create_task(self._probed(query, idle))
        self._running.add(probing)
        probing.add_done_callback(self._running.discard)

    async def drain(self) -> None:
        await all_finished(self._running)

    async def _probed(self, query: Query, idle: Sequence[EnsembleMember]) -> None:
        rows = await self._asking(query, idle)
        await asyncio.to_thread(self._write_rows, query.query_id, rows)

    def _write_rows(self, query_id: str, rows: Sequence[MemberPrediction]) -> None:
        try:
            self._database.write_member_predictions(rows)
        except WriteError as refused:
            logger.error(
                f"stage_tag=store process_tag=gateway error_type=rows_not_written "
                f"query_id={query_id} recovery_action=the_store_holds_them "
                f"detail={refused}"
            )
