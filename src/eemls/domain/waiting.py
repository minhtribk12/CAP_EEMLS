from __future__ import annotations

import asyncio
from collections.abc import Set


async def all_finished(tasks: Set[asyncio.Task[None]]) -> None:
    while pending := [task for task in tasks if not task.done()]:
        await asyncio.gather(*pending, return_exceptions=True)
