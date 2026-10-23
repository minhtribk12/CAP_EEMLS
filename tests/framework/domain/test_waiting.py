from __future__ import annotations

import asyncio
import threading

from eemls.domain.waiting import all_finished

DEADLINE_S = 5.0


async def finished_but_still_held() -> set[asyncio.Task[None]]:
    answering: set[asyncio.Task[None]] = set()

    async def answer() -> None:
        return None

    task = asyncio.create_task(answer())
    answering.add(task)
    task.add_done_callback(answering.discard)
    await asyncio.sleep(0)
    return answering


def returns_in_time(run: object) -> bool:
    worker = threading.Thread(target=lambda: asyncio.run(run()), daemon=True)  # type: ignore[operator]
    worker.start()
    worker.join(DEADLINE_S)
    return not worker.is_alive()


def test_a_finished_query_its_callback_has_not_removed_yet_does_not_hold_the_wait() -> None:
    async def run() -> None:
        answering = await finished_but_still_held()
        assert any(task.done() for task in answering)
        await all_finished(answering)

    assert returns_in_time(run)


def test_it_waits_for_a_query_still_being_answered() -> None:
    async def run() -> None:
        answering: set[asyncio.Task[None]] = set()
        done: list[bool] = []

        async def answer() -> None:
            await asyncio.sleep(0.05)
            done.append(True)

        task = asyncio.create_task(answer())
        answering.add(task)
        task.add_done_callback(answering.discard)
        await all_finished(answering)
        assert done == [True]

    assert returns_in_time(run)


def held(answering: set[asyncio.Task[None]], task: asyncio.Task[None]) -> None:
    answering.add(task)
    task.add_done_callback(answering.discard)


def test_an_empty_set_returns_at_once() -> None:
    async def run() -> None:
        await all_finished(set())

    assert returns_in_time(run)


def test_a_query_accepted_while_it_waits_is_waited_for_too() -> None:
    async def run() -> None:
        answering: set[asyncio.Task[None]] = set()
        done: list[str] = []

        async def late() -> None:
            await asyncio.sleep(0.05)
            done.append("late")

        async def first() -> None:
            held(answering, asyncio.create_task(late()))
            done.append("first")

        held(answering, asyncio.create_task(first()))
        await all_finished(answering)
        assert done == ["first", "late"]

    assert returns_in_time(run)


def test_a_query_whose_answer_failed_does_not_hold_the_wait() -> None:
    async def run() -> None:
        answering: set[asyncio.Task[None]] = set()

        async def failing() -> None:
            raise RuntimeError("the member did not answer")

        held(answering, asyncio.create_task(failing()))
        await all_finished(answering)

    assert returns_in_time(run)


def test_a_cancelled_query_does_not_hold_the_wait() -> None:
    async def run() -> None:
        answering: set[asyncio.Task[None]] = set()
        task = asyncio.create_task(asyncio.sleep(10))
        held(answering, task)
        task.cancel()
        await all_finished(answering)

    assert returns_in_time(run)
