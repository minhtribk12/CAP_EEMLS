from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from eemls_consumer.load_test import (
    Step,
    StepOutcome,
    reported,
    run_load_test,
    send_times,
    steps_of,
    table,
)

AT = datetime(2026, 10, 18, 9, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += max(0.0, seconds)
        await asyncio.sleep(0)


def run(steps: tuple[Step, ...], post: object, *, max_in_flight: int = 100) -> list[StepOutcome]:
    clock = Clock()
    return asyncio.run(
        run_load_test(
            steps,
            post=post,  # type: ignore[arg-type]
            clock=clock,
            sleep=clock.sleep,
            wall=lambda: AT + timedelta(seconds=clock.now),
            max_in_flight=max_in_flight,
        )
    )


class TestTheSteps:
    def test_each_rate_is_held_for_the_length_of_a_step(self) -> None:
        assert steps_of([10.0, 20.0], step_s=30.0) == (Step(10.0, 30.0), Step(20.0, 30.0))

    @pytest.mark.parametrize(("rates", "step_s"), [([0.0], 30.0), ([10.0], 0.0), ([], 30.0)])
    def test_a_rate_or_a_length_that_asks_nothing_is_refused(
        self, rates: list[float], step_s: float
    ) -> None:
        with pytest.raises(ValueError, match="step"):
            steps_of(rates, step_s=step_s)

    def test_a_query_is_sent_every_one_over_the_rate(self) -> None:
        assert send_times(Step(10.0, 0.5)) == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])


class TestTheRateItKeeps:
    def test_every_query_is_sent_however_long_its_answer_takes(self) -> None:
        release = asyncio.Event()

        async def slow(index: int) -> int:
            await release.wait()
            return 202

        async def later() -> list[StepOutcome]:
            clock = Clock()
            sending = asyncio.create_task(
                run_load_test(
                    (Step(20.0, 1.0),),
                    post=slow,
                    clock=clock,
                    sleep=clock.sleep,
                    wall=lambda: AT,
                    max_in_flight=100,
                )
            )
            while clock.now < 0.95:
                await asyncio.sleep(0)
            release.set()
            return await sending

        (sent,) = asyncio.run(later())

        assert sent.accepted == 20

    def test_a_query_over_the_in_flight_cap_is_counted_and_not_sent(self) -> None:
        release = asyncio.Event()
        started: list[int] = []

        async def held(index: int) -> int:
            started.append(index)
            await release.wait()
            return 202

        async def later() -> list[StepOutcome]:
            clock = Clock()
            sending = asyncio.create_task(
                run_load_test(
                    (Step(20.0, 1.0),),
                    post=held,
                    clock=clock,
                    sleep=clock.sleep,
                    wall=lambda: AT,
                    max_in_flight=5,
                )
            )
            while clock.now < 0.95:
                await asyncio.sleep(0)
            release.set()
            return await sending

        (sent,) = asyncio.run(later())

        assert len(started) == 5
        assert (sent.accepted, sent.not_sent) == (5, 15)

    @pytest.mark.parametrize(("status", "kind"), [(202, "accepted"), (503, "refused")])
    def test_each_answer_is_counted_by_what_it_said(self, status: int, kind: str) -> None:
        async def answer(index: int) -> int:
            return status

        (sent,) = run((Step(10.0, 1.0),), answer)

        assert getattr(sent, kind) == 10

    def test_an_answer_that_never_came_is_counted_as_failed(self) -> None:
        async def broken(index: int) -> int:
            raise httpx.ConnectError("the control plane closed the connection")

        (sent,) = run((Step(10.0, 1.0),), broken)

        assert sent.failed == 10

    def test_each_step_records_the_stretch_of_time_it_asked_in(self) -> None:
        async def answer(index: int) -> int:
            return 202

        first, second = run((Step(10.0, 2.0), Step(10.0, 3.0)), answer)

        assert (first.since, first.until) == (AT, AT + timedelta(seconds=2))
        assert (second.since, second.until) == (
            AT + timedelta(seconds=2),
            AT + timedelta(seconds=5),
        )


def served(index: int, *, response_ms: float, outcome: str = "accepted") -> dict[str, object]:
    return {
        "query_id": f"q-{index}",
        "response_time_ms": response_ms,
        "wait_ms": response_ms - 4.0,
        "aggregate_ms": 1.0,
        "outcome": outcome,
    }


def answered(index: int, *, fetch_ms: float, inference_ms: float) -> dict[str, object]:
    return {
        "query_id": f"q-{index}",
        "status": "ok",
        "in_ensemble": True,
        "fetch_ms": fetch_ms,
        "inference_ms": inference_ms,
    }


SENT = StepOutcome(rate_per_s=10.0, since=AT, until=AT, accepted=4, refused=1, failed=0, not_sent=0)


class TestTheReportOfAStep:
    def test_the_response_times_are_read_at_their_percentiles(self) -> None:
        rows = [served(index, response_ms=float(ms)) for index, ms in enumerate([10, 20, 30, 40])]

        held = reported(SENT, rows, [])

        assert held.p50_ms == pytest.approx(25.0)
        assert held.p95_ms == pytest.approx(38.5)
        assert held.p99_ms == pytest.approx(39.7)

    def test_a_query_no_member_answered_is_counted_and_left_out_of_the_times(self) -> None:
        rows = [served(0, response_ms=10.0), served(1, response_ms=1000.0, outcome="failed")]

        held = reported(SENT, rows, [])

        assert (held.answered, held.unanswered) == (1, 1)
        assert held.p99_ms == pytest.approx(10.0)

    def test_the_time_is_split_between_fetching_running_waiting_and_voting(self) -> None:
        rows = [served(0, response_ms=50.0)]
        members = [
            answered(0, fetch_ms=3.0, inference_ms=10.0),
            answered(0, fetch_ms=5.0, inference_ms=30.0),
        ]

        held = reported(SENT, rows, members)

        assert held.fetch_ms == pytest.approx(4.0)
        assert held.inference_ms == pytest.approx(20.0)
        assert held.wait_ms == pytest.approx(46.0)
        assert held.aggregate_ms == pytest.approx(1.0)

    def test_a_step_that_stored_no_answer_reports_no_times(self) -> None:
        held = reported(SENT, [], [])

        assert held.p50_ms is None
        assert held.answered == 0


class TestTheTable:
    def test_it_holds_one_row_per_step_under_one_header(self) -> None:
        rows = [served(0, response_ms=10.0)]
        written = table([reported(SENT, rows, []), reported(SENT, rows, [])])

        lines = written.strip().splitlines()

        assert lines[0].startswith("| rate/s |")
        assert len(lines) == 4
