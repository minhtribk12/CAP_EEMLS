from __future__ import annotations

import logging

import httpx
import pytest

from eemls.observation_service.operating_conditions import ConditionLookup


def hub(**answers: object) -> tuple[ConditionLookup, list[str]]:
    asked: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        image_id = request.url.path.removeprefix("/images/").removesuffix("/meta")
        asked.append(image_id)
        held = answers.get(image_id)
        if held is None:
            return httpx.Response(httpx.codes.NOT_FOUND, json={"detail": "no such image"})
        if isinstance(held, int):
            return httpx.Response(held)
        return httpx.Response(httpx.codes.OK, json=held)

    return (
        ConditionLookup(
            base_url="http://data-hub:8081",
            timeout_ms=1000.0,
            transport=httpx.MockTransport(handle),
        ),
        asked,
    )


def meta(condition: str) -> dict[str, object]:
    return {
        "image_id": "an-image",
        "label": "car",
        "dataset": "a-set",
        "original_label": "car",
        "condition": condition,
        "image_quality": "clear",
    }


class TestReadingTheConditionOfAnImage:
    async def test_it_is_the_condition_the_hub_reports(self) -> None:
        held, _ = hub(night_one=meta("night"))

        assert await held("night_one") == "night"

    async def test_the_day_is_read_as_the_day(self) -> None:
        held, _ = hub(day_one=meta("day"))

        assert await held("day_one") == "day"

    async def test_one_image_is_asked_about_once(self) -> None:
        held, asked = hub(one=meta("night"))

        assert await held("one") == "night"
        assert await held("one") == "night"
        assert asked == ["one"]

    async def test_every_image_is_asked_about_on_its_own(self) -> None:
        held, asked = hub(one=meta("night"), other=meta("day"))

        assert await held("one") == "night"
        assert await held("other") == "day"
        assert asked == ["one", "other"]


class TestWhenTheHubDoesNotAnswer:
    async def test_a_refusal_keeps_the_condition_of_the_last_image_answered(self) -> None:
        held, _ = hub(one=meta("night"), other=httpx.codes.INTERNAL_SERVER_ERROR)

        assert await held("one") == "night"
        assert await held("other") == "night"

    async def test_an_image_the_hub_does_not_hold_keeps_the_last_condition(self) -> None:
        held, _ = hub(one=meta("night"))

        assert await held("one") == "night"
        assert await held("nobody") == "night"

    async def test_a_hub_that_cannot_be_reached_keeps_the_last_condition(self) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to the data hub")

        held = ConditionLookup(
            base_url="http://data-hub:8081",
            timeout_ms=1000.0,
            transport=httpx.MockTransport(refuse),
        )

        assert await held("one") == "day"

    async def test_nothing_has_been_answered_yet_so_the_day_is_assumed(self) -> None:
        held, _ = hub()

        assert await held("one") == "day"

    async def test_an_answer_naming_no_condition_keeps_the_last_one(self) -> None:
        held, _ = hub(one=meta("night"), other={"image_id": "other", "label": "car"})

        assert await held("one") == "night"
        assert await held("other") == "night"

    async def test_an_answer_that_is_not_the_image_meta_of_the_hub_keeps_the_last_one(
        self,
    ) -> None:
        held, _ = hub(one=meta("day"), other={"condition": "night"})

        assert await held("one") == "day"
        assert await held("other") == "day"

    async def test_a_condition_this_system_does_not_know_is_not_taken(self) -> None:
        held, _ = hub(one=meta("night"), other=meta("dusk"))

        assert await held("one") == "night"
        assert await held("other") == "night"

    async def test_a_failure_is_not_remembered_as_an_answer(self) -> None:
        held, asked = hub(one=httpx.codes.INTERNAL_SERVER_ERROR)

        await held("one")
        await held("one")

        assert asked == ["one", "one"]

    async def test_the_failure_is_logged_with_what_was_done_about_it(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        held, _ = hub(one=httpx.codes.INTERNAL_SERVER_ERROR)

        with caplog.at_level(logging.WARNING):
            await held("one")

        assert "stage_tag=condition_lookup" in caplog.text
        assert "recovery_action=the_last_condition_is_kept" in caplog.text
        assert "image_id=one" in caplog.text

    async def test_a_refusal_is_logged_as_a_refusal(self, caplog: pytest.LogCaptureFixture) -> None:
        held, _ = hub(one=httpx.codes.INTERNAL_SERVER_ERROR)

        with caplog.at_level(logging.WARNING):
            await held("one")

        assert "error_type=refused" in caplog.text
        assert "detail=500" in caplog.text

    async def test_a_condition_we_do_not_know_is_logged_as_that_and_not_as_a_refusal(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        held, _ = hub(one=meta("dusk"))

        with caplog.at_level(logging.WARNING):
            await held("one")

        assert "error_type=unknown_condition" in caplog.text
        assert "detail=dusk" in caplog.text


class TestLettingItGo:
    async def test_what_it_already_knows_it_still_answers(self) -> None:
        held, _ = hub(one=meta("night"))

        await held("one")
        await held.close()

        assert await held("one") == "night"

    async def test_an_image_it_has_not_read_is_not_asked_for_once_it_is_closed(self) -> None:
        held, _ = hub(one=meta("night"), other=meta("day"))

        await held("one")
        await held.close()

        with pytest.raises(RuntimeError):
            await held("other")
