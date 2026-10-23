from __future__ import annotations

from eemls.observation_service.observation_service_api import (
    ProfileUpdateAnswer,
    ProfileUpdateRequest,
    RowBatch,
)
from eemls.testing.records import AT, decision, feedback, prediction, profile, served


def batch() -> RowBatch:
    return RowBatch(
        run_id="run-1",
        sender="control-plane-1",
        queries=(served(), served(query_id="q-2", outcome="refused", **unanswered())),
        member_predictions=(prediction(), prediction(in_ensemble=False)),
        feedback=(feedback(),),
        decisions=(decision(),),
    )


def unanswered() -> dict[str, object]:
    return dict.fromkeys(
        (
            "t_result_ready",
            "wait_ms",
            "aggregate_ms",
            "deployment_id",
            "n_members",
            "n_answered",
            "predicted_label",
            "predicted_confidence",
            "cost_per_1k",
            "throughput_rps",
        )
    )


class TestTheObservationServiceApi:
    def test_a_batch_comes_back_from_json_as_it_was_sent(self) -> None:
        sent = batch()

        assert RowBatch.model_validate_json(sent.model_dump_json()) == sent

    def test_a_batch_without_rows_holds_empty_tables(self) -> None:
        held = RowBatch.model_validate({"run_id": "run-1", "sender": "control-plane-1"})

        assert held.rows == 0

    def test_a_batch_counts_its_rows(self) -> None:
        assert batch().rows == 6

    def test_an_update_request_comes_back_from_json_as_it_was_sent(self) -> None:
        sent = ProfileUpdateRequest(at=AT)

        assert ProfileUpdateRequest.model_validate_json(sent.model_dump_json()) == sent

    def test_an_update_answer_comes_back_from_json_as_it_was_sent(self) -> None:
        sent = ProfileUpdateAnswer(profiles=(profile(), profile(instance_id="i-2")))

        assert ProfileUpdateAnswer.model_validate_json(sent.model_dump_json()) == sent
