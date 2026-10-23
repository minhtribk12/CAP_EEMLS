from __future__ import annotations

from datetime import timedelta

import pytest

from eemls.domain.records import Feedback, RunOwner, ServedQuery
from eemls.evaluation_engine.window import Window
from eemls.metrics_database.database import MetricsDatabase
from eemls.metrics_database.schema import CoverageError
from eemls.testing.records import AT, decision, member, prediction, profile, served


def answered(index: int, consumer_id: str = "c1_navigation", **changes: object) -> ServedQuery:
    arrived = AT + timedelta(seconds=index)
    return served(
        query_id=f"q-{consumer_id}-{index}",
        consumer_id=consumer_id,
        t_arrive=arrived,
        t_result_ready=arrived + timedelta(milliseconds=174),
        t_publish=arrived + timedelta(milliseconds=180),
        **changes,
    )


def labeled(query_id: str, true_label: str, consumer_id: str = "c1_navigation") -> Feedback:
    return Feedback(
        query_id=query_id,
        consumer_id=consumer_id,
        true_label=true_label,
        t_feedback=AT,
        delay_ms=1.0,
    )


class TestTheOrderOfAWindow:
    def test_a_window_comes_back_oldest_first(self, database: MetricsDatabase) -> None:
        for index in range(5):
            database.write_query(answered(index))

        rows = database.read_window("c1_navigation", 500)

        assert [row["query_id"] for row in rows] == [f"q-c1_navigation-{i}" for i in range(5)]

    def test_the_limit_keeps_the_newest_queries_not_the_first_written(
        self, database: MetricsDatabase
    ) -> None:
        for index in (2, 1, 0):
            database.write_query(answered(index))

        rows = database.read_window("c1_navigation", 2)

        assert [row["query_id"] for row in rows] == ["q-c1_navigation-1", "q-c1_navigation-2"]

    def test_the_last_rows_of_a_window_are_the_most_recent(self, database: MetricsDatabase) -> None:
        for index in range(5):
            database.write_query(answered(index))

        window = Window(database.read_window("c1_navigation", 500)).last(2)

        assert [row["query_id"] for row in window] == [
            "q-c1_navigation-3",
            "q-c1_navigation-4",
        ]


class TestWhatAKeyMeans:
    def test_the_same_query_written_twice_is_one_row(self, database: MetricsDatabase) -> None:
        database.write_query(answered(1, predicted_label="car"))
        database.write_query(answered(1, predicted_label="pedestrian"))

        rows = database.read_window("c1_navigation", 500)

        assert len(rows) == 1

    def test_repeated_feedback_does_not_change_the_label(self, database: MetricsDatabase) -> None:
        database.write_query(answered(1))
        database.write_feedback(labeled("q-c1_navigation-1", "car"))
        database.write_feedback(labeled("q-c1_navigation-1", "bus"))

        rows = database.read_window("c1_navigation", 500)

        assert rows[0]["true_label"] == "car"

    def test_one_member_answering_twice_is_one_prediction(self, database: MetricsDatabase) -> None:
        database.write_query(answered(1))
        database.write_member_predictions(
            (
                prediction(query_id="q-c1_navigation-1", total_ms=30.0),
                prediction(query_id="q-c1_navigation-1", total_ms=99.0),
            )
        )

        rows = database.read_window("c1_navigation", 500)

        assert rows[0]["total_ms"] == 30.0


class TestWhatIsJoinedOn:
    def test_feedback_addressed_to_another_consumer_is_not_read_as_this_one(
        self, database: MetricsDatabase
    ) -> None:
        database.write_query(answered(1))
        database.write_feedback(
            labeled("q-c1_navigation-1", "WRONG", consumer_id="c3_surveillance")
        )

        rows = database.read_window("c1_navigation", 500)

        assert rows[0]["true_label"] is None


class TestTheProfilesItAnswers:
    def test_it_answers_the_latest_profile_of_each_instance(
        self, database: MetricsDatabase
    ) -> None:
        database.write_profile(
            profile(mean_ms=42.0, window_start=AT, window_end=AT + timedelta(minutes=5))
        )
        database.write_profile(
            profile(
                mean_ms=50.0,
                window_start=AT + timedelta(minutes=5),
                window_end=AT + timedelta(minutes=10),
            )
        )

        held = database.read_profiles()

        assert [one.mean_ms for one in held] == [50.0]


class TestTheSlowestMember:
    def test_the_timings_are_one_member_s_own_and_add_up(self, database: MetricsDatabase) -> None:
        database.write_query(answered(1))
        database.write_member_predictions(
            (
                prediction(
                    query_id="q-c1_navigation-1",
                    instance_id="i-fast",
                    member=member("b3_v3", "x86_cpu"),
                    fetch_ms=2.0,
                    inference_ms=18.0,
                    total_ms=20.0,
                ),
                prediction(
                    query_id="q-c1_navigation-1",
                    instance_id="i-slow",
                    member=member("b6", "arm_cpu"),
                    fetch_ms=1.0,
                    inference_ms=50.0,
                    total_ms=51.0,
                ),
            )
        )

        row = database.read_window("c1_navigation", 500)[0]

        assert (row["fetch_ms"], row["inference_ms"], row["total_ms"]) == (1.0, 50.0, 51.0)


class TestAMemberThatNeverAnswered:
    def test_a_row_naming_no_copy_is_stored_and_read_back(self, database: MetricsDatabase) -> None:
        database.write_query(answered(1))
        database.write_member_predictions(
            (
                prediction(
                    query_id="q-c1_navigation-1",
                    member=member("b3_v3", "x86_cpu"),
                    instance_id=None,
                    node=None,
                    replica=None,
                    provider=None,
                    number_format=None,
                    cache_hit=None,
                    predicted_label=None,
                    confidence=None,
                    status="timeout",
                    fetch_ms=0.0,
                    inference_ms=0.0,
                    total_ms=1000.0,
                ),
            )
        )

        rows = database.read_member_answers(since=AT, until=AT + timedelta(minutes=1))

        assert [row["status"] for row in rows] == ["timeout"]
        assert rows[0]["instance_id"] is None

    def test_the_slowest_member_of_a_query_is_the_slowest_that_ran(
        self, database: MetricsDatabase
    ) -> None:
        database.write_query(answered(1))
        database.write_member_predictions(
            (
                prediction(
                    query_id="q-c1_navigation-1",
                    member=member("b3_v3", "x86_cpu"),
                    fetch_ms=2.0,
                    inference_ms=18.0,
                    total_ms=21.0,
                ),
                prediction(
                    query_id="q-c1_navigation-1",
                    member=member("b6", "arm_cpu"),
                    instance_id=None,
                    node=None,
                    replica=None,
                    provider=None,
                    number_format=None,
                    cache_hit=None,
                    predicted_label=None,
                    confidence=None,
                    status="timeout",
                    fetch_ms=0.0,
                    inference_ms=0.0,
                    total_ms=1000.0,
                ),
            )
        )

        row = database.read_window("c1_navigation", 500)[0]

        assert (row["total_ms"], row["fetch_ms"], row["inference_ms"]) == (21.0, 2.0, 18.0)


class TestAServiceServingAndProbed:
    def test_one_service_holds_a_row_for_each_of_the_two(self, database: MetricsDatabase) -> None:
        database.write_query(answered(1))
        database.write_member_predictions(
            (
                prediction(
                    query_id="q-c1_navigation-1",
                    instance_id="i-serving",
                    member=member("b3_v3", "x86_cpu"),
                    in_ensemble=True,
                ),
                prediction(
                    query_id="q-c1_navigation-1",
                    instance_id="i-probed",
                    member=member("b3_v3", "x86_cpu"),
                    in_ensemble=False,
                ),
            )
        )

        rows = database.read_member_answers(since=AT, until=AT + timedelta(minutes=1))

        assert sorted(str(row["instance_id"]) for row in rows) == ["i-probed", "i-serving"]

    def test_the_probed_answer_never_times_the_query(self, database: MetricsDatabase) -> None:
        database.write_query(answered(1))
        database.write_member_predictions(
            (
                prediction(
                    query_id="q-c1_navigation-1",
                    member=member("b3_v3", "x86_cpu"),
                    in_ensemble=True,
                    total_ms=21.0,
                ),
                prediction(
                    query_id="q-c1_navigation-1",
                    member=member("b6", "arm_cpu"),
                    in_ensemble=False,
                    fetch_ms=2.0,
                    inference_ms=400.0,
                    total_ms=500.0,
                ),
            )
        )

        row = database.read_window("c1_navigation", 500)[0]

        assert row["total_ms"] == 21.0


class TestTheMemberAnswersItAnswers:
    def test_it_returns_one_row_per_member_with_the_true_label(
        self, database: MetricsDatabase
    ) -> None:
        database.write_query(answered(1))
        database.write_member_predictions(
            (
                prediction(
                    query_id="q-c1_navigation-1",
                    instance_id="i-1",
                    member=member("b3_v3", "x86_cpu"),
                    total_ms=20.0,
                ),
                prediction(
                    query_id="q-c1_navigation-1",
                    instance_id="i-2",
                    member=member("b6", "arm_cpu"),
                    total_ms=51.0,
                ),
            )
        )
        database.write_feedback(labeled("q-c1_navigation-1", "car"))

        rows = database.read_member_answers(since=AT, until=AT + timedelta(minutes=1))

        assert sorted(str(row["instance_id"]) for row in rows) == ["i-1", "i-2"]
        assert {str(row["true_label"]) for row in rows} == {"car"}
        assert sorted(float(str(row["total_ms"])) for row in rows) == [20.0, 51.0]

    def test_a_member_answer_outside_the_window_is_not_returned(
        self, database: MetricsDatabase
    ) -> None:
        database.write_query(answered(1))
        database.write_member_predictions((prediction(query_id="q-c1_navigation-1"),))

        rows = database.read_member_answers(
            since=AT + timedelta(minutes=1), until=AT + timedelta(minutes=2)
        )

        assert rows == []

    def test_an_answer_whose_query_has_no_label_yet_comes_back_without_one(
        self, database: MetricsDatabase
    ) -> None:
        database.write_query(answered(1))
        database.write_member_predictions((prediction(query_id="q-c1_navigation-1"),))

        rows = database.read_member_answers(since=AT, until=AT + timedelta(minutes=1))

        assert len(rows) == 1
        assert rows[0]["true_label"] is None


class TestTheDecisionsAReaderSees:
    def test_the_newest_decisions_come_back_first(self, database: MetricsDatabase) -> None:
        for index in range(5):
            database.write_decision(
                decision(decision_id=f"d-{index}", t=AT + timedelta(seconds=index))
            )

        read = database.read_decisions(3)

        assert [one.decision_id for one in read] == ["d-4", "d-3", "d-2"]

    def test_decisions_of_one_moment_come_back_in_one_order(
        self, database: MetricsDatabase
    ) -> None:
        for consumer_id in ("c1_navigation", "c2_safety", "c3_surveillance"):
            database.write_decision(
                decision(decision_id=f"d-{consumer_id}", consumer_id=consumer_id, t=AT)
            )

        read = database.read_decisions(10)

        assert [one.decision_id for one in read] == [
            "d-c3_surveillance",
            "d-c2_safety",
            "d-c1_navigation",
        ]

    def test_the_count_of_scored_deployments_comes_back_beside_the_candidates(
        self, database: MetricsDatabase
    ) -> None:
        database.write_decision(decision(decision_id="d-1", candidates=19, scored=8))
        database.write_decision(decision(decision_id="d-2", t=AT + timedelta(seconds=1)))

        read = {one.decision_id: one for one in database.read_decisions(2)}

        assert (read["d-1"].candidates, read["d-1"].scored) == (19, 8)
        assert read["d-2"].scored is None

    def test_a_reader_asking_for_none_gets_none(self, database: MetricsDatabase) -> None:
        database.write_decision(decision())

        assert list(database.read_decisions(0)) == []


class TestWhoOwnsTheRun:
    def test_the_first_instance_to_claim_the_run_owns_it(self, database: MetricsDatabase) -> None:
        held = database.claim_run(RunOwner(instance_id="control-plane-1", claimed_at=AT))

        assert held == "control-plane-1"

    def test_a_second_instance_is_told_who_already_owns_it(self, database: MetricsDatabase) -> None:
        database.claim_run(RunOwner(instance_id="control-plane-1", claimed_at=AT))

        held = database.claim_run(RunOwner(instance_id="control-plane-2", claimed_at=AT))

        assert held == "control-plane-1"

    def test_the_owner_claiming_again_still_owns_it(self, database: MetricsDatabase) -> None:
        owner = RunOwner(instance_id="control-plane-1", claimed_at=AT)
        database.claim_run(owner)

        assert database.claim_run(owner) == "control-plane-1"


class TestTheMemberWindowAPolicyReads:
    def member_rows(self, database: MetricsDatabase, count: int) -> None:
        for index in range(count):
            database.write_query(answered(index))
            database.write_member_predictions(
                [
                    prediction(
                        query_id=f"q-c1_navigation-{index}",
                        member=member("b6"),
                        predicted_label="car",
                    ),
                    prediction(
                        query_id=f"q-c1_navigation-{index}",
                        member=member("b5"),
                        predicted_label="bus",
                    ),
                ]
            )

    def test_it_answers_one_row_per_member_per_query(self, database: MetricsDatabase) -> None:
        self.member_rows(database, 3)

        rows = database.read_member_window("c1_navigation", 500)

        assert len(rows) == 6
        assert {str(row["predicted_label"]) for row in rows} == {"car", "bus"}

    def test_the_limit_counts_queries_and_not_rows(self, database: MetricsDatabase) -> None:
        self.member_rows(database, 4)

        rows = database.read_member_window("c1_navigation", 2)

        assert len({str(row["query_id"]) for row in rows}) == 2
        assert len(rows) == 4

    def test_the_limit_keeps_the_newest_queries(self, database: MetricsDatabase) -> None:
        self.member_rows(database, 4)

        rows = database.read_member_window("c1_navigation", 2)

        assert {str(row["query_id"]) for row in rows} == {
            "q-c1_navigation-2",
            "q-c1_navigation-3",
        }

    def test_the_true_label_is_there_once_it_has_arrived(self, database: MetricsDatabase) -> None:
        self.member_rows(database, 2)
        database.write_feedback(labeled("q-c1_navigation-0", "car"))

        rows = database.read_member_window("c1_navigation", 500)

        labels = {str(row["query_id"]): row["true_label"] for row in rows}
        assert labels["q-c1_navigation-0"] == "car"
        assert labels["q-c1_navigation-1"] is None

    def test_another_consumer_s_queries_are_not_in_this_window(
        self, database: MetricsDatabase
    ) -> None:
        self.member_rows(database, 1)
        database.write_query(answered(9, consumer_id="c2_safety"))
        database.write_member_predictions(
            [prediction(query_id="q-c2_safety-9", member=member("b6"))]
        )

        rows = database.read_member_window("c1_navigation", 500)

        assert all(str(row["query_id"]).startswith("q-c1_navigation") for row in rows)

    def test_it_comes_back_in_one_order(self, database: MetricsDatabase) -> None:
        self.member_rows(database, 3)

        once = database.read_member_window("c1_navigation", 500)
        again = database.read_member_window("c1_navigation", 500)

        held = [(str(row["query_id"]), str(row["service_name"])) for row in once]
        assert held == [(str(row["query_id"]), str(row["service_name"])) for row in again]
        assert held == sorted(held)

    def test_a_consumer_with_no_queries_reads_nothing(self, database: MetricsDatabase) -> None:
        assert list(database.read_member_window("c9_nobody", 500)) == []


class TestTheQueriesOfAStretchOfTime:
    def test_it_returns_the_queries_that_arrived_inside_the_stretch(
        self, database: MetricsDatabase
    ) -> None:
        for index in range(6):
            database.write_query(answered(index))

        rows = database.read_served(
            consumer_id="c1_navigation",
            since=AT + timedelta(seconds=2),
            until=AT + timedelta(seconds=4),
        )

        assert [row["query_id"] for row in rows] == ["q-c1_navigation-2", "q-c1_navigation-3"]

    def test_it_returns_only_the_consumer_asked_about(self, database: MetricsDatabase) -> None:
        database.write_query(answered(1))
        database.write_query(answered(1, consumer_id="c2_safety"))

        rows = database.read_served(
            consumer_id="c2_safety", since=AT, until=AT + timedelta(minutes=1)
        )

        assert [row["consumer_id"] for row in rows] == ["c2_safety"]

    def test_each_row_carries_where_its_response_time_went(self, database: MetricsDatabase) -> None:
        database.write_query(answered(1))

        (row,) = database.read_served(
            consumer_id="c1_navigation", since=AT, until=AT + timedelta(minutes=1)
        )

        assert float(str(row["response_time_ms"])) == pytest.approx(174.0)
        assert {"wait_ms", "aggregate_ms", "outcome", "n_members", "n_answered"} <= set(row)


class TestEverythingARunStored:
    def test_it_returns_every_query_of_the_run_oldest_first(
        self, database: MetricsDatabase
    ) -> None:
        for index in (2, 0, 1):
            database.write_query(answered(index))
        database.write_query(answered(0, consumer_id="c2_safety"))

        rows = database.read_rows("queries")

        assert [row["query_id"] for row in rows] == [
            "q-c1_navigation-0",
            "q-c2_safety-0",
            "q-c1_navigation-1",
            "q-c1_navigation-2",
        ]

    def test_it_returns_every_member_answer_and_label_and_decision(
        self, database: MetricsDatabase
    ) -> None:
        database.write_query(answered(0))
        database.write_member_predictions(
            (prediction(query_id="q-c1_navigation-0", member=member("b6", "x86_cpu")),)
        )
        database.write_feedback(labeled("q-c1_navigation-0", "car"))
        database.write_decision(decision())

        assert len(database.read_rows("member_predictions")) == 1
        assert [row["true_label"] for row in database.read_rows("feedback")] == ["car"]
        assert len(database.read_rows("decisions")) == 1

    def test_a_table_the_store_does_not_hold_is_refused(self, database: MetricsDatabase) -> None:
        with pytest.raises(CoverageError, match="nowhere"):
            database.read_rows("nowhere")
