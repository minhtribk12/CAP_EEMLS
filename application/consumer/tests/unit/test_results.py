from __future__ import annotations

import pytest

from eemls_consumer.results import Pending, ResultError


class TestAMatch:
    def test_a_result_matches_its_request_and_gives_the_round_trip(self) -> None:
        pending = Pending(timeout_ms=2000.0)
        pending.sent(correlation_id="corr-1", at_s=10.0)

        held = pending.arrived(correlation_id="corr-1", at_s=10.025)

        assert held.round_trip is not None
        assert held.outcome == "matched"
        assert held.round_trip.round_trip_ms == pytest.approx(25.0)

    def test_a_matched_request_is_no_longer_in_flight(self) -> None:
        pending = Pending(timeout_ms=2000.0)
        pending.sent(correlation_id="corr-1", at_s=10.0)

        pending.arrived(correlation_id="corr-1", at_s=10.01)

        assert pending.in_flight == 0

    def test_one_correlation_id_cannot_name_two_requests(self) -> None:
        pending = Pending(timeout_ms=2000.0)
        pending.sent(correlation_id="corr-1", at_s=10.0)

        with pytest.raises(ResultError, match="corr-1"):
            pending.sent(correlation_id="corr-1", at_s=10.0)


class TestARepeat:
    def test_a_second_delivery_of_one_result_is_dropped(self) -> None:
        pending = Pending(timeout_ms=2000.0)
        pending.sent(correlation_id="corr-1", at_s=10.0)
        pending.arrived(correlation_id="corr-1", at_s=10.01)

        again = pending.arrived(correlation_id="corr-1", at_s=10.02)

        assert (again.outcome, again.round_trip) == ("duplicate", None)

    def test_a_result_for_a_request_never_sent_is_counted_apart(self) -> None:
        pending = Pending(timeout_ms=2000.0)

        held = pending.arrived(correlation_id="corr-9", at_s=10.0)

        assert held.outcome == "unknown"

    def test_it_remembers_only_the_last_matched_ids(self) -> None:
        pending = Pending(timeout_ms=2000.0, remember=2)
        for number in range(3):
            pending.sent(correlation_id=f"corr-{number}", at_s=10.0)
            pending.arrived(correlation_id=f"corr-{number}", at_s=10.01)

        assert pending.arrived(correlation_id="corr-0", at_s=10.02).outcome == "unknown"


class TestATimeout:
    def test_a_request_with_no_result_inside_the_timeout_is_lost(self) -> None:
        pending = Pending(timeout_ms=2000.0)
        pending.sent(correlation_id="corr-1", at_s=10.0)

        expired = pending.expired(now_s=12.001)

        assert expired == ("corr-1",)
        assert pending.in_flight == 0

    def test_a_request_still_inside_the_timeout_is_not_lost(self) -> None:
        pending = Pending(timeout_ms=2000.0)
        pending.sent(correlation_id="corr-1", at_s=10.0)

        assert pending.expired(now_s=11.5) == ()
        assert pending.in_flight == 1

    def test_a_result_that_arrives_after_the_timeout_is_counted_late(self) -> None:
        pending = Pending(timeout_ms=2000.0)
        pending.sent(correlation_id="corr-1", at_s=10.0)
        pending.expired(now_s=12.001)

        late = pending.arrived(correlation_id="corr-1", at_s=12.5)

        assert late.outcome == "late"

    def test_a_result_arriving_at_the_timeout_itself_is_still_matched(self) -> None:
        pending = Pending(timeout_ms=2000.0)
        pending.sent(correlation_id="corr-1", at_s=10.0)

        assert pending.expired(now_s=12.0) == ()
        assert pending.arrived(correlation_id="corr-1", at_s=12.0).outcome == "matched"


class TestARequestThatNeverLeft:
    def test_a_withdrawn_request_is_never_counted_lost(self) -> None:
        pending = Pending(timeout_ms=2000.0)
        pending.sent(correlation_id="corr-1", at_s=10.0)

        pending.withdrawn(correlation_id="corr-1")

        assert (pending.in_flight, pending.expired(now_s=99.0)) == (0, ())

    def test_a_request_withdrawn_twice_is_no_worse_than_once(self) -> None:
        pending = Pending(timeout_ms=2000.0)
        pending.sent(correlation_id="corr-1", at_s=10.0)

        pending.withdrawn(correlation_id="corr-1")
        pending.withdrawn(correlation_id="corr-1")

        assert pending.in_flight == 0
