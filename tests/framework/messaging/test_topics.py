from __future__ import annotations

import pytest

from eemls.messaging.topics import (
    TopicError,
    instance_topic,
    instances_of_run,
    parts_of,
    result_topic,
)

PREFIX = "eemls/results"


class TestBuildingOne:
    def test_a_result_topic_names_the_consumer_and_then_the_client(self) -> None:
        held = result_topic(prefix=PREFIX, consumer_id="c2_safety", client_id="client-1")

        assert held == "eemls/results/c2_safety/client-1"

    @pytest.mark.parametrize("name", ["", "c2/safety", "c2+safety", "c2#safety", " c2", "c2 "])
    def test_a_name_no_topic_can_carry_is_refused(self, name: str) -> None:
        with pytest.raises(TopicError):
            result_topic(prefix=PREFIX, consumer_id=name, client_id="client-1")

    @pytest.mark.parametrize("name", ["", "client/1", "client+1", "client#1"])
    def test_a_client_name_no_topic_can_carry_is_refused(self, name: str) -> None:
        with pytest.raises(TopicError):
            result_topic(prefix=PREFIX, consumer_id="c2_safety", client_id=name)

    @pytest.mark.parametrize("prefix", ["", "eemls/results/", "/eemls/results", "eemls/+"])
    def test_a_prefix_no_topic_can_carry_is_refused(self, prefix: str) -> None:
        with pytest.raises(TopicError):
            result_topic(prefix=prefix, consumer_id="c2_safety", client_id="client-1")


class TestReadingOne:
    def test_it_reads_back_the_consumer_and_the_client(self) -> None:
        assert parts_of("eemls/results/c2_safety/client-1", prefix=PREFIX) == (
            "c2_safety",
            "client-1",
        )

    def test_a_topic_under_another_prefix_is_refused(self) -> None:
        with pytest.raises(TopicError, match="is not under"):
            parts_of("other/results/c2_safety/client-1", prefix=PREFIX)

    def test_a_topic_naming_one_part_too_few_is_refused(self) -> None:
        with pytest.raises(TopicError, match="names 1 part"):
            parts_of("eemls/results/c2_safety", prefix=PREFIX)

    def test_a_topic_naming_one_part_too_many_is_refused(self) -> None:
        with pytest.raises(TopicError, match="names 3 parts"):
            parts_of("eemls/results/c2_safety/client-1/again", prefix=PREFIX)


class TestTheTopicsOfTheCopies:
    def test_a_copy_s_topic_names_the_run_and_then_the_copy(self) -> None:
        held = instance_topic(prefix="eemls/instances", run_id="r1", instance_id="infer-1")

        assert held == "eemls/instances/r1/infer-1"

    def test_the_control_plane_hears_every_copy_of_its_run(self) -> None:
        assert instances_of_run(prefix="eemls/instances", run_id="r1") == "eemls/instances/r1/+"

    @pytest.mark.parametrize("name", ["", "infer/1", "infer+1", "infer#1"])
    def test_a_copy_name_no_topic_can_carry_is_refused(self, name: str) -> None:
        with pytest.raises(TopicError):
            instance_topic(prefix="eemls/instances", run_id="r1", instance_id=name)
