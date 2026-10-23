from __future__ import annotations

from pathlib import Path

import pytest

from eemls.testing.records import profile
from search_space import main

FIVE_MODELS_ON_TWO_DEVICE_TYPES = (
    "--throughput",
    "m1=3.1,10",
    "--throughput",
    "m2=4,12",
    "--throughput",
    "m3=8,20",
    "--throughput",
    "m4=12,30",
    "--throughput",
    "m5=35.2,35.2",
)


def table(capsys: pytest.CaptureFixture[str], *argv: str) -> dict[str, list[str]]:
    assert main(list(argv)) == 0
    lines = capsys.readouterr().out.splitlines()
    header = lines[0].split("|")
    return {
        row.split("|")[0].strip(): [cell.strip() for cell in row.split("|")]
        for row in lines[2:]
        if "|" in row
    } | {"header": [cell.strip() for cell in header]}


def column(held: dict[str, list[str]], rate: str, name: str) -> str:
    return held[rate][held["header"].index(name)]


def test_one_request_per_second_gives_25_ensembles(capsys: pytest.CaptureFixture[str]) -> None:
    held = table(capsys, *FIVE_MODELS_ON_TWO_DEVICE_TYPES, "--rates", "1")

    assert column(held, "1", "ensembles") == "25"


def test_one_request_per_second_gives_130_deployments(capsys: pytest.CaptureFixture[str]) -> None:
    held = table(capsys, *FIVE_MODELS_ON_TWO_DEVICE_TYPES, "--rates", "1")

    assert column(held, "1", "naive") == "130"
    assert column(held, "1", "scale reduction") == "130"


def test_100_requests_per_second_need_3_to_33_replicas(
    capsys: pytest.CaptureFixture[str],
) -> None:
    held = table(capsys, *FIVE_MODELS_ON_TWO_DEVICE_TYPES, "--rates", "100")

    assert column(held, "100", "replicas") == "3 to 33"


def test_cap_weighs_one_candidate_per_inference_service(
    capsys: pytest.CaptureFixture[str],
) -> None:
    held = table(capsys, *FIVE_MODELS_ON_TWO_DEVICE_TYPES, "--rates", "100")

    assert column(held, "100", "cap per action") == "10"


def test_the_spaces_are_printed_for_every_rate_of_figure_9(
    capsys: pytest.CaptureFixture[str],
) -> None:
    held = table(capsys, *FIVE_MODELS_ON_TWO_DEVICE_TYPES)

    assert [rate for rate in held if rate != "header"] == ["1", "5", "10", "20", "50", "100"]


def test_the_sizes_are_shown_in_powers_of_ten(capsys: pytest.CaptureFixture[str]) -> None:
    held = table(capsys, *FIVE_MODELS_ON_TWO_DEVICE_TYPES, "--rates", "100")

    assert column(held, "100", "log10 naive") == "21.4"
    assert column(held, "100", "log10 scale reduction") == "6.4"


def test_the_committed_profiles_are_read_without_throughputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profiles = tmp_path / "profiles.jsonl"
    profiles.write_text(
        "".join(
            profile(
                service_name=f"infer-{model}", model=model, throughput_rps=10.0
            ).model_dump_json()
            + "\n"
            for model in ("m1", "m2")
        ),
        encoding="utf-8",
    )

    held = table(capsys, "--profiles", str(profiles), "--device-types", "2", "--rates", "20")

    assert column(held, "20", "naive") == str(2**2 + 2**2 + 2**4)


def test_a_throughput_without_a_value_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--throughput", "m1="]) == 2
    assert "m1" in capsys.readouterr().err
