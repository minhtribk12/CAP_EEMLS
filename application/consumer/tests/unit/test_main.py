from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eemls.config.reading import ConfigError
from eemls.monitoring.metrics import Monitoring
from eemls_consumer.config import load_consumer_config
from eemls_consumer.main import (
    label_delay_queries,
    main,
    metrics_app,
    result_timeout_s,
    run,
    schedule_of,
)
from support.consumer import Moment, Stub, advancing, built, label_count

ROOT = Path(__file__).resolve().parents[4]
SCENARIO = {
    "day_length_s": 60.0,
    "periods": [
        {
            "name": "morning",
            "starts_at_s": 0.0,
            "class_mix": {"pedestrian": 1.0},
            "condition": "day",
            "weather": "clear",
        }
    ],
    "events": [{"starts_at_s": 10.0, "ends_at_s": 20.0, "image_quality": "blurred"}],
}


def scenario_file(root: Path, held: object = None) -> Path:
    path = root / "scenario.yaml"
    path.write_text(yaml.safe_dump(SCENARIO if held is None else held), encoding="utf-8")
    return path


class TestTheScenario:
    def test_the_run_takes_its_rate_from_the_contract(self, tmp_path: Path) -> None:
        held = schedule_of(scenario_file(tmp_path), request_rate_per_s=20.0)

        assert held.request_rate_per_s == 20.0
        assert held.periods[0].name == "morning"
        assert held.image_quality_at(15.0) == "blurred"

    def test_a_file_that_is_not_a_scenario_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "scenario.yaml"
        path.write_text("- not a scenario\n", encoding="utf-8")

        with pytest.raises(ConfigError, match="not a mapping"):
            schedule_of(path, request_rate_per_s=20.0)


class TestTheFeedbackDelay:
    def test_the_delay_in_intervals_becomes_a_delay_in_queries(self) -> None:
        config = load_consumer_config(ROOT / "config")
        contract = config.contracts["c2_safety"]

        assert label_delay_queries(contract, config.services.consumer) == 75

    def test_the_drain_outlasts_the_wait_for_a_result(self) -> None:
        config = load_consumer_config(ROOT / "config")

        assert result_timeout_s(config) == pytest.approx(2.0)


class TestTheLoop:
    def test_it_drives_the_client_until_the_run_is_over(self) -> None:
        stub = Stub()
        moment = Moment()
        client, _, _ = built(stub, rate=20.0, send_window=1000, moment=moment)
        waits: list[float] = []

        answered = run(
            client,
            duration_s=2.0,
            result_timeout_s=0.0,
            interval_s=0.01,
            sleeper=advancing(moment, 0.5, waits),
        )

        assert answered == len(stub.queries) == 30
        assert waits == [0.01, 0.01, 0.01, 0.01]

    def test_it_asks_nothing_more_while_it_drains_what_is_waiting(self) -> None:
        stub = Stub()
        moment = Moment()
        client, _, _ = built(stub, rate=20.0, send_window=1000, fraction=1.0, moment=moment)

        run(
            client,
            duration_s=1.0,
            result_timeout_s=20.0,
            interval_s=0.01,
            sleeper=advancing(moment, 0.5),
        )

        assert len(stub.queries) == 10
        assert [label_count(one) for one in stub.returned] == [10]

    def test_a_run_of_no_length_asks_nothing(self) -> None:
        stub = Stub()
        client, _, _ = built(stub)

        assert run(client, duration_s=0.0, result_timeout_s=0.0, interval_s=0.01) == 0
        assert stub.queries == []


class TestTheMetricsEndpoint:
    def test_the_client_serves_the_configured_metrics_path(self) -> None:
        settings = load_consumer_config(ROOT / "config").services.monitoring

        app = metrics_app(Monitoring.from_config(settings))

        paths = {getattr(route, "path", "") for route in app.routes}
        assert settings.metrics_path in paths


class TestTheCommand:
    def test_a_consumer_the_configuration_does_not_name_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "--config",
                str(ROOT / "config"),
                "--consumer",
                "c9_nobody",
                "--scenario",
                str(scenario_file(tmp_path)),
                "--duration-s",
                "1",
                "--seed",
                "0",
            ]
        )

        assert code == 1
        assert "c9_nobody" in capsys.readouterr().err

    def test_a_client_without_a_seed_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            main(
                [
                    "--config",
                    str(ROOT / "config"),
                    "--consumer",
                    "c9_nobody",
                    "--scenario",
                    str(scenario_file(tmp_path)),
                    "--duration-s",
                    "1",
                ]
            )

    def test_a_scenario_that_is_not_one_is_refused_before_anything_is_opened(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "scenario.yaml"
        path.write_text("periods: []\n", encoding="utf-8")

        code = main(
            [
                "--config",
                str(ROOT / "config"),
                "--consumer",
                "c2_safety",
                "--scenario",
                str(path),
                "--duration-s",
                "1",
                "--seed",
                "0",
            ]
        )

        assert code == 1
        assert "refused" in capsys.readouterr().err


class Listening:
    def __init__(self, steps: list[str], *, granted: bool) -> None:
        self.steps = steps
        self.granted = granted

    def start(self) -> None:
        self.steps.append("subscribing")

    def wait_until_ready(self, timeout_s: float) -> bool:
        self.steps.append("subscribed" if self.granted else "not subscribed")
        return self.granted

    def close(self) -> None:
        self.steps.append("closed")


class Closing:
    def close(self) -> None:
        pass


class TestTheResultSubscription:
    def started(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, granted: bool
    ) -> tuple[int, list[str]]:
        steps: list[str] = []

        def sending(*_: object, **__: object) -> int:
            steps.append("sending")
            return 0

        monkeypatch.setattr(
            "eemls_consumer.main.build_consumer",
            lambda *_, **__: (Closing(), Listening(steps, granted=granted), None),
        )
        monkeypatch.setattr("eemls_consumer.main.start_metrics_server", lambda *_: None)
        monkeypatch.setattr("eemls_consumer.main.run", sending)
        code = main(
            [
                "--config",
                str(ROOT / "config"),
                "--consumer",
                "c2_safety",
                "--scenario",
                str(scenario_file(tmp_path)),
                "--duration-s",
                "1",
                "--seed",
                "0",
            ]
        )
        return code, steps

    def test_no_query_is_sent_before_the_subscription_is_granted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        code, steps = self.started(tmp_path, monkeypatch, granted=True)

        assert code == 0
        assert steps == ["subscribing", "subscribed", "sending", "closed"]

    def test_a_subscription_never_granted_is_refused_before_any_query(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, steps = self.started(tmp_path, monkeypatch, granted=False)

        assert code == 1
        assert "sending" not in steps
        assert "subscription" in capsys.readouterr().err
