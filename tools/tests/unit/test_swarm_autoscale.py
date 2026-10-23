from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eemls.config.models import Autoscale
from eemls.testing.trees import committed_tree
from swarm_autoscale import Scaling, cpu_use_of, main, replicas_of, scaled

SETTINGS = Autoscale(enabled=True, max_replicas=4, cpu_percent=70, cooldown_s=60)
STATS = "\n".join(
    [
        '{"Name":"eemls_infer-a.1.x1","CPUPerc":"35.00%"}',
        '{"Name":"eemls_infer-a.2.x2","CPUPerc":"70.00%"}',
        '{"Name":"eemls_infer-b.1.x3","CPUPerc":"0.50%"}',
        '{"Name":"eemls_control-plane.1.x4","CPUPerc":"90.00%"}',
        '{"Name":"other_infer-a.1.x5","CPUPerc":"99.00%"}',
    ]
)


def test_use_is_the_percent_of_each_copy_request_per_service() -> None:
    use = cpu_use_of(STATS, "eemls", {"infer-a", "infer-b"}, cpu_cores=0.5)
    assert use == {"infer-a": [70.0, 140.0], "infer-b": [1.0]}


def test_replicas_are_the_count_each_service_asks_for() -> None:
    listing = "eemls_infer-a 1/3\neemls_infer-b 2/2 (max 1 per node)\nother_infer-a 5/5"
    assert replicas_of(listing, "eemls") == {"infer-a": 3, "infer-b": 2}


def test_high_use_adds_copies_at_once() -> None:
    scaling = scaled(2, [140.0, 140.0], minimum=1, settings=SETTINGS, previous=Scaling(), now=0.0)
    assert scaling.replicas == 4
    assert scaling.low_since is None


def test_copies_never_pass_max_replicas() -> None:
    scaling = scaled(3, [300.0] * 3, minimum=1, settings=SETTINGS, previous=Scaling(), now=0.0)
    assert scaling.replicas == 4


def test_low_use_waits_for_the_cooldown_before_removing_copies() -> None:
    first = scaled(3, [10.0] * 3, minimum=1, settings=SETTINGS, previous=Scaling(), now=100.0)
    assert (first.replicas, first.low_since) == (3, 100.0)
    early = scaled(3, [10.0] * 3, minimum=1, settings=SETTINGS, previous=first, now=159.0)
    assert (early.replicas, early.low_since) == (3, 100.0)
    late = scaled(3, [10.0] * 3, minimum=1, settings=SETTINGS, previous=early, now=160.0)
    assert (late.replicas, late.low_since) == (1, None)


def test_use_rising_again_restarts_the_cooldown() -> None:
    low = scaled(3, [10.0] * 3, minimum=1, settings=SETTINGS, previous=Scaling(), now=0.0)
    busy = scaled(3, [70.0] * 3, minimum=1, settings=SETTINGS, previous=low, now=30.0)
    assert busy == Scaling(replicas=3, low_since=None)
    again = scaled(3, [10.0] * 3, minimum=1, settings=SETTINGS, previous=busy, now=61.0)
    assert (again.replicas, again.low_since) == (3, 61.0)


def test_copies_never_drop_below_the_service_replicas() -> None:
    low = scaled(4, [0.0] * 4, minimum=2, settings=SETTINGS, previous=Scaling(), now=0.0)
    done = scaled(4, [0.0] * 4, minimum=2, settings=SETTINGS, previous=low, now=60.0)
    assert done.replicas == 2


def test_a_service_below_its_replicas_is_raised_to_them() -> None:
    scaling = scaled(1, [0.0], minimum=2, settings=SETTINGS, previous=Scaling(), now=0.0)
    assert scaling.replicas == 2


def test_refuses_to_start_when_autoscaling_is_off(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tree = committed_tree(tmp_path / "config")
    path = tree / "models" / "inference_services.yaml"
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body["autoscale"]["enabled"] = False
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    assert main(["--config", str(tree)]) == 1
    assert "autoscale.enabled is false" in capsys.readouterr().err
