from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from eemls.testing.trees import committed_tree
from support.deployment import copied_scripts, deploy_options

TOKEN = "K10test0000000000000000::server:0000000000"  # noqa: S105

FAKE = f"""#!/bin/sh
here="$(dirname "$0")"
printf '%s\\n' "$(basename "$0")" "$@" > "$here/calls/$$.args"
cat > "$here/calls/$$.in"
case "$*" in *node-token*) echo "{TOKEN}" ;; esac
"""
# poetry runs eemls-deploy here and refuses options before a later `--`.
DEPLOY = f"""#!/bin/sh
shift
if [ "$1" = -- ]; then
  shift
else
  case " $* " in *" -- "*)
    for arg; do
      case "$arg" in
        --) break ;;
        --*) echo "The option \\"$arg\\" does not exist" >&2; exit 1 ;;
      esac
    done ;;
  esac
fi
shift
exec {sys.executable} -m eemls_tools.deploy_command "$@"
"""


def ran(root: Path, script: str, *args: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    bin_dir = root / "bin"
    (bin_dir / "calls").mkdir(parents=True)
    for tool in ("ssh", "kubectl"):
        (bin_dir / tool).write_text(FAKE, encoding="utf-8")
    (bin_dir / "poetry").write_text(DEPLOY, encoding="utf-8")
    for tool in ("ssh", "kubectl", "poetry"):
        (bin_dir / tool).chmod(0o755)
    copied_scripts(root, "kube")
    committed_tree(root / "config")
    done = subprocess.run(
        ["sh", f"deploy/kube/{script}", *args],
        cwd=root,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    return done, bin_dir / "calls"


def calls(folder: Path) -> list[tuple[list[str], str]]:
    return [
        (
            path.read_text(encoding="utf-8").splitlines(),
            path.with_suffix(".in").read_text(encoding="utf-8"),
        )
        for path in sorted(folder.glob("*.args"))
    ]


class TestJoiningAnAgent:
    def test_the_join_token_reaches_the_agent_on_its_input_only(self, tmp_path: Path) -> None:
        done, folder = ran(tmp_path, "k3s/agent.sh", "me@agent", "me@server", "10.0.0.1")
        made = calls(folder)

        assert done.returncode == 0, done.stderr
        assert TOKEN not in done.stdout + done.stderr
        assert all(TOKEN not in " ".join(args) for args, _ in made)
        assert [args[1] for args, given in made if TOKEN in given] == ["me@agent"]

    def test_the_agent_joins_the_server_it_is_given(self, tmp_path: Path) -> None:
        _, folder = ran(tmp_path, "k3s/agent.sh", "me@agent", "me@server", "10.0.0.1")
        [(args, _)] = [one for one in calls(folder) if one[0][1] == "me@agent"]

        assert "https://10.0.0.1:6443" in " ".join(args)


class TestLabelingANode:
    def test_a_node_offers_its_device_type(self, tmp_path: Path) -> None:
        done, folder = ran(tmp_path, "label.sh", "pi-1", "raspberry_pi4")

        assert done.returncode == 0, done.stderr
        assert calls(folder)[0][0] == [
            "kubectl",
            "label",
            "node",
            "pi-1",
            "eemls.device.raspberry_pi4=true",
            "--overwrite",
        ]

    def test_a_core_node_also_runs_the_core_services(self, tmp_path: Path) -> None:
        _, folder = ran(tmp_path, "label.sh", "dell", "precision_5820", "core")

        assert calls(folder)[0][0][4:] == [
            "eemls.device.precision_5820=true",
            "eemls.role.core=true",
            "--overwrite",
        ]

    def test_a_device_type_no_price_names_is_refused(self, tmp_path: Path) -> None:
        done, folder = ran(tmp_path, "label.sh", "pi-1", "toaster")

        assert done.returncode == 1
        assert "toaster is not a device type of config/cost/prices.yaml" in done.stderr
        assert calls(folder) == []


KUBECONFIG_HERE = "server: https://127.0.0.1:6443\n"


def served_kubeconfig(root: Path, answer: str) -> Path:
    bin_dir = root / "bin"
    (bin_dir / "calls").mkdir(parents=True)
    (bin_dir / "ssh").write_text(
        FAKE.replace(f'echo "{TOKEN}"', f"printf '{answer}'").replace("node-token", "k3s.yaml"),
        encoding="utf-8",
    )
    (bin_dir / "ssh").chmod(0o755)
    copied_scripts(root, "kube")
    return bin_dir


def server(root: Path, bin_dir: Path, out: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sh", "deploy/kube/k3s/server.sh", "me@server", "10.0.0.1", str(out)],
        cwd=root,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )


class TestStartingTheServer:
    def test_the_kubeconfig_reaches_the_server_at_its_address_and_is_the_owner_s_only(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "kube" / "eemls.yaml"

        done = server(tmp_path, served_kubeconfig(tmp_path, KUBECONFIG_HERE), out)

        assert done.returncode == 0, done.stderr
        assert out.read_text(encoding="utf-8") == "server: https://10.0.0.1:6443\n"
        assert out.stat().st_mode & 0o777 == 0o600
        assert "6443" not in done.stdout

    def test_the_server_s_certificate_names_its_address(self, tmp_path: Path) -> None:
        bin_dir = served_kubeconfig(tmp_path, KUBECONFIG_HERE)
        server(tmp_path, bin_dir, tmp_path / "eemls.yaml")

        installed = [args for args, _ in calls(bin_dir / "calls") if "get.k3s.io" in " ".join(args)]

        assert "--tls-san 10.0.0.1" in " ".join(installed[0])

    def test_an_empty_kubeconfig_is_refused_and_the_old_one_kept(self, tmp_path: Path) -> None:
        out = tmp_path / "eemls.yaml"
        out.write_text("old\n", encoding="utf-8")

        done = server(tmp_path, served_kubeconfig(tmp_path, ""), out)

        assert done.returncode == 1
        assert "no kubeconfig came back" in done.stdout
        assert out.read_text(encoding="utf-8") == "old\n"


def a_run_up(root: Path, port: int) -> None:
    services = root / "runs" / "kube" / "r1" / "config" / "services"
    services.mkdir(parents=True)
    (services / "inference_service.yaml").write_text(f"port: {port}\n", encoding="utf-8")
    (root / "runs" / "kube" / "current").write_text("r1\n", encoding="utf-8")


class TestTheBurst:
    def test_a_pod_in_the_cluster_sends_the_service_its_predictions(self, tmp_path: Path) -> None:
        a_run_up(tmp_path, 8082)
        done, folder = ran(tmp_path, "burst.sh", "infer-b1-general-pi4", "img-1", "0", "4", "0")
        [started] = [args for args, _ in calls(folder) if "run" in args]
        sent = " ".join(started)

        assert done.returncode == 0, done.stderr
        assert started[1:5] == ["-n", "eemls", "run", "eemls-burst"]
        assert "http://infer-b1-general-pi4:8082/predict" in sent
        assert '{"image_id": "img-1"}' in sent

    def test_it_sends_to_the_port_of_the_run_s_own_config(self, tmp_path: Path) -> None:
        a_run_up(tmp_path, 9099)
        _, folder = ran(tmp_path, "burst.sh", "infer-b1-general-pi4", "img-1", "0", "4", "0")
        [started] = [args for args, _ in calls(folder) if "run" in args]

        assert "http://infer-b1-general-pi4:9099/predict" in " ".join(started)

    def test_it_prints_the_copies_of_the_service_through_the_burst(self, tmp_path: Path) -> None:
        a_run_up(tmp_path, 8082)
        _, folder = ran(tmp_path, "burst.sh", "infer-b1-general-pi4", "img-1", "0", "4", "0")
        asked = [args for args, _ in calls(folder) if "hpa" in args]

        assert asked
        assert all("infer-b1-general-pi4" in args for args in asked)


PLANNED = "cap 0 exp-cap-s0 recurring/cap/seed-0 22 recurring"


class TestAnExperimentRunAgain:
    def test_each_run_starts_without_the_last_run_s_folder(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        poetry = (
            f'#!/bin/sh\ncase "$*" in *"eemls-exp plan"*) echo "{PLANNED}" ;; esac\n'
            'while [ $# -gt 0 ]; do [ "$1" = --out ] && mkdir -p "$2"; shift; done\n'
        )
        (bin_dir / "poetry").write_text(poetry, encoding="utf-8")
        (bin_dir / "poetry").chmod(0o755)
        copied_scripts(tmp_path, "kube")
        (tmp_path / "deploy" / "kube" / "up.sh").write_text(
            '[ -e "runs/kube/$RUN_ID" ] && echo held > seen || echo clean > seen\n'
            'mkdir -p "runs/kube/$RUN_ID/rows"\n',
            encoding="utf-8",
        )
        for script in ("run.sh", "down.sh"):
            (tmp_path / "deploy" / "kube" / script).write_text("", encoding="utf-8")
        (tmp_path / "runs" / "kube" / "exp-cap-s0" / "here").mkdir(parents=True)

        done = subprocess.run(
            ["sh", "deploy/kube/experiment.sh"],
            cwd=tmp_path,
            env={
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "EEMLS_DATABASE_PASSWORD": "test",
                "PLAN": "experiments/runs/recurring_ten_days.yaml",
                "OUT": "out",
            },
            capture_output=True,
            text=True,
            check=False,
        )

        assert done.returncode == 0, done.stderr
        assert (tmp_path / "seen").read_text(encoding="utf-8") == "clean\n"
        assert (tmp_path / "out" / "recurring" / "cap" / "seed-0" / "rows").is_dir()
        assert not (tmp_path / "runs" / "kube" / "exp-cap-s0").exists()


class TestTheDatabaseForward:
    def test_a_forward_that_cannot_listen_stops_the_script_with_its_reason(
        self, tmp_path: Path
    ) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "kubectl").write_text(
            "#!/bin/sh\necho 'bind: address already in use' >&2\nexit 1\n", encoding="utf-8"
        )
        (bin_dir / "kubectl").chmod(0o755)
        copied_scripts(tmp_path, "kube")
        (tmp_path / "out").mkdir()

        done = subprocess.run(
            ["sh", "-c", ". deploy/kube/common.sh; OUT=out; forward postgres; echo forwarded"],
            cwd=tmp_path,
            env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
            capture_output=True,
            text=True,
            check=False,
        )

        assert done.returncode != 0
        assert "forwarded" not in done.stdout
        assert "address already in use" in done.stdout


FORWARDING = (
    '#!/bin/sh\nhere="$(dirname "$0")"\necho "$*" >> "$here/kubectl.args"\n'
    'case "$*" in\n'
    '  *port-forward*) echo $$ > "$here/forward.pid"; exec sleep 30 ;;\n'
    '  *"wait --for=condition=available"*) exit 1 ;;\n'
    "esac\n"
)


def forwarding(root: Path, poetry: str) -> Path:
    bin_dir = root / "bin"
    bin_dir.mkdir()
    (bin_dir / "poetry").write_text(poetry, encoding="utf-8")
    (bin_dir / "kubectl").write_text(FORWARDING, encoding="utf-8")
    for tool in ("poetry", "kubectl"):
        (bin_dir / tool).chmod(0o755)
    copied_scripts(root, "kube")
    return bin_dir


def stopped(pid_file: Path) -> bool:
    pid = int(pid_file.read_text(encoding="utf-8"))
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        try:
            status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        except (FileNotFoundError, ProcessLookupError):
            return True
        if "State:\tZ" in status:
            return True
        time.sleep(0.1)
    return False


class TestAFailureWhileTheDatabaseIsForwarded:
    def test_a_failed_start_removes_what_it_made_and_stops_the_forward(
        self, tmp_path: Path
    ) -> None:
        bin_dir = forwarding(tmp_path, "#!/bin/sh\n")

        done = subprocess.run(
            ["sh", "deploy/kube/up.sh"],
            cwd=tmp_path,
            env={
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "EEMLS_DATABASE_PASSWORD": "test",
                "RUN_ID": "r1",
            },
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

        assert done.returncode != 0
        calls = (bin_dir / "kubectl.args").read_text(encoding="utf-8")
        assert "-n eemls delete pvc metrics-database-r1 --ignore-not-found" in calls
        assert "run.yaml" not in calls.split("apply -f runs/kube/r1/kube/run.yaml", 1)[1]
        assert not (tmp_path / "runs" / "kube" / "current").exists()
        assert stopped(bin_dir / "forward.pid")

    def test_a_failed_collect_stops_the_forward(self, tmp_path: Path) -> None:
        bin_dir = forwarding(tmp_path, '#!/bin/sh\ncase "$*" in *eemls-collect*) exit 1 ;; esac\n')
        run = tmp_path / "runs" / "kube" / "r1"
        run.mkdir(parents=True)
        (tmp_path / "runs" / "kube" / "current").write_text("r1\n", encoding="utf-8")
        for name in ("profiles", "images", "seed"):
            (run / name).write_text("0\n", encoding="utf-8")

        done = subprocess.run(
            ["sh", "deploy/kube/run.sh"],
            cwd=tmp_path,
            env={
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "EEMLS_DATABASE_PASSWORD": "test",
                "DURATION_S": "1",
            },
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

        assert done.returncode != 0
        assert stopped(bin_dir / "forward.pid")


NAMING = (
    '#!/bin/sh\ncase "$*" in\n'
    "  *control_plane*) echo control-plane ;;\n"
    "  *observation_service*) echo observation-service ;;\n"
    "  *metrics_database*) echo metrics-database ;;\n"
    "  *data_hub*) echo data-hub ;;\n"
    "  *broker*) echo broker ;;\n"
    "esac\n"
)


class TestTheEndOfARun:
    def test_run_stops_the_control_plane_then_the_observation_service_then_collects(
        self, tmp_path: Path
    ) -> None:
        bin_dir = forwarding(tmp_path, NAMING)
        run = tmp_path / "runs" / "kube" / "r1"
        run.mkdir(parents=True)
        (tmp_path / "runs" / "kube" / "current").write_text("r1\n", encoding="utf-8")
        for name in ("profiles", "images", "seed"):
            (run / name).write_text("0\n", encoding="utf-8")
        (run / "kube" / "inference").mkdir(parents=True)

        subprocess.run(
            ["sh", "deploy/kube/run.sh"],
            cwd=tmp_path,
            env={
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "EEMLS_DATABASE_PASSWORD": "test",
                "DURATION_S": "1",
            },
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

        held = (bin_dir / "kubectl.args").read_text(encoding="utf-8").splitlines()
        order = [
            next(at for at, line in enumerate(held) if part in line)
            for part in (
                "delete deploy control-plane",
                "delete -f runs/kube/r1/kube/inference",
                "delete deploy observation-service",
                "port-forward",
            )
        ]
        assert order == sorted(order)

    def test_down_stops_each_service_after_the_ones_that_send_to_it(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "poetry").write_text(NAMING, encoding="utf-8")
        (bin_dir / "kubectl").write_text(
            '#!/bin/sh\necho "$*" >> "$(dirname "$0")/kubectl.args"\n', encoding="utf-8"
        )
        for tool in ("poetry", "kubectl"):
            (bin_dir / tool).chmod(0o755)
        copied_scripts(tmp_path, "kube")
        (tmp_path / "runs" / "kube" / "r1" / "kube" / "inference").mkdir(parents=True)
        (tmp_path / "runs" / "kube" / "current").write_text("r1\n", encoding="utf-8")

        done = subprocess.run(
            ["sh", "deploy/kube/down.sh"],
            cwd=tmp_path,
            env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
            capture_output=True,
            text=True,
            check=False,
        )
        held = (bin_dir / "kubectl.args").read_text(encoding="utf-8").splitlines()
        order = [
            next(at for at, line in enumerate(held) if part in line)
            for part in (
                "delete job",
                "delete deploy control-plane",
                "delete -f runs/kube/r1/kube/inference",
                "delete deploy observation-service",
                "delete deploy data-hub",
                "delete deploy broker",
                "delete deploy metrics-database",
                "delete -k",
            )
        ]

        assert done.returncode == 0, done.stderr
        assert order == sorted(order)

    def test_load_after_run_refuses_in_one_line_and_starts_no_pod(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "poetry").write_text(NAMING, encoding="utf-8")
        (bin_dir / "kubectl").write_text(
            '#!/bin/sh\necho "$*" >> "$(dirname "$0")/kubectl.args"\n'
            'case "$*" in *"get deploy"*) exit 1 ;; esac\n',
            encoding="utf-8",
        )
        for tool in ("poetry", "kubectl"):
            (bin_dir / tool).chmod(0o755)
        copied_scripts(tmp_path, "kube")
        a_run_up(tmp_path, 8082)

        done = subprocess.run(
            ["sh", "deploy/kube/load_test.sh"],
            cwd=tmp_path,
            env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "READY_SECONDS": "1"},
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

        assert done.returncode != 0
        assert len(done.stdout.splitlines()) == 1
        assert "control plane" in done.stdout
        assert "apply" not in (bin_dir / "kubectl.args").read_text(encoding="utf-8")

    def test_down_fails_when_the_cluster_refuses_to_delete_the_run(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "poetry").write_text(NAMING, encoding="utf-8")
        (bin_dir / "kubectl").write_text(
            '#!/bin/sh\ncase "$*" in "delete -k"*) exit 1 ;; esac\n', encoding="utf-8"
        )
        for tool in ("poetry", "kubectl"):
            (bin_dir / tool).chmod(0o755)
        copied_scripts(tmp_path, "kube")
        (tmp_path / "runs" / "kube" / "r1").mkdir(parents=True)
        (tmp_path / "runs" / "kube" / "current").write_text("r1\n", encoding="utf-8")

        done = subprocess.run(
            ["sh", "deploy/kube/down.sh"],
            cwd=tmp_path,
            env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
            capture_output=True,
            text=True,
            check=False,
        )

        assert done.returncode != 0
        assert (tmp_path / "runs" / "kube" / "current").exists()


class TestARunIdOutsideTheRule:
    def test_up_refuses_it_before_anything_is_made(self, tmp_path: Path) -> None:
        bin_dir = forwarding(tmp_path, '#!/bin/sh\necho "$*" >> "$(dirname "$0")/poetry.args"\n')

        done = subprocess.run(
            ["sh", "deploy/kube/up.sh"],
            cwd=tmp_path,
            env={
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "EEMLS_DATABASE_PASSWORD": "test",
                "RUN_ID": "My_Run",
            },
            capture_output=True,
            text=True,
            check=False,
        )

        assert done.returncode == 1
        assert done.stdout.strip().splitlines() == [
            "run id My_Run must be lowercase letters, digits and hyphens, "
            "with no hyphen at either end"
        ]
        assert not (tmp_path / "runs").exists()
        assert not (bin_dir / "kubectl.args").exists()
        assert not (bin_dir / "poetry.args").exists()


class TestAnotherRunUp:
    def test_up_refuses_to_start_and_leaves_the_other_run_alone(self, tmp_path: Path) -> None:
        bin_dir = forwarding(tmp_path, '#!/bin/sh\necho "$*" >> "$(dirname "$0")/poetry.args"\n')
        other = tmp_path / "runs" / "kube" / "r0"
        other.mkdir(parents=True)
        (other / "seed").write_text("0\n", encoding="utf-8")
        (tmp_path / "runs" / "kube" / "current").write_text("r0\n", encoding="utf-8")

        done = subprocess.run(
            ["sh", "deploy/kube/up.sh"],
            cwd=tmp_path,
            env={
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "EEMLS_DATABASE_PASSWORD": "test",
                "RUN_ID": "r1",
            },
            capture_output=True,
            text=True,
            check=False,
        )

        assert done.returncode == 1
        assert done.stdout.strip().splitlines() == ["run r0 is up; run deploy/kube/down.sh first"]
        assert not (bin_dir / "kubectl.args").exists()
        assert not (bin_dir / "poetry.args").exists()
        assert (tmp_path / "runs" / "kube" / "current").read_text(encoding="utf-8") == "r0\n"
        assert sorted(p.name for p in (tmp_path / "runs" / "kube").iterdir()) == ["current", "r0"]
        assert [p.name for p in other.iterdir()] == ["seed"]


def free_id(root: Path, run_id: str) -> str:
    copied_scripts(root, "kube")
    done = subprocess.run(
        ["sh", "-c", f". deploy/kube/common.sh; free_run_id {run_id}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


KUBECTL = (
    '#!/bin/sh\ncase "$*" in\n'
    "  *port-forward*) exec sleep 30 ;;\n"
    '  *"get pod"*) echo Succeeded ;;\n'
    "esac\n"
)


class TestARunIdUsedBefore:
    def test_a_new_id_is_kept(self, tmp_path: Path) -> None:
        assert free_id(tmp_path, "profiles") == "profiles"

    def test_an_id_used_before_gets_the_first_free_number_in_front(self, tmp_path: Path) -> None:
        (tmp_path / "runs" / "kube" / "profiles").mkdir(parents=True)
        (tmp_path / "runs" / "kube" / "1-profiles").mkdir()

        assert free_id(tmp_path, "profiles") == "2-profiles"

    def test_up_starts_the_run_under_the_free_id(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "poetry").write_text("#!/bin/sh\n", encoding="utf-8")
        (bin_dir / "kubectl").write_text(KUBECTL, encoding="utf-8")
        for tool in ("poetry", "kubectl"):
            (bin_dir / tool).chmod(0o755)
        copied_scripts(tmp_path, "kube")
        (tmp_path / "runs" / "kube" / "profiles").mkdir(parents=True)

        subprocess.run(
            ["sh", "deploy/kube/up.sh"],
            cwd=tmp_path,
            env={
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "EEMLS_DATABASE_PASSWORD": "test",
                "RUN_ID": "profiles",
            },
            capture_output=True,
            text=True,
            check=False,
        )

        assert (tmp_path / "runs" / "kube" / "current").read_text(
            encoding="utf-8"
        ) == "1-profiles\n"

    def test_a_failed_apply_leaves_no_current_run(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "poetry").write_text("#!/bin/sh\n", encoding="utf-8")
        (bin_dir / "kubectl").write_text(
            '#!/bin/sh\ncase "$*" in apply*) exit 1 ;; esac\n', encoding="utf-8"
        )
        for tool in ("poetry", "kubectl"):
            (bin_dir / tool).chmod(0o755)
        copied_scripts(tmp_path, "kube")

        done = subprocess.run(
            ["sh", "deploy/kube/up.sh"],
            cwd=tmp_path,
            env={
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "EEMLS_DATABASE_PASSWORD": "test",
                "RUN_ID": "r1",
            },
            capture_output=True,
            text=True,
            check=False,
        )

        assert done.returncode != 0
        assert not (tmp_path / "runs" / "kube" / "current").exists()

    def test_a_failed_apply_removes_what_it_made(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "poetry").write_text("#!/bin/sh\n", encoding="utf-8")
        (bin_dir / "kubectl").write_text(
            '#!/bin/sh\necho "$*" >> "$(dirname "$0")/kubectl.args"\n'
            'case "$*" in "apply -k"*) exit 1 ;; esac\n',
            encoding="utf-8",
        )
        for tool in ("poetry", "kubectl"):
            (bin_dir / tool).chmod(0o755)
        copied_scripts(tmp_path, "kube")

        subprocess.run(
            ["sh", "deploy/kube/up.sh"],
            cwd=tmp_path,
            env={
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "EEMLS_DATABASE_PASSWORD": "test",
                "RUN_ID": "r1",
            },
            capture_output=True,
            check=False,
        )

        calls = (bin_dir / "kubectl.args").read_text(encoding="utf-8")
        assert "-n eemls delete pvc metrics-database-r1 --ignore-not-found" in calls
        assert "run.yaml" not in calls.split("apply -f runs/kube/r1/kube/run.yaml", 1)[1]
        assert not (tmp_path / "runs" / "kube" / "current").exists()


def recording(bin_dir: Path) -> Path:
    bin_dir.mkdir()
    (bin_dir / "poetry").write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" >> "$(dirname "$0")/poetry.args"\n', encoding="utf-8"
    )
    (bin_dir / "kubectl").write_text(KUBECTL, encoding="utf-8")
    for tool in ("poetry", "kubectl"):
        (bin_dir / tool).chmod(0o755)
    return bin_dir / "poetry.args"


class TestTheProfilesOfARun:
    def test_up_keeps_the_profiles_it_started_from(self, tmp_path: Path) -> None:
        recording(tmp_path / "bin")
        copied_scripts(tmp_path, "kube")

        subprocess.run(
            ["sh", "deploy/kube/up.sh"],
            cwd=tmp_path,
            env={
                **os.environ,
                "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
                "EEMLS_DATABASE_PASSWORD": "test",
                "RUN_ID": "r1",
                "SERVICE_PROFILES": "models/profiles_cluster.jsonl",
            },
            capture_output=True,
            text=True,
            check=False,
        )

        held = (tmp_path / "runs" / "kube" / "r1" / "profiles").read_text(encoding="utf-8")
        assert held == "models/profiles_cluster.jsonl\n"

    def test_up_keeps_the_image_store_it_started_from(self, tmp_path: Path) -> None:
        recording(tmp_path / "bin")
        copied_scripts(tmp_path, "kube")

        subprocess.run(
            ["sh", "deploy/kube/up.sh"],
            cwd=tmp_path,
            env={
                **os.environ,
                "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
                "EEMLS_DATABASE_PASSWORD": "test",
                "RUN_ID": "r1",
                "IMAGES": "/data/store",
            },
            capture_output=True,
            text=True,
            check=False,
        )

        held = (tmp_path / "runs" / "kube" / "r1" / "images").read_text(encoding="utf-8")
        assert held == "/data/store\n"

    def test_run_sizes_the_copies_from_the_profiles_the_run_started_from(
        self, tmp_path: Path
    ) -> None:
        args = recording(tmp_path / "bin")
        copied_scripts(tmp_path, "kube")
        run = tmp_path / "runs" / "kube" / "r1"
        run.mkdir(parents=True)
        (tmp_path / "runs" / "kube" / "current").write_text("r1\n", encoding="utf-8")
        (run / "profiles").write_text("models/profiles_cluster.jsonl\n", encoding="utf-8")
        (run / "images").write_text("/data/store\n", encoding="utf-8")
        (run / "seed").write_text("0\n", encoding="utf-8")

        subprocess.run(
            ["sh", "deploy/kube/run.sh"],
            cwd=tmp_path,
            env={
                **os.environ,
                "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
                "EEMLS_DATABASE_PASSWORD": "test",
                "DURATION_S": "1",
            },
            capture_output=True,
            text=True,
            check=False,
        )

        given = args.read_text(encoding="utf-8").splitlines()
        assert given[given.index("--profiles") + 1] == "models/profiles_cluster.jsonl"
        assert given[given.index("--images") + 1] == "/data/store"


def a_script(root: Path, script: str, **env: str) -> None:
    subprocess.run(
        ["sh", f"deploy/kube/{script}"],
        cwd=root,
        env={
            **os.environ,
            "PATH": f"{root / 'bin'}:{os.environ['PATH']}",
            "EEMLS_DATABASE_PASSWORD": "test",
            **env,
        },
        capture_output=True,
        text=True,
        check=False,
    )


class TestTheSeedOfARun:
    def test_up_deploys_the_control_plane_with_the_seed_it_is_given(self, tmp_path: Path) -> None:
        args = recording(tmp_path / "bin")
        copied_scripts(tmp_path, "kube")

        a_script(tmp_path, "up.sh", RUN_ID="r1", SEED="3")

        (up,) = deploy_options(args.read_text(encoding="utf-8").splitlines())
        assert up["--seed"] == "3"

    def test_up_and_run_deploy_the_control_plane_alike_for_one_seed(self, tmp_path: Path) -> None:
        args = recording(tmp_path / "bin")
        copied_scripts(tmp_path, "kube")

        a_script(tmp_path, "up.sh", RUN_ID="r1", SEED="3")
        a_script(tmp_path, "run.sh", DURATION_S="1")

        up, run = deploy_options(args.read_text(encoding="utf-8").splitlines())
        consumers = {"--scenario", "--scenarios", "--duration-s"}
        assert {k: v for k, v in run.items() if k not in consumers} == up

    def test_run_keeps_the_seed_the_run_was_started_with(self, tmp_path: Path) -> None:
        args = recording(tmp_path / "bin")
        copied_scripts(tmp_path, "kube")

        a_script(tmp_path, "up.sh", RUN_ID="r1", SEED="3")
        a_script(tmp_path, "run.sh", DURATION_S="1")

        up, run = deploy_options(args.read_text(encoding="utf-8").splitlines())
        assert (up["--seed"], run["--seed"]) == ("3", "3")

    def test_run_ignores_a_seed_other_than_the_one_the_run_was_started_with(
        self, tmp_path: Path
    ) -> None:
        args = recording(tmp_path / "bin")
        copied_scripts(tmp_path, "kube")

        a_script(tmp_path, "up.sh", RUN_ID="r1", SEED="3")
        a_script(tmp_path, "run.sh", SEED="5", DURATION_S="1")

        _, run = deploy_options(args.read_text(encoding="utf-8").splitlines())
        assert run["--seed"] == "3"


def a_cluster(root: Path, phase: str) -> dict[str, str]:
    """Fake poetry and kubectl; every pod stays in `phase`."""
    bin_dir = root / "bin"
    bin_dir.mkdir()
    (bin_dir / "poetry").write_text("#!/bin/sh\necho '{}'\n", encoding="utf-8")
    (bin_dir / "kubectl").write_text(
        f'#!/bin/sh\ncase "$*" in *"get pod"*) echo {phase} ;; esac\n', encoding="utf-8"
    )
    for tool in ("poetry", "kubectl"):
        (bin_dir / tool).chmod(0o755)
    copied_scripts(root, "kube")
    a_run_up(root, 8082)
    return {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "READY_SECONDS": "1"}


class TestAPodInTheCluster:
    def test_a_pod_that_never_finishes_stops_the_script(self, tmp_path: Path) -> None:
        env = a_cluster(tmp_path, "Pending")

        done = subprocess.run(
            ["sh", "-c", ". deploy/kube/common.sh; current_run; in_cluster p consumer true"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

        assert done.returncode != 0
        assert "p never finished" in done.stdout

    def test_a_failed_load_test_fails_the_script(self, tmp_path: Path) -> None:
        env = a_cluster(tmp_path, "Failed")

        done = subprocess.run(
            ["sh", "deploy/kube/load_test.sh"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

        assert done.returncode != 0

    def test_a_one_off_pod_mounts_every_file_of_the_run_s_config_map(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "poetry").write_text(DEPLOY, encoding="utf-8")
        (bin_dir / "kubectl").write_text(
            '#!/bin/sh\nprintf \'%s\\n\' "$@" >> "$(dirname "$0")/kubectl.args"\n'
            'case "$*" in *"get pod"*) echo Succeeded ;; esac\n'
            'case "$3 $4" in "apply -f") cp "$5" "$(dirname "$0")/pod.yaml" ;; esac\n',
            encoding="utf-8",
        )
        for tool in ("poetry", "kubectl"):
            (bin_dir / tool).chmod(0o755)
        copied_scripts(tmp_path, "kube")
        run = tmp_path / "runs" / "kube" / "r1"
        tree = committed_tree(run / "config")
        (tree / "local").mkdir()
        (tree / "local" / "images.yaml").write_text(
            "registry: registry.example\nversion: v1\n", encoding="utf-8"
        )
        (tmp_path / "runs" / "kube" / "current").write_text("r1\n", encoding="utf-8")

        done = subprocess.run(
            ["sh", "-c", ". deploy/kube/common.sh; current_run; in_cluster p consumer true"],
            cwd=tmp_path,
            env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert done.returncode == 0, done.stderr
        args = (bin_dir / "kubectl.args").read_text(encoding="utf-8").splitlines()
        pod = yaml.safe_load((bin_dir / "pod.yaml").read_text(encoding="utf-8"))
        files = sorted(
            str(path.relative_to(tree))
            for path in tree.rglob("*.yaml")
            if path.relative_to(tree).parts[0] != "local"
        )

        assert str(run / "p.yaml").removeprefix(f"{tmp_path}/") in args
        assert pod["spec"]["volumes"][0]["configMap"]["items"] == [
            {"key": path.replace("/", "__"), "path": path} for path in files
        ]
        assert pod["spec"]["containers"][0]["image"] == "registry.example/eemls-consumer:v1"
        assert pod["spec"]["containers"][0]["command"] == ["true"]
        assert not (run / "p.yaml").exists()


def test_the_cluster_plans_an_archived_profile_against_the_archived_scenarios(
    tmp_path: Path,
) -> None:
    args = recording(tmp_path / "bin")
    copied_scripts(tmp_path, "kube")

    subprocess.run(
        ["sh", "deploy/kube/experiment.sh"],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
            "EEMLS_DATABASE_PASSWORD": "test",
            "PLAN": "experiments/archived/runs/shuffled_days_seed_3.yaml",
            "OUT": "out",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    given = args.read_text(encoding="utf-8").splitlines()
    assert given[given.index("--scenarios") + 1] == "experiments/archived/scenarios"


class TestTheScenariosOfARun:
    def test_run_reads_the_scenario_from_the_folder_it_is_given(self, tmp_path: Path) -> None:
        args = recording(tmp_path / "bin")
        copied_scripts(tmp_path, "kube")

        a_script(tmp_path, "up.sh", RUN_ID="r1")
        a_script(tmp_path, "run.sh", DURATION_S="1", SCENARIOS="experiments/archived/scenarios")

        _, run = deploy_options(args.read_text(encoding="utf-8").splitlines())
        assert run["--scenarios"] == "experiments/archived/scenarios"

    def test_run_reads_the_scenario_from_the_committed_folder_by_default(
        self, tmp_path: Path
    ) -> None:
        args = recording(tmp_path / "bin")
        copied_scripts(tmp_path, "kube")

        a_script(tmp_path, "up.sh", RUN_ID="r1")
        a_script(tmp_path, "run.sh", DURATION_S="1")

        _, run = deploy_options(args.read_text(encoding="utf-8").splitlines())
        assert run["--scenarios"] == "experiments/scenarios"

    def test_an_experiment_runs_its_consumers_on_the_scenarios_it_planned_against(
        self, tmp_path: Path
    ) -> None:
        args = recording(tmp_path / "bin")
        (tmp_path / "bin" / "poetry").write_text(
            '#!/bin/sh\nprintf "%s\\n" "$@" >> "$(dirname "$0")/poetry.args"\n'
            'case "$2 $3" in "eemls-exp plan") echo "cap 0 r1 cap_0 60.0 smoke" ;; esac\n',
            encoding="utf-8",
        )
        copied_scripts(tmp_path, "kube")
        (tmp_path / "out" / "cap_0" / "config").mkdir(parents=True)

        a_script(
            tmp_path,
            "experiment.sh",
            PLAN="experiments/archived/runs/shuffled_days_seed_3.yaml",
            OUT="out",
        )

        options = deploy_options(args.read_text(encoding="utf-8").splitlines())
        [run] = [one for one in options if "--scenario" in one]
        assert run["--scenarios"] == "experiments/archived/scenarios"


ORDERED = (
    '#!/bin/sh\necho "$(basename "$0") $*" >> "$(dirname "$0")/calls.log"\n'
    'case "$*" in\n'
    "  *port-forward*) exec sleep 30 ;;\n"
    '  *"get pod"*) echo Succeeded ;;\n'
    '  *"eemls-deploy name"*) eval last=\\${$#}; echo "$last" | tr _ - ;;\n'
    "esac\n"
)


def ordered(root: Path) -> Path:
    bin_dir = root / "bin"
    bin_dir.mkdir()
    for tool in ("poetry", "kubectl"):
        (bin_dir / tool).write_text(ORDERED, encoding="utf-8")
        (bin_dir / tool).chmod(0o755)
    copied_scripts(root, "kube")
    return bin_dir / "calls.log"


def a_run_with_its_config(root: Path) -> None:
    run = root / "runs" / "kube" / "r1"
    (run / "config").mkdir(parents=True)
    for name in ("profiles", "images", "seed"):
        (run / name).write_text("0\n", encoding="utf-8")
    (root / "runs" / "kube" / "current").write_text("r1\n", encoding="utf-8")


def first_at(log: list[str], part: str) -> int:
    return next(at for at, line in enumerate(log) if part in line)


class TestTheObservationServiceOfARun:
    def test_up_waits_until_the_observation_service_is_ready(self, tmp_path: Path) -> None:
        log = ordered(tmp_path)

        a_script(tmp_path, "up.sh", RUN_ID="r1")

        held = log.read_text(encoding="utf-8").splitlines()
        assert first_at(held, "rollout status deploy/metrics-database") < first_at(
            held, "rollout status deploy/observation-service"
        )

    def test_run_stops_the_control_plane_before_it_collects_the_rows(self, tmp_path: Path) -> None:
        log = ordered(tmp_path)
        a_run_with_its_config(tmp_path)

        a_script(tmp_path, "run.sh", DURATION_S="1")

        held = log.read_text(encoding="utf-8").splitlines()
        assert first_at(held, "wait --for=condition=complete job") < first_at(
            held, "delete deploy control-plane"
        )
        assert first_at(held, "delete deploy control-plane") < first_at(held, "eemls-collect")

    def test_down_stops_the_consumers_the_control_plane_the_observation_service_and_the_rest(
        self, tmp_path: Path
    ) -> None:
        log = ordered(tmp_path)
        a_run_with_its_config(tmp_path)

        a_script(tmp_path, "down.sh")

        held = log.read_text(encoding="utf-8").splitlines()
        order = [
            first_at(held, part)
            for part in (
                "delete job --all",
                "delete deploy control-plane",
                "delete deploy observation-service",
                "delete -k",
            )
        ]
        assert order == sorted(order)
        assert "--cascade=foreground" in held[order[1]]
