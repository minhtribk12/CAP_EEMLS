from __future__ import annotations

import ast
import hashlib
import importlib.util
import re
import subprocess
import tomllib
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from eemls_tools.deployment_plan import CONFIG_PATH, SCENARIO_PATH
from tests.support.imports import imported_names

ROOT = Path(__file__).resolve().parents[2]
APPLICATION = ROOT / "application"
BAKE = ROOT / "docker-bake.hcl"

FOLDERS = ("control-plane", "observation-service", "data-hub", "consumer", "inference-service")
SETTINGS = {
    "control-plane": "control_plane",
    "observation-service": "observation_service",
    "data-hub": "data_hub",
    "inference-service": "inference_service",
}
PACKAGE_OF = {
    "inference-service": "eemls_inference_service",
    "data-hub": "eemls_data_hub",
    "consumer": "eemls_consumer",
    "control-plane": "eemls_control_plane",
    "observation-service": "eemls_observation_service",
}
ENTRYPOINT_OF = {
    "inference-service": "eemls-inference-service",
    "data-hub": "eemls-data-hub",
    "consumer": "eemls-consumer",
    "control-plane": "eemls-control-plane",
    "observation-service": "eemls-observation-service",
}
SCRIPTS_OF = {
    "inference-service": ("eemls-inference-service",),
    "data-hub": ("eemls-data-hub",),
    "consumer": ("eemls-consumer", "eemls-load"),
    "control-plane": ("eemls-control-plane", "eemls-warm", "eemls-collect"),
    "observation-service": ("eemls-observation-service",),
}
CPU_TAGS = ("cpu-amd64", "cpu-arm64")


def dockerfile(folder: str, name: str = "Dockerfile") -> str:
    return (APPLICATION / folder / name).read_text(encoding="utf-8")


def bake() -> str:
    return BAKE.read_text(encoding="utf-8")


def target_of(name: str) -> str:
    target = re.search(rf'target "{name}" {{(.*?)\n}}', bake(), flags=re.DOTALL)
    assert target is not None, name
    return target.group(1)


def project_of(folder: str) -> dict[str, object]:
    with (APPLICATION / folder / "pyproject.toml").open("rb") as file:
        project: dict[str, object] = tomllib.load(file)["project"]
    return project


def pinned_requirement(folder: str) -> str:
    dependencies = project_of(folder)["dependencies"]
    assert isinstance(dependencies, list)
    [requirement] = [
        str(one) for one in dependencies if re.fullmatch(r"eemls(?:\[[^\]]*\])?==\S+", one)
    ]
    return requirement


def framework_file(folder: str) -> re.Match[str]:
    line = (APPLICATION / folder / "framework.txt").read_text(encoding="utf-8")
    found = re.fullmatch(r"(eemls(?:\[[^\]]*\])?==(\S+)) --hash=sha256:([0-9a-f]{64})\n", line)
    assert found is not None, line
    return found


def load_pin() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pin", ROOT / "tools" / "pin.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fresh_digests(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    pin = load_pin()
    digests: dict[str, str] = {}
    for version in sorted({framework_file(folder).group(2) for folder in FOLDERS}):
        wheel = pin.built_wheel(ROOT, version, tmp_path_factory.mktemp(version))
        digests[version] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    return digests


def ports_of(folder: str) -> list[int]:
    if folder == "consumer":
        settings = yaml.safe_load((ROOT / "config/services/consumer.yaml").read_text())
        return [settings["metrics_port"]]
    settings = yaml.safe_load((ROOT / f"config/services/{SETTINGS[folder]}.yaml").read_text())
    return [settings["port"]]


def copies(folder: str, name: str = "Dockerfile") -> list[str]:
    return [
        line
        for line in dockerfile(folder, name).splitlines()
        if line.split(" ", 1)[0] in {"COPY", "ADD"}
    ]


def copied_sources(folder: str, name: str = "Dockerfile") -> list[str]:
    return [
        part
        for line in copies(folder, name)
        for part in line.split()[1:-1]
        if not part.startswith("--")
    ]


def test_every_application_folder_is_named_in_every_list() -> None:
    folders = sorted(path.name for path in APPLICATION.glob("*") if path.is_dir())

    assert sorted(FOLDERS) == folders
    for named in (PACKAGE_OF, ENTRYPOINT_OF, SCRIPTS_OF):
        assert sorted(named) == folders
    assert sorted(SETTINGS) == sorted(set(folders) - {"consumer"})


@pytest.mark.parametrize("folder", FOLDERS)
def test_every_application_owns_a_dockerfile_and_a_readme(folder: str) -> None:
    assert (APPLICATION / folder / "Dockerfile").is_file()
    assert (APPLICATION / folder / "README.md").is_file()


@pytest.mark.parametrize("folder", FOLDERS)
def test_the_readme_names_every_port_the_settings_give(folder: str) -> None:
    readme = (APPLICATION / folder / "README.md").read_text(encoding="utf-8")

    for port in ports_of(folder):
        assert re.search(rf"^\| {port} \|", readme, flags=re.MULTILINE), port


@pytest.mark.parametrize("folder", FOLDERS)
def test_the_image_exposes_the_ports_the_settings_give(folder: str) -> None:
    exposed = re.search(r"^EXPOSE (.+)$", dockerfile(folder), flags=re.MULTILINE)

    assert exposed is not None
    assert sorted(int(port) for port in exposed.group(1).split()) == sorted(ports_of(folder))


@pytest.mark.parametrize("folder", FOLDERS)
class TestEachImage:
    def test_it_builds_the_framework_wheel_from_the_framework_context(self, folder: str) -> None:
        body = dockerfile(folder)

        assert re.search(
            r"^FROM --platform=\$BUILDPLATFORM python:3\.12-slim AS framework-wheel$",
            body,
            flags=re.MULTILINE,
        )
        assert "COPY --from=framework pyproject.toml LICENSE ./" in body
        assert "COPY --from=framework src/eemls ./src/eemls" in body
        assert "poetry==2.5.1" in body.split("AS framework-wheel")[1].split("\nFROM ")[0]
        assert "poetry build --format wheel --output /wheels" in body
        assert "pinned" not in body

    def test_it_installs_the_framework_wheel_only_by_the_hash_it_pins(self, folder: str) -> None:
        body = dockerfile(folder)

        assert "COPY framework.txt ./" in body
        assert "--mount=type=bind,from=framework-wheel,source=/wheels,target=/opt/framework" in body
        assert (
            "pip install --no-deps --no-index --require-hashes --requirement framework.txt"
            " --find-links /opt/framework"
        ) in body

    def test_it_installs_the_locked_packages_then_the_framework_then_its_package(
        self, folder: str
    ) -> None:
        body = dockerfile(folder)

        assert body.index("COPY pyproject.toml poetry.lock README.md ./") < body.index(
            "poetry export --only main"
        )
        assert body.index("--require-hashes --requirement requirements.txt") < body.index(
            "--requirement framework.txt"
        )
        assert body.index("--requirement framework.txt") < body.index(
            "RUN pip install --no-deps .\n"
        )

    def test_it_copies_only_its_source_and_project_files(self, folder: str) -> None:
        extra = ["cuda-requirements.txt"] if folder == "inference-service" else []
        own = [line for line in copies(folder) if "--from=framework " not in line]
        sources = [part for line in own for part in line.split()[1:-1] if not part.startswith("--")]

        assert sorted(sources) == sorted(
            [
                "framework.txt",
                "pyproject.toml",
                "poetry.lock",
                "README.md",
                "src",
                "/opt/venv",
                *extra,
            ]
        )

    def test_it_installs_its_package_without_its_dependencies_after_the_code(
        self, folder: str
    ) -> None:
        body = dockerfile(folder)

        assert body.index("COPY src ./src") < body.index("RUN pip install --no-deps .\n")

    def test_it_starts_its_console_script(self, folder: str) -> None:
        assert f'ENTRYPOINT ["{ENTRYPOINT_OF[folder]}"]' in dockerfile(folder)

    def test_every_console_script_points_at_a_main_of_its_own_package(self, folder: str) -> None:
        scripts = project_of(folder)["scripts"]
        assert isinstance(scripts, dict)

        assert tuple(scripts) == SCRIPTS_OF[folder]
        for target in scripts.values():
            module, function = target.split(":")
            assert module.split(".")[0] == PACKAGE_OF[folder]
            path = APPLICATION / folder / "src" / Path(*module.split("."))
            tree = ast.parse(path.with_suffix(".py").read_text(encoding="utf-8"))
            assert function in {
                node.name for node in tree.body if isinstance(node, ast.FunctionDef)
            }

    def test_it_points_the_service_at_the_mounted_config(self, folder: str) -> None:
        assert f"EEMLS_CONFIG_DIR={CONFIG_PATH}" in dockerfile(folder)

    def test_it_holds_no_config_or_scenarios(self, folder: str) -> None:
        held = copied_sources(folder)

        assert not [one for one in held if one.split("/")[0] in {"config", "experiments"}]

    @pytest.mark.parametrize("left_out", ["tests", ".venv", "**/__pycache__"])
    def test_its_build_context_leaves_out_what_no_image_needs(
        self, folder: str, left_out: str
    ) -> None:
        ignored = (APPLICATION / folder / ".dockerignore").read_text(encoding="utf-8").splitlines()

        assert left_out in ignored

    def test_the_bake_file_builds_it_from_its_own_folder(self, folder: str) -> None:
        name = "_inference-service" if folder == "inference-service" else folder
        target = target_of(name)

        assert f'context    = "application/{folder}"' in target
        assert 'dockerfile = "Dockerfile"' in target

    def test_the_bake_file_gives_it_the_framework_context_of_its_pin(self, folder: str) -> None:
        name = "_inference-service" if folder == "inference-service" else folder
        variable = "FRAMEWORK_" + folder.replace("-", "_").upper()

        assert re.search(rf'variable "{variable}" {{\s*default = "\."\s*}}', bake())
        assert f"framework = {variable}" in target_of(name)

    def test_the_readme_says_the_config_is_mounted(self, folder: str) -> None:
        readme = (APPLICATION / folder / "README.md").read_text(encoding="utf-8")
        mounted = rf"^- Mount the config folder at `{CONFIG_PATH}`"

        assert re.search(mounted, readme, flags=re.MULTILINE)

    def test_its_framework_file_names_the_pin_of_its_project(self, folder: str) -> None:
        assert framework_file(folder).group(1) == pinned_requirement(folder)

    def test_its_framework_file_names_the_hash_of_a_fresh_build_of_that_version(
        self, folder: str, fresh_digests: dict[str, str]
    ) -> None:
        found = framework_file(folder)

        assert found.group(3) == fresh_digests[found.group(2)]

    def test_it_has_no_pinned_folder(self, folder: str) -> None:
        assert not (APPLICATION / folder / "pinned").exists()

    def test_poetry_takes_the_framework_from_the_repository_root(self, folder: str) -> None:
        with (APPLICATION / folder / "pyproject.toml").open("rb") as file:
            poetry = tomllib.load(file)["tool"]["poetry"]["dependencies"]["eemls"]
        lock = (APPLICATION / folder / "poetry.lock").read_text(encoding="utf-8")

        assert poetry["path"] == "../.."
        assert poetry["develop"] is False
        assert '[package.source]\ntype = "directory"\nurl = "../.."' in lock


def test_the_control_plane_takes_extra_approach_packages_as_a_build_argument() -> None:
    body = dockerfile("control-plane")

    assert re.search(r'^ARG APPROACH_PACKAGES=""$', body, flags=re.MULTILINE)
    assert 'if [ -n "$APPROACH_PACKAGES" ]; then pip install $APPROACH_PACKAGES; fi' in body
    assert body.index("$APPROACH_PACKAGES; fi") < body.index("RUN pip install --no-deps .\n")


def test_the_inference_service_takes_its_runtime_as_a_build_argument() -> None:
    body = dockerfile("inference-service")

    assert re.search(r"^ARG RUNTIME=python:3\.12-slim$", body, flags=re.MULTILINE)
    assert re.search(r"^FROM \$\{RUNTIME\} AS builder$", body, flags=re.MULTILINE)
    assert re.search(r"^ARG ONNX_RUNTIME=\"\"$", body, flags=re.MULTILINE)


def test_no_target_builds_from_a_shared_base() -> None:
    assert "base" not in re.findall(r'target "([^"]+)"', bake())
    assert "target:base" not in bake()
    assert not (ROOT / "docker").exists()


@pytest.mark.parametrize("group", ["default", "local"])
def test_every_bake_group_builds_the_observation_service(group: str) -> None:
    found = re.search(rf'group "{group}" {{\s*targets = \[([^\]]*)\]', bake())
    assert found is not None
    target = "observation-service" if group == "default" else "observation-service-local"

    assert f'"{target}"' in found.group(1)


@pytest.mark.parametrize("tag", CPU_TAGS)
def test_the_bake_file_builds_each_cpu_tag_of_the_inference_service(tag: str) -> None:
    assert f"inference-service:{tag}" in bake()


def test_make_images_pushes_to_the_registry_the_local_file_names() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    target = re.search(r"^images:\n((?:\t.*\n)+)", makefile, flags=re.MULTILINE)

    assert target is not None
    assert "deploy/images.sh" in target.group(1)
    script = (ROOT / "deploy" / "images.sh").read_text(encoding="utf-8")
    assert "local/images.yaml" in script
    assert "tools/pin.py --bake --push" in script


def test_no_wheel_is_tracked_by_git() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "*.whl"], cwd=ROOT, check=True, capture_output=True, text=True
    )

    assert tracked.stdout == ""


def test_no_pinned_folder_is_named_anywhere() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "pinned" not in (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "wheels:" not in makefile
    assert "--wheels" not in makefile + (ROOT / "deploy" / "images.sh").read_text(encoding="utf-8")


def test_the_local_images_are_baked_with_the_framework_contexts() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    target = re.search(r"^images-local:\n((?:\t.*\n)+)", makefile, flags=re.MULTILINE)

    assert target is not None
    assert "tools/pin.py --bake --load local" in target.group(1)


def locked_version(package: str) -> str:
    lock = (APPLICATION / "inference-service" / "poetry.lock").read_text(encoding="utf-8")
    found = re.search(rf'^name = "{package}"\nversion = "([^"]+)"', lock, flags=re.MULTILINE)
    assert found is not None, package
    return found.group(1)


def test_the_bake_file_builds_the_cuda_tag_of_the_inference_service() -> None:
    assert "inference-service:cuda-amd64" in bake()


CUDA_REQUIREMENTS = APPLICATION / "inference-service" / "cuda-requirements.txt"


def cuda_requirements() -> list[list[str]]:
    text = CUDA_REQUIREMENTS.read_text(encoding="utf-8").replace("\\\n", " ")
    return [line.split() for line in text.splitlines() if line.strip()]


def test_the_cuda_tag_installs_the_hashed_requirements_without_their_dependencies() -> None:
    swapped = "--require-hashes --no-deps --requirement cuda-requirements.txt"

    assert f'ONNX_RUNTIME = "{swapped}"' in target_of("inference-service-cuda-amd64")
    assert "COPY cuda-requirements.txt ./" in dockerfile("inference-service")


def test_the_cuda_requirements_swap_in_the_gpu_runtime_of_the_locked_version() -> None:
    names = [line[0] for line in cuda_requirements()]

    assert f"onnxruntime-gpu=={locked_version('onnxruntime')}" in names


def test_every_cuda_requirement_is_pinned_and_hashed() -> None:
    lines = cuda_requirements()

    assert len(lines) > 1
    for name, *hashes in lines:
        assert re.fullmatch(r"[A-Za-z0-9_.\-]+==[0-9][0-9A-Za-z.]*", name)
        assert hashes and all(re.fullmatch(r"--hash=sha256:[0-9a-f]{64}", one) for one in hashes)


def jetson(name: str = "jetson.Dockerfile") -> str:
    return dockerfile("inference-service", name)


class TestTheJetsonImage:
    def test_the_bake_file_builds_the_jetson_tag_of_the_inference_service(self) -> None:
        assert "inference-service:jetson" in bake()

    def test_the_jetson_images_are_built_inside_the_inference_service_folder(self) -> None:
        for name, file in (
            ("jetpack", "jetson.Dockerfile"),
            ("jetson-runtime", "jetson.Dockerfile"),
            ("onnxruntime-jetson", "onnxruntime.Dockerfile"),
        ):
            target = target_of(name)
            assert 'context    = "application/inference-service"' in target
            assert f'dockerfile = "{file}"' in target

    def test_the_runtime_is_the_jetpack_release_the_configuration_records(self) -> None:
        release = yaml.safe_load((ROOT / "config/services/deployment.yaml").read_text())[
            "jetpack_release"
        ]

        assert f"nvcr.io/nvidia/l4t-jetpack:r{release}" in jetson()

    def test_the_runtime_carries_the_python_the_package_needs(self) -> None:
        assert re.search(r"python install 3\.12\.\d+", jetson())

    def test_one_version_names_the_onnx_runtime_built_and_the_one_swapped_in(self) -> None:
        version = re.search(r'variable "ORT_JETSON" {\s*default = "([\d.]+)"', bake())

        assert version is not None
        assert "args       = { ORT_VERSION = ORT_JETSON }" in bake()
        assert "onnxruntime_gpu-${ORT_JETSON}-cp312" in bake()
        assert '--branch "v${ORT_VERSION}"' in jetson("onnxruntime.Dockerfile")

    def test_the_jetson_onnx_runtime_is_no_newer_than_the_locked_one(self) -> None:
        version = re.search(r'variable "ORT_JETSON" {\s*default = "([\d.]+)"', bake())
        assert version is not None

        def parts(text: str) -> tuple[int, ...]:
            return tuple(int(part) for part in text.split("."))

        assert parts(version.group(1)) <= parts(locked_version("onnxruntime"))

    def test_onnx_runtime_is_built_for_the_orin_gpu_with_tensorrt(self) -> None:
        build = jetson("onnxruntime.Dockerfile")

        assert "CMAKE_CUDA_ARCHITECTURES=87" in build
        assert "--use_tensorrt" in build

    def test_the_built_wheel_lands_in_the_inference_service_folder(self) -> None:
        output = '["type=local,dest=application/inference-service/onnxruntime-wheels"]'

        assert f"output     = {output}" in target_of("onnxruntime-jetson")
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert "/application/inference-service/onnxruntime-wheels/" in ignored

    def test_the_jetson_tag_swaps_in_the_wheel_built_for_the_boards(self) -> None:
        target = target_of("inference-service-jetson")

        assert 'runtime = "target:jetson-runtime"' in target
        assert 'wheels = "application/inference-service/onnxruntime-wheels"' in target
        assert 'RUNTIME      = "runtime"' in target
        assert "/opt/wheels/onnxruntime_gpu-" in target

    def test_the_jetson_device_class_runs_the_jetson_tag(self) -> None:
        tags = yaml.safe_load((ROOT / "config/services/deployment.yaml").read_text())["image_tags"]

        assert tags["jetson"] == "jetson"

    def test_the_runtime_is_plain_ubuntu_with_only_the_libraries_onnx_runtime_loads(self) -> None:
        final = jetson().split("\nFROM ")[-1]

        assert final.startswith("ubuntu:22.04")
        assert "COPY --from=jetpack /opt/python /opt/python" in final
        assert "libcudnn.so.9" in jetson() and "libnvinfer.so.10" in jetson()
        assert "jetson-slim" not in bake()

    def test_onnx_runtime_is_built_on_the_full_jetpack_stage(self) -> None:
        assert 'jetson = "target:jetpack"' in target_of("onnxruntime-jetson")
        assert 'target     = "jetpack"' in target_of("jetpack")


def test_every_image_is_named_eemls_and_its_service_under_the_registry() -> None:
    tags = re.findall(r'"\$\{REGISTRY\}/([^:"]+):', bake())

    assert tags
    assert set(tags) == {f"eemls-{folder}" for folder in FOLDERS}


def test_the_inference_image_carries_no_model() -> None:
    assert not [one for one in copied_sources("inference-service") if one.split("/")[0] == "models"]


@pytest.mark.parametrize(
    "name", ["jetson.Dockerfile", "onnxruntime.Dockerfile"], ids=lambda name: name
)
def test_no_jetson_image_holds_config_or_scenarios(name: str) -> None:
    held = copied_sources("inference-service", name)

    assert not [one for one in held if one.split("/")[0] in {"config", "experiments"}]


def test_the_consumer_readme_says_the_scenarios_are_mounted() -> None:
    readme = (APPLICATION / "consumer" / "README.md").read_text(encoding="utf-8")

    assert f"Mount the scenario files at `{SCENARIO_PATH}`" in readme


def test_the_profile_loader_runs_in_the_control_plane_image() -> None:
    imported = {
        name.split(".")[0]
        for name in imported_names(ROOT / "tools" / "load_profiles.py")
        if name.startswith("eemls")
    }

    assert imported <= {"eemls", "eemls_control_plane"}


def test_the_root_environment_runs_the_framework_from_its_own_source() -> None:
    lock = (ROOT / "poetry.lock").read_text(encoding="utf-8")

    assert not re.search(r'^name = "eemls"$', lock, flags=re.MULTILINE)


def test_the_root_lock_resolves_no_application_and_so_no_framework_pin() -> None:
    lock = (ROOT / "poetry.lock").read_text(encoding="utf-8")
    with (ROOT / "pyproject.toml").open("rb") as file:
        development = tomllib.load(file)["tool"]["poetry"]["group"]["dev"]["dependencies"]

    assert not [one for one in development if one.startswith("eemls")]
    assert not re.search(r'^name = "eemls-', lock, flags=re.MULTILINE)


def test_make_install_adds_every_application_and_the_tools_without_their_dependencies() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    target = re.search(r"^install:\n((?:\t.*\n)+)", makefile, flags=re.MULTILINE)
    assert target is not None

    assert "PROJECTS := $(wildcard application/*) tools" in makefile
    assert "pip install --no-deps $(addprefix --editable ,$(PROJECTS))" in target.group(1)


def test_the_image_script_reads_only_the_local_image_names() -> None:
    script = (ROOT / "deploy" / "images.sh").read_text(encoding="utf-8")

    assert "load_images_config('config')" in script
    assert "load_config" not in script
