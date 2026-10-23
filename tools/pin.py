from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = re.compile(r"eemls(?:\[([^\]]*)\])?(?![\w-])")
PINNED = re.compile(r"eemls(?:\[[^\]]*\])?==(\S+) --hash=sha256:([0-9a-f]{64})")
SECTION = "[tool.poetry.dependencies]"


class RefusalError(Exception):
    pass


def wheel_name(version: str) -> str:
    return f"eemls-{version}-py3-none-any.whl"


def tag_name(version: str) -> str:
    return f"eemls-v{version}"


def framework_version(root: Path) -> str:
    with (root / "pyproject.toml").open("rb") as file:
        version: str = tomllib.load(file)["project"]["version"]
    return version


def framework_line(version: str, extras: list[str], digest: str) -> str:
    marked = f"[{','.join(extras)}]" if extras else ""
    return f"eemls{marked}=={version} --hash=sha256:{digest}\n"


def poetry_line(extras: list[str]) -> str:
    listed = ", ".join(f'"{extra}"' for extra in extras)
    extra = f", extras = [{listed}]" if extras else ""
    return f'eemls = {{ path = "../..", develop = false{extra} }}'


def framework_extras(text: str) -> tuple[str, list[str]]:
    dependencies = tomllib.loads(text)["project"]["dependencies"]
    [found] = [match for one in dependencies if (match := FRAMEWORK.match(one))]
    extras = [one.strip() for one in (found.group(1) or "").split(",") if one.strip()]
    return found.string, extras


def pinned_project(text: str, version: str) -> str:
    requirement, extras = framework_extras(text)
    marked = f"[{','.join(extras)}]" if extras else ""
    text = text.replace(f'"{requirement}"', f'"eemls{marked}=={version}"', 1)
    line = poetry_line(extras)
    if SECTION in text:
        return re.sub(r"(?m)^eemls = \{.*\}$", line, text, count=1)
    start = text.index("[tool.poetry]\n")
    end = text.index("\n[", start + 1) + 1
    return f"{text[:end]}{SECTION}\n{line}\n\n{text[end:]}"


def pinned_framework(folder: Path) -> tuple[str, str]:
    file = folder / "framework.txt"
    found = PINNED.fullmatch(file.read_text(encoding="utf-8").strip()) if file.is_file() else None
    if found is None:
        raise RefusalError(f"{folder.name} pins no eemls version in framework.txt")
    return found.group(1), found.group(2)


def poetry_build(source: Path, output: Path) -> None:
    subprocess.run(
        ["poetry", "build", "--format", "wheel", "--output", str(output)],
        cwd=source,
        check=True,
        capture_output=True,
    )


def tagged_source(root: Path, version: str, folder: Path) -> Path:
    tag = tag_name(version)
    known = subprocess.run(
        ["git", "rev-parse", "--quiet", "--verify", f"refs/tags/{tag}"],
        cwd=root,
        capture_output=True,
    )
    if known.returncode != 0:
        raise RefusalError(f"no git tag {tag}: release {version} first")
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "source.tar"
    subprocess.run(
        ["git", "archive", "--output", str(archive), tag], cwd=root, check=True, capture_output=True
    )
    with tarfile.open(archive) as files:
        files.extractall(folder / "source", filter="data")
    archive.unlink()
    return folder / "source"


def framework_source(root: Path, version: str, folder: Path) -> Path:
    if version == framework_version(root):
        return root
    return tagged_source(root, version, folder)


def wheel_from(source: Path, version: str, output: Path) -> Path:
    poetry_build(source, output)
    wheel = output / wheel_name(version)
    if not wheel.is_file():
        raise RefusalError(f"the tag {tag_name(version)} builds no eemls {version} wheel")
    return wheel


def built_wheel(root: Path, version: str, output: Path) -> Path:
    with tempfile.TemporaryDirectory() as folder:
        return wheel_from(framework_source(root, version, Path(folder)), version, output)


def relock(root: Path, source: Path, folder: Path) -> None:
    if source == root:
        subprocess.run(["poetry", "lock"], cwd=folder, check=True)
        return
    copy = source / "application" / folder.name
    copy.mkdir(parents=True, exist_ok=True)
    for name in ("pyproject.toml", "poetry.lock", "README.md"):
        if (folder / name).is_file():
            shutil.copyfile(folder / name, copy / name)
    subprocess.run(["poetry", "lock"], cwd=copy, check=True)
    shutil.copyfile(copy / "poetry.lock", folder / "poetry.lock")


def pin(root: Path, application: str, version: str) -> None:
    folder = root / "application" / application
    if not folder.is_dir():
        raise RefusalError(f"{folder} is no application")
    project = folder / "pyproject.toml"
    text = pinned_project(project.read_text(encoding="utf-8"), version)
    with tempfile.TemporaryDirectory() as temporary:
        source = framework_source(root, version, Path(temporary))
        wheel = wheel_from(source, version, Path(temporary) / "wheel")
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        project.write_text(text, encoding="utf-8")
        line = framework_line(version, framework_extras(text)[1], digest)
        (folder / "framework.txt").write_text(line, encoding="utf-8")
        relock(root, source, folder)


def context_variable(application: str) -> str:
    return "FRAMEWORK_" + re.sub(r"\W", "_", application).upper()


def framework_contexts(root: Path, folder: Path) -> dict[str, str]:
    sources: dict[str, str] = {}
    digests: dict[str, str] = {}
    contexts: dict[str, str] = {}
    for application in sorted(path for path in (root / "application").iterdir() if path.is_dir()):
        version, pinned = pinned_framework(application)
        if version not in sources:
            source = framework_source(root, version, folder / version)
            wheel = wheel_from(source, version, folder / "wheels" / version)
            digests[version] = hashlib.sha256(wheel.read_bytes()).hexdigest()
            sources[version] = "." if source == root else str(source)
        if pinned != digests[version]:
            raise RefusalError(
                f"{application.name} pins eemls {version} with hash {pinned[:12]}, but the source "
                f"builds {digests[version][:12]}: run make pin APP={application.name} "
                f"VERSION={version} (or a new version)"
            )
        contexts[context_variable(application.name)] = sources[version]
    return contexts


def bake(root: Path, arguments: list[str]) -> int:
    with tempfile.TemporaryDirectory() as folder:
        contexts = framework_contexts(root, Path(folder))
        done = subprocess.run(
            ["docker", "buildx", "bake", *arguments], cwd=root, env=os.environ | contexts
        )
    return done.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pin")
    parser.add_argument("application", nargs="?")
    parser.add_argument("version", nargs="?")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bake", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.bake is None and not (args.application and args.version):
        parser.error("give an application and a version, or --bake with its arguments")
    try:
        if args.bake is not None:
            return bake(args.root, args.bake)
        pin(args.root, args.application, args.version)
    except RefusalError as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
