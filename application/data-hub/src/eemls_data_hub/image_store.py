from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from eemls.domain.errors import EemlsError
from eemls.domain.types import Condition, FrozenModel, Identifier, ImageQuality

MANIFEST = "manifest.jsonl"
TRAINED = "trained.txt"


class StoreError(EemlsError): ...


class NoSuchImageError(StoreError): ...


class StoredImage(FrozenModel):
    image_id: Identifier
    label: Identifier
    dataset: Identifier
    original_label: str
    condition: Condition
    image_quality: ImageQuality
    path: str


class ImageStore:
    def __init__(self, root: Path, *, max_image_bytes: int) -> None:
        self._root = root
        self._max_image_bytes = max_image_bytes
        self._held: dict[str, StoredImage] = {}
        self._by_label: dict[tuple[str, str, str], list[str]] = {}

    @classmethod
    def opened(cls, root: Path, *, max_image_bytes: int) -> ImageStore:
        store = cls(root, max_image_bytes=max_image_bytes)
        try:
            store._read_manifest()
        except (OSError, UnicodeDecodeError) as refused:
            raise StoreError(f"{root} could not be read: {refused}") from refused
        return store

    def __len__(self) -> int:
        return len(self._held)

    def meta(self, image_id: str) -> StoredImage:
        held = self._held.get(image_id)
        if held is None:
            raise NoSuchImageError(f"{image_id} is not an image this store holds")
        return held

    def bytes_of(self, image_id: str) -> bytes:
        held = self.meta(image_id)
        path = self._root / held.path
        try:
            body = path.read_bytes()
        except OSError as refused:
            raise StoreError(f"{image_id}: {path} could not be read: {refused}") from refused
        if len(body) > self._max_image_bytes:
            raise StoreError(
                f"{image_id}: {len(body)} bytes is over the {self._max_image_bytes} allowed"
            )
        return body

    def ids(self, *, label: str, condition: str, image_quality: str) -> Sequence[str]:
        """Image ids of one class, condition and quality, sorted."""
        return tuple(self._by_label.get((label, condition, image_quality), ()))

    def conditions(self) -> Sequence[str]:
        return tuple(sorted({held.condition for held in self._held.values()}))

    def labels(self) -> Sequence[str]:
        return tuple(sorted({held.label for held in self._held.values()}))

    def _read_manifest(self) -> None:
        path = self._root / MANIFEST
        if not path.is_file():
            raise StoreError(f"{path} is not there, so the store holds nothing")
        trained_path = self._root / TRAINED
        trained = (
            set(trained_path.read_text(encoding="utf-8").split())
            if trained_path.is_file()
            else set()
        )
        for number, row in _rows(path):
            held = _validated(row, path, number)
            if not _present(self._root / held.path, f"{path}:{number}: {held.image_id}"):
                raise StoreError(
                    f"{path}:{number}: {held.image_id} names {held.path}, which is not there"
                )
            if held.image_id in self._held:
                raise StoreError(f"{path}:{number}: {held.image_id} is held twice")
            self._held[held.image_id] = held
            if held.image_id in trained:
                continue
            self._by_label.setdefault((held.label, held.condition, held.image_quality), []).append(
                held.image_id
            )
        for ids in self._by_label.values():
            ids.sort()


def _rows(path: Path) -> Iterator[tuple[int, Mapping[str, object]]]:
    with path.open(encoding="utf-8") as held:
        for number, line in enumerate(held, 1):
            if not line.strip():
                continue
            try:
                yield number, json.loads(line)
            except json.JSONDecodeError as refused:
                raise StoreError(f"{path}:{number}: not a JSON object: {refused}") from refused


def _present(path: Path, named: str) -> bool:
    try:
        return path.is_file()
    except OSError as refused:
        raise StoreError(f"{named}: {path} could not be read: {refused}") from refused


def _validated(row: Mapping[str, object], path: Path, number: int) -> StoredImage:
    try:
        return StoredImage.model_validate(row)
    except ValueError as refused:
        raise StoreError(f"{path}:{number}: {refused}") from refused
