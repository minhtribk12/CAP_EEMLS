from __future__ import annotations

from collections.abc import Iterator, Sequence
from types import MappingProxyType
from typing import overload

from eemls.domain.types import Row


class Window(Sequence[Row]):
    """Read-only rows, oldest first."""

    def __init__(self, rows: Sequence[Row]) -> None:
        self.rows: tuple[Row, ...] = tuple(
            row if isinstance(row, MappingProxyType) else MappingProxyType(dict(row))
            for row in rows
        )

    def __len__(self) -> int:
        return len(self.rows)

    @overload
    def __getitem__(self, index: int) -> Row: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Row, ...]: ...

    def __getitem__(self, index: int | slice) -> Row | tuple[Row, ...]:
        return self.rows[index]

    def __iter__(self) -> Iterator[Row]:
        return iter(self.rows)

    def last(self, count: int) -> Window:
        return Window(self.rows[-count:] if count < len(self.rows) else self.rows)

    def labeled(self, attribute: str) -> Window:
        return Window(tuple(row for row in self.rows if row.get(attribute) is not None))


def class_share(window: Window, classes: Sequence[str], *, attribute: str) -> tuple[float, ...]:
    held = [str(row[attribute]) for row in window if row.get(attribute)]
    if not held:
        return tuple(0.0 for _ in classes)
    return tuple(held.count(name) / len(held) for name in classes)
