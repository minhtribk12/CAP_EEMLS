from __future__ import annotations

from pydantic import computed_field

from eemls.config.schema import ExtraFieldsIgnored


class DatasetClasses(ExtraFieldsIgnored):
    dominant_classes: tuple[str, ...]
    tail_classes: tuple[str, ...]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def classes(self) -> tuple[str, ...]:
        return self.dominant_classes + self.tail_classes
