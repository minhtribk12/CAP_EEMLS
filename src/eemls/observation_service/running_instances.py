from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from eemls.domain.records import Instance
from eemls.messaging.instance_announcements import InstanceAnnouncement


class InstanceRecords(Protocol):
    def register_instance(self, instance: Instance) -> None: ...

    def write_instance_stopped(self, instance_id: str, stopped_at: datetime) -> None: ...


def instance_recorder(
    database: InstanceRecords, *, clock: Callable[[], datetime]
) -> Callable[[InstanceAnnouncement], None]:
    def recorded(announcement: InstanceAnnouncement) -> None:
        database.register_instance(announcement.instance)
        if announcement.state == "stopped" and announcement.instance.stopped_at is not None:
            database.write_instance_stopped(
                announcement.instance.instance_id, announcement.instance.stopped_at
            )
        elif announcement.state == "lost":
            database.write_instance_stopped(announcement.instance.instance_id, clock())

    return recorded
