from __future__ import annotations

from datetime import datetime, timedelta

import httpx

from eemls.config.reading import Reader, validate
from eemls.config.services import MonitoringConfig
from eemls.domain.cost import DeviceRate
from eemls.metrics_database.buffered_writer import WriteError
from eemls.metrics_database.testing import MemoryMetricsDatabase
from eemls.monitoring.metrics import Monitoring
from eemls.observation_service.observation_service_api import ProfileUpdateAnswer, RowBatch
from eemls.observation_service.profile_updates import ProfileSettings
from eemls.testing.records import AT, decision, feedback, prediction, profile, served
from eemls.testing.trees import TREE
from eemls_observation_service.http_app import build_app

OBSERVATION = "services/monitoring.yaml"
RATES = {
    "jetson_orin_nano": DeviceRate(
        power_watts=20.0,
        hardware_price=699.0,
        lifetime_years=3,
        energy_price_per_kwh=0.20,
        eur_per_cost_unit=0.00001,
    )
}
SERVICE = "infer-b6-jetson_orin_nano"


def settings_at(at: datetime) -> ProfileSettings:
    return ProfileSettings(
        window_start=at - timedelta(seconds=60),
        window_end=at,
        rates=RATES,
        min_labeled_per_class=1,
        min_labeled_overall=1,
    )


class Refusing(MemoryMetricsDatabase):
    def flush(self) -> None:
        raise WriteError("the store refused 5 rows: the database went away")


def client(database: MemoryMetricsDatabase) -> httpx.AsyncClient:
    monitoring = Monitoring.from_config(
        validate(MonitoringConfig, Reader(TREE).read_file(OBSERVATION), OBSERVATION)
    )
    app = build_app(
        database=database, run_id="run-1", settings_at=settings_at, monitoring=monitoring
    )
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://observation-service"
    )


def body(run_id: str = "run-1") -> dict[str, object]:
    return RowBatch(
        run_id=run_id,
        sender="control-plane-1",
        queries=(served(),),
        member_predictions=(prediction(), prediction(in_ensemble=False)),
        feedback=(feedback(),),
        decisions=(decision(),),
    ).model_dump(mode="json")


class TestStoringRows:
    async def test_a_batch_is_stored_and_counted(self) -> None:
        database = MemoryMetricsDatabase()

        answer = await client(database).post("/rows", json=body())

        assert answer.status_code == 200
        assert answer.json() == {"stored": 5}
        assert list(database.queries) == ["q-1"]
        assert len(database.predictions) == 2
        assert list(database.feedback) == ["q-1"]
        assert list(database.decisions) == ["d-1"]

    async def test_a_batch_posted_twice_is_stored_once(self) -> None:
        database = MemoryMetricsDatabase()
        held = client(database)

        await held.post("/rows", json=body())
        answer = await held.post("/rows", json=body())

        assert answer.status_code == 200
        assert len(database.read_rows("queries")) == 1
        assert len(database.read_rows("member_predictions")) == 2

    async def test_a_batch_of_another_run_is_refused(self) -> None:
        database = MemoryMetricsDatabase()

        answer = await client(database).post("/rows", json=body("run-2"))

        assert answer.status_code == 409
        assert database.queries == {}

    async def test_a_batch_the_database_refuses_is_answered_unavailable(self) -> None:
        answer = await client(Refusing()).post("/rows", json=body())

        assert answer.status_code == 503

    async def test_a_batch_that_is_not_one_is_refused(self) -> None:
        answer = await client(MemoryMetricsDatabase()).post("/rows", json={"run_id": "run-1"})

        assert answer.status_code == 422


class TestUpdatingProfiles:
    def stored(self) -> MemoryMetricsDatabase:
        database = MemoryMetricsDatabase()
        database.write_profile(
            profile(
                service_name=SERVICE,
                window_start=AT - timedelta(minutes=10),
                window_end=AT - timedelta(minutes=5),
            )
        )
        arrived = AT + timedelta(seconds=1)
        database.write_query(
            served(t_arrive=arrived, t_result_ready=arrived + timedelta(milliseconds=174))
        )
        database.write_member_predictions((prediction(predicted_label="bus", total_ms=30.0),))
        database.write_feedback(feedback(true_label="car"))
        return database

    async def test_the_updated_profiles_are_answered_and_stored(self) -> None:
        database = self.stored()
        at = AT + timedelta(seconds=40)

        answer = await client(database).post("/profiles/update", json={"at": at.isoformat()})

        assert answer.status_code == 200
        updated = ProfileUpdateAnswer.model_validate(answer.json()).profiles
        assert [(one.service_name, one.window_end) for one in updated] == [(SERVICE, at)]
        assert updated[0].general_accuracy == 0.0
        assert updated[0] in database.profiles.values()

    async def test_an_update_the_database_refuses_is_answered_unavailable(self) -> None:
        database = Refusing()

        answer = await client(database).post("/profiles/update", json={"at": AT.isoformat()})

        assert answer.status_code == 503


class TestItsHealth:
    async def test_it_is_alive_and_ready(self) -> None:
        held = client(MemoryMetricsDatabase())

        assert (await held.get("/healthz")).status_code == 200
        assert (await held.get("/readyz")).status_code == 200

    async def test_it_exposes_its_metrics(self) -> None:
        assert (await client(MemoryMetricsDatabase()).get("/metrics")).status_code == 200
