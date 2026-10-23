# eemls-observation-service

The Observation Service stores the rows of a run and updates the inference service profiles.

- `POST /rows`: a batch `{run_id, sender, queries, member_predictions, feedback, decisions}`. Answers `200 {stored}` after commit, `409` for another run, `503` when the database refuses.
- `POST /profiles/update`: `{at}`. Updates the profiles over the window that ends at `at` and answers `{profiles}`.
- Records the running instances from their retained announcements on the broker.
- Package: `eemls_observation_service`. Command: `eemls-observation-service`.
- Depends on the framework `eemls` with `database`, FastAPI, Uvicorn and Pydantic.
- Reads `services/observation_service.yaml`, `services/metrics_database.yaml`, `services/monitoring.yaml`, `services/broker.yaml` and `cost/prices.yaml` from the config folder.
- Config folder: `--config <folder>`, else `EEMLS_CONFIG_DIR`; one of them is needed.
- Settings in `services/observation_service.yaml`: `url` (how the other services reach it), `host`, `port` (8083), `request_timeout_ms`, `profile_window_s`, `profile_min_labeled_per_class`, `profile_min_labeled_overall`.
- Runs on the core node. It starts before the control plane and stops after it.
- Needs `EEMLS_DATABASE_PASSWORD`. Also takes `EEMLS_RUN_ID`, `EEMLS_INSTANCE_ID`.
- Test: `poetry install && poetry run pytest`.
- Check: `poetry run ruff check . && poetry run mypy`.

## Image

- [`Dockerfile`](Dockerfile), built from this folder.
- Builds the framework wheel inside the image build, from the `framework` build context ([`docker-bake.hcl`](../../docker-bake.hcl)).
- Installs that wheel only if its hash is the one in [`framework.txt`](framework.txt), then this app.
- `make pin APP=<app> VERSION=<version>` at the repository root writes [`framework.txt`](framework.txt) and relocks this app.
- `RUNTIME`: the image below. Default `python:3.12-slim`.
- Runs as uid 10001. Entry point: `eemls-observation-service`.
- Holds no config, data or models.

| Port | Serves |
|---|---|
| 8083 | HTTP API, `/healthz`, `/readyz`, `/metrics` |

- Mount the config folder at `/opt/eemls/config` (`EEMLS_CONFIG_DIR`).
