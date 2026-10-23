# eemls-consumer

The consumer sends queries to the control plane and reads the results.
The load tool sends queries at fixed rates and reports what came back.

- Sends one consumer's requests, on schedule.
- Sends back true labels for the share the contract sets.
- Labels go back once every decision interval, within delay + interval − 1 queries.
- Exits after a set time.
- Package: `eemls_consumer`. Commands: `eemls-consumer`, `eemls-load`.
- Depends on the framework `eemls` (extra `database`), FastAPI, HTTPX, Uvicorn, Pydantic and PyYAML.
- Reads `services/consumer.yaml`, `services/monitoring.yaml`, `services/broker.yaml` and `consumers/` from the config folder.
- The load tool also reads `services/control_plane.yaml`, `services/metrics_database.yaml` and `datasets/sources.yaml`.
- Config folder: `--config <folder>`, else `EEMLS_CONFIG_DIR`; one of them is needed.
- Test: `poetry install && poetry run pytest`.
- Check: `poetry run ruff check . && poetry run mypy`.

## Image

- [`Dockerfile`](Dockerfile), built from this folder.
- Builds the framework wheel inside the image build, from the `framework` build context ([`docker-bake.hcl`](../../docker-bake.hcl)).
- Installs that wheel only if its hash is the one in [`framework.txt`](framework.txt), then this app.
- `make pin APP=<app> VERSION=<version>` at the repository root writes [`framework.txt`](framework.txt) and relocks this app.
- `RUNTIME`: the image below. Default `python:3.12-slim`.
- Runs as uid 10001. Entry point: `eemls-consumer`.
- Holds no config, data or models.

| Port | Serves |
|---|---|
| 9103 | `/metrics` |

- Takes `--consumer`, `--scenario`, `--duration-s` and `--seed`; all required.
- Mount the config folder at `/opt/eemls/config` (`EEMLS_CONFIG_DIR`).
- Mount the scenario files at `/opt/eemls/scenarios`.

```bash
docker buildx bake consumer
```
