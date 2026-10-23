# eemls-control-plane

The control plane: it serves each query through the current ensemble and adapts the ensemble with an approach.

- Takes requests on `POST /classify`.
- Calls the ensemble, aggregates, publishes.
- Runs the decision loop.
- Sends its rows to the Observation Service and asks it to update the profiles.
- Reads recent windows from the metrics database.
- Package: `eemls_control_plane`. Commands: `eemls-control-plane`, `eemls-warm`, `eemls-collect`.
- Depends on the framework `eemls` with its `database` and `approaches` extras, FastAPI, HTTPX, Uvicorn and Pydantic.
- Approaches are found at start through the `eemls.approaches` entry points; the framework holds the built-in ones.
- Config folder: `--config <folder>`, else `EEMLS_CONFIG_DIR`; one of them is needed.
- `eemls-collect` writes a run's rows from the metrics database to files. Stop the control plane first, so its last rows are stored.
- Test: `poetry install && poetry run pytest`.
- Check: `poetry run ruff check . && poetry run mypy`.

## Image

- [`Dockerfile`](Dockerfile), built from this folder.
- Builds the framework wheel inside the image build, from the `framework` build context ([`docker-bake.hcl`](../../docker-bake.hcl)).
- Installs that wheel only if its hash is the one in [`framework.txt`](framework.txt), then this app.
- `make pin APP=<app> VERSION=<version>` at the repository root writes [`framework.txt`](framework.txt) and relocks this app.
- `RUNTIME`: the image below. Default `python:3.12-slim`.
- `APPROACH_PACKAGES`: build argument; extra approach packages installed into the image. Default none.
- Runs as uid 10001. Entry point: `eemls-control-plane`.
- Holds no config, data or models.

| Port | Serves |
|---|---|
| 8080 | HTTP API, `/healthz`, `/readyz`, `/metrics` |

- Mount the config folder at `/opt/eemls/config` (`EEMLS_CONFIG_DIR`).
- Needs `EEMLS_DATABASE_PASSWORD`, `EEMLS_SEED`.
- Also takes `EEMLS_RUN_ID`, `EEMLS_INSTANCE_ID`.

```bash
docker buildx bake control-plane --set control-plane.args.APPROACH_PACKAGES=<package>
```
