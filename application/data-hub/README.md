# eemls-data-hub

The data hub serves the images of the prepared dataset by id over HTTP.

- `GET /images/{id}`: bytes.
- `GET /images/{id}/meta`: class, dataset, original class, condition, quality.
- Package: `eemls_data_hub`. Command: `eemls-data-hub`.
- Depends on the framework `eemls`, FastAPI, Uvicorn and Pydantic.
- Reads `services/data_hub.yaml` and `services/monitoring.yaml` from the config folder.
- Config folder: `--config <folder>`, else `EEMLS_CONFIG_DIR`; one of them is needed.
- Test: `poetry install && poetry run pytest`.
- Check: `poetry run ruff check . && poetry run mypy`.

## Image

- [`Dockerfile`](Dockerfile), built from this folder.
- Builds the framework wheel inside the image build, from the `framework` build context ([`docker-bake.hcl`](../../docker-bake.hcl)).
- Installs that wheel only if its hash is the one in [`framework.txt`](framework.txt), then this app.
- `make pin APP=<app> VERSION=<version>` at the repository root writes [`framework.txt`](framework.txt) and relocks this app.
- `RUNTIME`: the image below. Default `python:3.12-slim`.
- Runs as uid 10001. Entry point: `eemls-data-hub`.
- Holds no config, data or models.

| Port | Serves |
|---|---|
| 8081 | HTTP API, `/healthz`, `/readyz`, `/metrics` |

- Mount the config folder at `/opt/eemls/config` (`EEMLS_CONFIG_DIR`).
- Mount images at `/var/lib/eemls/images` (`image_root`).

```bash
docker buildx bake data-hub
```
