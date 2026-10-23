# eemls-inference-service

The inference service. It loads one ONNX model and answers each request with a ranked prediction.

- Package: `eemls_inference_service`. Command: `eemls-inference-service`.
- Depends on the framework `eemls`, ONNX Runtime, FastAPI, Uvicorn, httpx, Pillow and NumPy.
- Reads its settings from the config folder.
- Config folder: `--config <folder>`, else `EEMLS_CONFIG_DIR`; one of them is needed.

Test it from this folder:

```bash
poetry install
poetry run pytest -m "not integration"
```

The real-model tests skip when the model files or [`config/local/models.yaml`](../../config/local/models.yaml) are missing.

## Image

- [`Dockerfile`](Dockerfile), built from this folder.
- Builds the framework wheel inside the image build, from the `framework` build context ([`docker-bake.hcl`](../../docker-bake.hcl)).
- Installs that wheel only if its hash is the one in [`framework.txt`](framework.txt), then this app.
- `make pin APP=<app> VERSION=<version>` at the repository root writes [`framework.txt`](framework.txt) and relocks this app.
- Runs as uid 10001. Entry point: `eemls-inference-service`.
- Holds no config, data or models.
- One copy of one inference service. `EEMLS_SERVICE_NAME` picks it.
- Mount the config folder at `/opt/eemls/config` (`EEMLS_CONFIG_DIR`).
- Mount the device's models at `/opt/eemls/models`, with [`models.yaml`](../../config/local/models.yaml) (the file digests).
  - Kubernetes: `EEMLS_MODEL_SOURCES=/opt/eemls/models/models.yaml`.
  - Swarm: mounted at `/opt/eemls/config/local/models.yaml`.
- Also takes `EEMLS_INSTANCE_ID`, `EEMLS_NODE`, `EEMLS_REPLICA`.

| Port | Serves |
|---|---|
| 8082 | `/predict`, `/healthz`, `/readyz`, `/metrics` |

## Tags

- `RUNTIME`: the image below. Default `python:3.12-slim`.
- `ONNX_RUNTIME`: the pip arguments that install another ONNX Runtime instead of the CPU one. Empty keeps the CPU one.
- `cuda-amd64` installs the hashed [`cuda-requirements.txt`](cuda-requirements.txt).

| Tag (`eemls-inference-service:`) | Runs on |
|---|---|
| `cpu-amd64` | x86 CPU |
| `cpu-arm64` | ARM CPU, Raspberry Pi |
| `cuda-amd64` | x86 with NVIDIA GPU |
| `jetson` | Jetson Orin |

## CUDA tag

- CUDA 13, cuDNN 9. Volta or newer.
- Older GPU (Pascal): serves on CPU, logs `provider_unavailable`.
- No driver: still serves, on CPU.
- Node needs driver 580+ and NVIDIA Container Toolkit. Run with `--gpus all`.
- `pip check` warns about `onnxruntime`. Safe: `onnxruntime-gpu` replaces it.

## Jetson tag

- Jetson runtime for inference, plus its ONNX Runtime. Build both on a board.

| File | Builds |
|---|---|
| [`jetson.Dockerfile`](jetson.Dockerfile) | Ubuntu 22.04 with Python 3.12 and only the CUDA 12.6, cuDNN 9.3 and TensorRT 10.3 libraries ONNX Runtime loads, taken from JetPack r36.4.0 (about 2.5 GB) |
| [`onnxruntime.Dockerfile`](onnxruntime.Dockerfile) | ONNX Runtime `ORT_JETSON` ([`docker-bake.hcl`](../../docker-bake.hcl)) for the Orin GPU (`sm_87`), CUDA and TensorRT, into [`onnxruntime-wheels/`](onnxruntime-wheels/) |

```bash
docker buildx bake onnxruntime-jetson   # once: hours on an Orin Nano, writes onnxruntime-wheels/
docker buildx bake --load inference-service-jetson
```

- Board must run JetPack 6 with L4T R36.4.x, the same line as `jetpack_release` in [`config/services/deployment.yaml`](../../config/services/deployment.yaml). A board on 36.4.7 works.
- Run with `--runtime nvidia`.
- On Kubernetes, `runtime_classes` in [`config/services/deployment.yaml`](../../config/services/deployment.yaml) gives the GPU tags the runtime class `nvidia`.
- ONNX Runtime here is older than on x86 ([`pyproject.toml`](pyproject.toml)): later versions need CUDA 12.8,
  newer than JetPack 6's 12.6.
- Builds faster on the AGX Orin: `--set onnxruntime-jetson.args.PARALLEL=4`.
