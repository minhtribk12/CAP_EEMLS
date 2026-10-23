# EEMLS

Ensemble ML serving on edge devices, adapted at runtime by the CDM-Adaptive Policy (CAP).

- Several consumers send image classification queries.
- Each consumer is served by an ensemble of inference services.
  - An inference service is one model on one device type.
- Each consumer states a contract: the metrics it cares about (consumer-defined metrics, CDMs), their weights, and a cost weight.
- A policy watches those metrics and changes the ensemble while the system runs.

## Architecture

| Service | Does |
|---|---|
| Consumer | Sends queries with an image id. Gets results. Returns feedback on some. |
| Control plane | Calls the ensemble's members, takes the vote, publishes the result, sends its rows to the Observation Service, and lets the policy change the ensemble. |
| Inference service | Fetches the image, runs its model, answers label and confidence. |
| Data hub | Serves dataset images by id. |
| Observation Service | Stores the rows of a run in the metrics database and updates the inference service profiles. |
| Broker | Carries results to consumers over MQTT. |

- A metrics database (PostgreSQL) stores every query, every member answer and every decision.
- Every part is a real service. Only the consumers' load is generated.

## Layout

| Path | Holds |
|---|---|
| [`application/`](application/) | one folder per service: its code, tests and Dockerfile |
| [`src/eemls/`](src/eemls/) | the framework every service installs; see [`src/eemls/README.md`](src/eemls/README.md) |
| [`tools/`](tools/) | config, deploy and experiment commands; model, dataset and profile scripts |
| [`config/`](config/) | all settings |
| [`experiments/`](experiments/) | scenarios and experiment plans |
| [`examples/`](examples/) | examples of the local files and of overrides |
| [`deploy/`](deploy/) | one host (Docker), Kubernetes and k3s |
| [`docs/`](docs/) | how to run it |

## Start

- Install: Python 3.11 or 3.12 and Poetry, then `make install`: the framework, every service and the tools.
- First run, on one machine, with public models: the quick start in [`docs/run.md`](docs/run.md#quick-start).
- How to run everything by hand: [`docs/run.md`](docs/run.md).
- Checks: `make check` (lint, types, spelling, tests). Coverage: `make coverage`.

## License

Apache-2.0, see [`LICENSE`](LICENSE).
