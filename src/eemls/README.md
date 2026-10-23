# eemls

The framework: one package, `eemls`. The applications in [application](../../application) install it.

## Modules

| Module | What it holds |
|---|---|
| [domain](domain) | records, types, contracts, consumer-defined metrics, cost |
| [config](config) | the data models of the YAML config, how it is read and checked |
| [monitoring](monitoring) | the declared Prometheus metrics and the `/metrics` endpoint |
| [messaging](messaging) | the MQTT broker, topics, result listener, instance announcements |
| [image_catalogue](image_catalogue) | reads the image catalogue of the Data Hub |
| [interfaces](interfaces) | the HTTP request and answer models of the control plane, Data Hub and inference service |
| [testing](testing) | test helpers: broker fake, records, metrics, config trees, images, a clean `EEMLS_` environment |
| [evaluation_engine](evaluation_engine) | evaluates the consumer-defined metrics over a window, scores them against the contract, consumer evaluation, profile estimates |
| [metrics_database](metrics_database) | table schema, PostgreSQL store, recent windows, buffered writes; `testing` is a fake with the same interface |
| [observation_service](observation_service) | runtime context, operating conditions, consumer feedback, running instances, profile updates, idle service profiling, the Observation Service API, metrics reporting and metrics storage; `testing` is an in-process Observation Service |
| [orchestration](orchestration) | query forwarding, aggregation, result delivery, ensemble deployments; `testing` holds fake members |
| [ensemble_optimization](ensemble_optimization) | the interface (`Policy`, `Approach`, `Observation`, `PolicyDecision`), approach loading, the decision loop, the conformance kit in `testing` |

## Approaches

Each approach registers under the entry point group `eemls.approaches`, in [pyproject.toml](../../pyproject.toml). An approach from another package registers the same way.

- [CAP](ensemble_optimization/approaches/cap): selects the ensemble for each consumer from the context; entry point `cap`.
- [Baselines](ensemble_optimization/approaches/baselines): scale reduction, SneakPeek, AxiomVision, Cocktail, FrugalML and keep the ensemble; one entry point each.

## Extras

| Extra | Installs | Needed by |
|---|---|---|
| `database` | psycopg | `metrics_database.postgres` |
| `approaches` | numpy | `ensemble_optimization.approaches.cap` |
| `testing` | pillow, pytest | `testing.images` |

## Test it

From the repository root:

```bash
make install
poetry run pytest tests/framework
```
