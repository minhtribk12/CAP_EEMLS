# eemls-tools

Tools for EEMLS: check and render the configuration, deploy the services and run the experiments.

- Package: `eemls_tools`. Commands: `eemls-config`, `eemls-deploy`, `eemls-exp`.
- Depends on the framework `eemls[database,approaches,testing]` and every application: eemls-inference-service, eemls-data-hub, eemls-consumer, eemls-control-plane, eemls-observation-service.
- The scripts (`*.py` in this folder) sit next to [src](src) and [tests](tests); their tests live in [tests](tests).
- Tests read the committed [config](../config), [deploy](../deploy) and [experiments](../experiments) folders of the repository.
- Test: `make install` at the repository root, then `poetry run pytest` from this folder.

## Commands

| Command | Does |
|---|---|
| `eemls-config check`, `render` | check a configuration tree; write a tree with overrides applied |
| `eemls-deploy swarm`, `kube` | write the Docker Swarm stack file or the Kubernetes files of a run |
| `eemls-deploy labels` | print the node labels of a device type; `--core` adds the core label |
| `eemls-deploy name` | print the host name of one service |
| `eemls-deploy pod` | write a one-off pod on a core node |
| `eemls-exp plan`, `render` | list the runs of an experiment plan; write the configuration of one approach |

- `eemls-config` takes the tree as its first argument.
- `eemls-deploy` needs `--config <folder>`.
- `eemls-exp` takes `--config <folder>`; default `EEMLS_CONFIG_DIR`, else [config](../config).

## Release

| Command | Does |
|---|---|
| `make pin APP=<app> VERSION=<version>` | build the framework wheel of that version once, write its pin and hash into [`application/<app>/framework.txt`](../application), point the app's project at that version and relock it |
| `make release VERSION=<version>` | tag the commit `eemls-v<version>`; the version must be the one of [pyproject.toml](../pyproject.toml), with no uncommitted change under [src](../src) or in [pyproject.toml](../pyproject.toml) |

- The root version builds from the working tree; any other version builds from its tag `eemls-v<version>`.
- A build is byte-identical, so the wheel each image builds matches the hash in `framework.txt`.
- `make images` and `make images-local` bake through [`pin.py`](pin.py) `--bake`: it gives each app the root as its `framework` build context, or a checkout of the tag it pins; it refuses an app whose tag is missing, or whose `framework.txt` hash is not the hash its source builds.
- An app's own Poetry lock takes the framework from the repository root (`path = "../.."`).
