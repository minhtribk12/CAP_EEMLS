#!/bin/sh
# Copy one device's models and models.yaml to it.
#
#   [OVERRIDES=<file>] sh deploy/edge/models.sh <username>@<host> jetson_orin_nano [/srv/eemls/models]
set -eu
HOST="${1:?is the device to copy to, as user@host}"
DEVICE_TYPE="${2:?is the device type, as config/cost/prices.yaml names it}"
DEST="${3:-/srv/eemls/models}"
[ -f config/local/models.yaml ] || { echo "no config/local/models.yaml: run tools/new_local.py and fill it in"; exit 1; }
CONFIG=config
if [ -n "${OVERRIDES:-}" ]; then
  scratch="$(mktemp -d)"
  trap 'rm -rf "$scratch"' EXIT
  CONFIG="$scratch/config"
  poetry run eemls-config render config --overrides "$OVERRIDES" --out "$CONFIG" >/dev/null
fi
models="$(poetry run python3 -c "
import sys
from eemls_tools.config_tree import load_config
from eemls_tools.deployment_plan import models_on
print(*models_on(load_config(sys.argv[2]), sys.argv[1]))
" "$DEVICE_TYPE" "$CONFIG")"
[ -n "$models" ] || { echo "no service is placed on $DEVICE_TYPE"; exit 1; }
ssh "$HOST" "mkdir -p '$DEST'"
for model in $models; do
  [ -d "models/$model" ] || { echo "models/$model is not here: build or fetch it first"; exit 1; }
  rsync -a --delete "models/$model/" "$HOST:$DEST/$model/"
done
rsync -a config/local/models.yaml "$HOST:$DEST/models.yaml"
echo "$HOST: $models in $DEST"
