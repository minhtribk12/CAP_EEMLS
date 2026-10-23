#!/bin/sh
# Drive one consumer at rising rates; report to runs/one_host/load.json.
#
#   RATES=10,20,50,100 STEP_S=30 CONSUMER=c3_surveillance make load
set -eu
. deploy/one_host/common.sh
need_password
load_defaults
OUT=runs/one_host
[ -f "$OUT/run_id" ] || { echo "no system is up; run make up first"; exit 1; }
RUN_ID="$(cat "$OUT/run_id")"

docker run --rm --network "${STACK}_eemls" --user "$(id -u):$(id -g)" -e EEMLS_DATABASE_PASSWORD \
  -v "$PWD/$OUT/config:/opt/eemls/config:ro" -v "$PWD/$OUT:/opt/eemls/out" \
  --entrypoint eemls-load "$REGISTRY/eemls-consumer:latest" \
  --consumer "$CONSUMER" --rates "$RATES" --step-s "$STEP_S" --run-id "$RUN_ID" \
  --out /opt/eemls/out/load.json
