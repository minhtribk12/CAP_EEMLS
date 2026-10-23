#!/bin/sh
# Drive one consumer at rising rates; report to runs/kube/<run>/load.txt.
#
#   RATES=10,20,50,100 STEP_S=30 CONSUMER=c3_surveillance sh deploy/kube/load_test.sh
set -eu
. deploy/kube/common.sh
current_run
load_defaults
kubectl -n eemls get deploy "$(service_name control_plane)" >/dev/null 2>&1 \
  || { echo "the control plane is not running; run deploy/kube/down.sh and deploy/kube/up.sh first"; exit 1; }

status=0
TAIL=200 in_cluster eemls-load consumer eemls-load --consumer "$CONSUMER" --rates "$RATES" \
  --step-s "$STEP_S" --run-id "$RUN_ID" > "$OUT/load.txt" || status=$?
cat "$OUT/load.txt"
exit "$status"
