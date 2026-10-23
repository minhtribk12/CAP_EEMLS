#!/bin/sh
# Stop everything `up.sh` started. DELETE_VOLUME=1 also deletes this run's database volume.
set -eu
. deploy/kube/common.sh
current_run
kubectl -n eemls delete job --all --ignore-not-found --wait=true >/dev/null 2>&1 || true
stop_deployment "$(service_name control_plane 2>/dev/null || true)" 2>/dev/null || true
stop_inference_services 2>/dev/null || true
for service in observation_service data_hub broker metrics_database; do
  stop_deployment "$(service_name "$service" 2>/dev/null || true)" 2>/dev/null || true
done
kubectl delete -k "$OUT/kube" --ignore-not-found >/dev/null
if [ "${DELETE_VOLUME:-0}" = 1 ]; then
  kubectl -n eemls delete pvc "metrics-database-$RUN_ID" --ignore-not-found >/dev/null 2>&1 || true
fi
rm -f "$CURRENT"
echo "down: run $RUN_ID"
