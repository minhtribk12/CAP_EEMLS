#!/bin/sh
# Send one inference service a burst; print its CPU and copies every 10 s.
#
#   sh deploy/kube/burst.sh <service> <image id> [seconds=60] [parallel=8] [after=150]
set -eu
SERVICE="${1:?is the inference service to send the burst to}"
IMAGE_ID="${2:?is an image the data hub serves}"
LENGTH="${3:-60}"
PARALLEL="${4:-8}"
WATCH_AFTER_S="${5:-150}"
. deploy/kube/common.sh
current_run
PORT="$(sed -n 's/^port: *//p' "$OUT/config/services/inference_service.yaml")"
kubectl -n eemls delete pod eemls-burst --ignore-not-found >/dev/null
kubectl -n eemls run eemls-burst --image=curlimages/curl:8.11.1 --restart=Never --command -- sh -c "
end=\$((\$(date +%s) + $LENGTH))
while [ \$(date +%s) -lt \$end ]; do
  i=0
  while [ \$i -lt $PARALLEL ]; do
    curl -s -o /dev/null -X POST http://$SERVICE:$PORT/predict \
      -H 'content-type: application/json' -d '{\"image_id\": \"$IMAGE_ID\"}' &
    i=\$((i + 1))
  done
  wait
done" >/dev/null
echo "seconds  cpu%  copies  wanted"
elapsed=0
while [ "$elapsed" -le $((LENGTH + WATCH_AFTER_S)) ]; do
  printf '%7s  ' "$elapsed"
  kubectl -n eemls get hpa "$SERVICE" --no-headers -o \
    custom-columns='USE:.status.currentMetrics[0].resource.current.averageUtilization,COPIES:.status.currentReplicas,WANTED:.status.desiredReplicas'
  [ "$elapsed" -lt $((LENGTH + WATCH_AFTER_S)) ] || break
  sleep 10
  elapsed=$((elapsed + 10))
done
kubectl -n eemls delete pod eemls-burst --ignore-not-found >/dev/null
