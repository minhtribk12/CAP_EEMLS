#!/bin/sh
# Stop everything `make up` started. Rows stay in the volume.
set -eu
. deploy/one_host/common.sh
CONFIG=runs/one_host/config
stop_consumers
if [ -d "$CONFIG" ]; then
  stop_service "$(service_name control_plane "$CONFIG")"
  stop_inference_services "$CONFIG"
  for service in observation_service data_hub broker metrics_database; do
    stop_service "$(service_name "$service" "$CONFIG")"
  done
fi
docker stack rm "$STACK" >/dev/null 2>&1 || true
deadline=$(( $(date +%s) + 120 ))
until [ -z "$(docker ps -aq --filter "label=com.docker.stack.namespace=$STACK")" ] \
    && [ -z "$(docker network ls --filter "name=${STACK}_eemls" -q)" ]; do
  [ "$(date +%s)" -lt "$deadline" ] || { echo "the stack is still stopping after 120 s"; exit 1; }
  sleep 1
done
rm -f runs/one_host/run_id
echo "down: stack $STACK removed"
