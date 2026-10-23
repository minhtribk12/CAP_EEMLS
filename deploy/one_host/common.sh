# Helpers shared by the one-host scripts.
#
#   . deploy/one_host/common.sh
. deploy/common.sh

REGISTRY=eemls.local
STACK=eemls

volume_of() {
  echo "${STACK}_metrics-database-$1"
}

# The host name of service $1 in the tree $2.
service_name() {
  poetry run eemls-deploy name --config "$2" "$1"
}

# Remove the stack's service $1, if named, and wait until its containers are gone.
stop_service() {
  [ -n "$1" ] || return 0
  docker service rm "${STACK}_$1" >/dev/null 2>&1 || return 0
  stopping_until=$(( $(date +%s) + 120 ))
  until [ -z "$(docker ps -aq --filter "label=com.docker.swarm.service.name=${STACK}_$1")" ]; do
    [ "$(date +%s)" -lt "$stopping_until" ] || { echo "$1 is still stopping after 120 s"; exit 1; }
    sleep 1
  done
}

# Remove every consumer service and wait until its containers are gone.
stop_consumers() {
  for service in $(docker service ls --filter "name=${STACK}_consumer-" --format '{{.Name}}'); do
    stop_service "${service#"${STACK}_"}"
  done
}

# Remove every service of the stack in the tree $1 but the consumers and the core services.
stop_inference_services() {
  core=" $(for service in control_plane observation_service data_hub broker metrics_database; do
    service_name "$service" "$1"; done | tr '\n' ' ') "
  for service in $(docker service ls --filter "label=com.docker.stack.namespace=$STACK" --format '{{.Name}}'); do
    service="${service#"${STACK}_"}"
    case "$service" in consumer-*) continue ;; esac
    case "$core" in *" $service "*) ;; *) stop_service "$service" ;; esac
  done
}

# Write $OUT/stack.yaml for RUN_ID, SERVICE_PROFILES and SEED; more options follow.
swarm_stack() {
  poetry run eemls-deploy swarm --config "$OUT/config" --registry "$REGISTRY" --version latest --profiles "$SERVICE_PROFILES" \
    --images "$PWD/data/hub" --models "$PWD/models" --mount-config "$PWD/$OUT/config" --run-id "$RUN_ID" \
    --seed "$SEED" --out "$OUT/stack.yaml" "$@"
}
