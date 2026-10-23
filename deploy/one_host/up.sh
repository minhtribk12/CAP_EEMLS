#!/bin/sh
# Start the system on this host as a Docker Swarm stack, load profiles, warm it up.
#
#   EEMLS_DATABASE_PASSWORD=... make up
set -eu
. deploy/one_host/common.sh
need_password
[ -f config/local/models.yaml ] || { echo "no config/local/models.yaml: run tools/new_local.py and fill it in"; exit 1; }
RUN_ID="${RUN_ID:-one-host-$(date +%Y%m%d-%H%M%S)}"
check_run_id "$RUN_ID"
OUT=runs/one_host
READY_SECONDS="${READY_SECONDS:-300}"
VOLUME="$(volume_of "$RUN_ID")"
if docker volume inspect "$VOLUME" >/dev/null 2>&1; then
  echo "run id $RUN_ID has a database volume already; pick another RUN_ID or remove the volume"; exit 1
fi
# A start that stops early removes what it made.
trap '[ $? -eq 0 ] || { sh deploy/one_host/down.sh >/dev/null 2>&1; docker volume rm "$VOLUME" >/dev/null 2>&1; }' EXIT
OVERRIDES="${OVERRIDES:-deploy/one_host/overrides.yaml}"
SERVICE_PROFILES="${SERVICE_PROFILES:-models/profiles.jsonl}"
WARM_IMAGES="${WARM_IMAGES:-50}"
SEED="${SEED:-0}"

if [ "$(docker info --format '{{.Swarm.LocalNodeState}}')" != active ]; then
  docker swarm init --advertise-addr 127.0.0.1 >/dev/null
fi
rm -rf "$OUT/config"
mkdir -p "$OUT"
config_into "$OUT/config" "$OVERRIDES"

cp config/local/models.yaml models/models.yaml

labels=""
for label in $(poetry run eemls-deploy labels --config "$OUT/config" --core); do
  labels="$labels --label-add $label"
done
docker node update $labels "$(docker info --format '{{.Swarm.NodeID}}')" >/dev/null
swarm_stack
CONTROL_PLANE="$(service_name control_plane "$OUT/config")"
OBSERVATION_SERVICE="$(service_name observation_service "$OUT/config")"
echo "$RUN_ID" > "$OUT/run_id"
echo "$SERVICE_PROFILES" > "$OUT/profiles"
echo "$SEED" > "$OUT/seed"
case "$SERVICE_PROFILES" in /*) SERVICE_PROFILES_PATH="$SERVICE_PROFILES" ;; *) SERVICE_PROFILES_PATH="$PWD/$SERVICE_PROFILES" ;; esac

# Check every local image exists.
for image in $(poetry run python3 -c "import sys, yaml; print(*sorted({s['image'] for s in yaml.safe_load(open(sys.argv[1]))['services'].values()}))" "$OUT/stack.yaml"); do
  case "$image" in "$REGISTRY"/*) docker image inspect "$image" >/dev/null 2>&1 || { echo "no image $image on this host; build it with make up"; exit 1; } ;; esac
done
docker stack deploy --detach=true --resolve-image never -c "$OUT/stack.yaml" "$STACK" >/dev/null

deadline=$(( $(date +%s) + READY_SECONDS ))
load_profiles_until "$OUT/load_profiles.log" "$deadline" docker run --rm --network "${STACK}_eemls" -e EEMLS_DATABASE_PASSWORD \
  -v "$PWD/tools:/opt/tools:ro" -v "$SERVICE_PROFILES_PATH:/opt/profiles.jsonl:ro" \
  -v "$PWD/$OUT/config:/opt/eemls/config:ro" --entrypoint python \
  "$REGISTRY/eemls-control-plane:latest" /opt/tools/load_profiles.py \
  --run "$RUN_ID" --profiles /opt/profiles.jsonl --config /opt/eemls/config

for service in "$OBSERVATION_SERVICE" "$CONTROL_PLANE"; do
  until [ "$(docker ps --filter "name=${STACK}_$service" --filter health=healthy -q)" ]; do
    [ "$(date +%s)" -lt "$deadline" ] || { echo "$service never became ready"; exit 1; }
    sleep 2
  done
done
docker run --rm --network "${STACK}_eemls" -v "$PWD/$OUT/config:/opt/eemls/config:ro" \
  --entrypoint eemls-warm "$REGISTRY/eemls-control-plane:latest" \
  --config /opt/eemls/config --images "$WARM_IMAGES" --seed "$SEED"
echo "up: stack $STACK, run $RUN_ID"
