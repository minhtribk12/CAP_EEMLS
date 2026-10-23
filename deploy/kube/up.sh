#!/bin/sh
# Start the system on Kubernetes, load profiles, warm it up.
#
#   EEMLS_DATABASE_PASSWORD=... sh deploy/kube/up.sh
set -eu
. deploy/kube/common.sh
need_password
OVERRIDES="${OVERRIDES:-deploy/kube/overrides.yaml}"
SERVICE_PROFILES="${SERVICE_PROFILES:-models/profiles.jsonl}"
IMAGES="${IMAGES:-/srv/eemls/images}"
WARM_IMAGES="${WARM_IMAGES:-50}"
SEED="${SEED:-0}"
RUN_ID="${RUN_ID:-kube-$(date +%Y%m%d-%H%M%S)}"
check_run_id "$RUN_ID"
[ ! -f "$CURRENT" ] || { echo "run $(cat "$CURRENT") is up; run deploy/kube/down.sh first"; exit 1; }
RUN_ID="$(free_run_id "$RUN_ID")"
echo "run id: $RUN_ID"
OUT="runs/kube/$RUN_ID"

mkdir -p "$OUT"
config_into "$OUT/config" "$OVERRIDES" >/dev/null
# Local tree: database at the forwarded port.
printf 'services/metrics_database.yaml:\n  host: 127.0.0.1\n  port: %s\n' "$DATABASE_FORWARD_PORT" > "$OUT/here.yaml"
poetry run eemls-config render "$OUT/config" --overrides "$OUT/here.yaml" --out "$OUT/here" >/dev/null
poetry run eemls-deploy kube --config "$OUT/config" --mount-config "$OUT/config" \
  --profiles "$SERVICE_PROFILES" --images "$IMAGES" --run-id "$RUN_ID" --out "$OUT/kube" --seed "$SEED" >/dev/null
echo "$SERVICE_PROFILES" > "$OUT/profiles"
echo "$IMAGES" > "$OUT/images"
echo "$SEED" > "$OUT/seed"
DATABASE="$(service_name metrics_database)"
OBSERVATION_SERVICE="$(service_name observation_service)"

echo "$RUN_ID" > "$CURRENT"
trap 'status=$?; stop_forward; [ $status -eq 0 ] || DELETE_VOLUME=1 sh deploy/kube/down.sh >/dev/null 2>&1' EXIT
kubectl apply -f "$OUT/kube/run.yaml" >/dev/null
kubectl -n eemls delete secret eemls-database --ignore-not-found >/dev/null
printf %s "$EEMLS_DATABASE_PASSWORD" \
  | kubectl -n eemls create secret generic eemls-database --from-file=password=/dev/stdin >/dev/null
kubectl apply -k "$OUT/kube" >/dev/null
kubectl -n eemls rollout status "deploy/$DATABASE" --timeout="${READY_SECONDS}s" >/dev/null
kubectl -n eemls rollout status "deploy/$OBSERVATION_SERVICE" --timeout="${READY_SECONDS}s" >/dev/null

forward "$DATABASE"
load_profiles_until "$OUT/load_profiles.log" "$(( $(date +%s) + READY_SECONDS ))" poetry run python3 tools/load_profiles.py \
  --run "$RUN_ID" --profiles "$SERVICE_PROFILES" --config "$OUT/here"
kubectl -n eemls wait --for=condition=available deploy --all --timeout="${READY_SECONDS}s" >/dev/null
in_cluster eemls-warm control-plane eemls-warm --config /opt/eemls/config --images "$WARM_IMAGES" --seed "$SEED"
echo "up: run $RUN_ID"
