#!/bin/sh
# Run every approach and seed of one experiment plan on this host.
#
#   EEMLS_DATABASE_PASSWORD=... PLAN=experiments/runs/recurring_ten_days.yaml \
#     APPROACHES=cap,cap_star SEEDS=0 DURATION_S=1800 make experiment
set -eu
. deploy/one_host/common.sh
need_password
experiment_folders local

poetry run eemls-config render config --overrides "${OVERRIDES:-deploy/one_host/overrides.yaml}" --out "$OUT/base"
experiment_plan "$OUT/base"

# Stop the system and remove the run's volume.
remove_run() {
  sh deploy/one_host/down.sh >/dev/null
  docker volume rm "$(volume_of "$1")" >/dev/null 2>&1 || true
}
trap 'remove_run "${run_id:-}"' EXIT

while read -r approach seed run_id folder planned_s scenario <&3; do
  run="$OUT/$folder"
  echo "run $run_id: approach $approach, seed $seed"
  remove_run "$run_id"
  experiment_render "$approach" "$OUT/base" "$run"
  CONFIG_DIR="$run/config" RUN_ID="$run_id" SEED="$seed" sh deploy/one_host/up.sh
  SCENARIOS="$SCENARIOS" SCENARIO="$scenario" DURATION_S="${DURATION_S:-$planned_s}" sh deploy/one_host/run.sh
  stop_service "$(service_name control_plane "$run/config")"
  stop_inference_services "$run/config"
  stop_service "$(service_name observation_service "$run/config")"
  docker run --rm --network "${STACK}_eemls" --user "$(id -u):$(id -g)" -e EEMLS_DATABASE_PASSWORD \
    -v "$run/config:/opt/eemls/config:ro" -v "$run:/opt/eemls/out" \
    --entrypoint eemls-collect "$REGISTRY/eemls-control-plane:latest" \
    --config /opt/eemls/config --run-id "$run_id" --out /opt/eemls/out/rows
  remove_run "$run_id"
done 3< "$OUT/plan.txt"
echo "experiment $NAME: rows under $OUT"
