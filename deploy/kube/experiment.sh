#!/bin/sh
# Run each approach and seed of an experiment plan on the cluster.
#
#   EEMLS_DATABASE_PASSWORD=... PLAN=experiments/runs/recurring_ten_days.yaml \
#     APPROACHES=cap,cap_star SEEDS=0 DURATION_S=1800 sh deploy/kube/experiment.sh
set -eu
. deploy/common.sh
need_password
experiment_folders kube
poetry run eemls-config render config --overrides "${OVERRIDES:-deploy/kube/overrides.yaml}" --out "$OUT/base" >/dev/null
experiment_plan "$OUT/base"

trap 'sh deploy/kube/down.sh >/dev/null 2>&1 || true' EXIT
while read -r approach seed run_id folder planned_s scenario <&3; do
  run="$OUT/$folder"
  echo "run $run_id: approach $approach, seed $seed"
  experiment_render "$approach" "$OUT/base" "$run"
  rm -rf "runs/kube/$run_id"
  CONFIG_DIR="$run/config" RUN_ID="$run_id" SEED="$seed" sh deploy/kube/up.sh
  SCENARIOS="$SCENARIOS" SCENARIO="$scenario" DURATION_S="${DURATION_S:-$planned_s}" sh deploy/kube/run.sh
  mv "runs/kube/$run_id/rows" "$run/rows"
  DELETE_VOLUME=1 sh deploy/kube/down.sh
  rm -rf "runs/kube/$run_id"
done 3< "$OUT/plan.txt"
echo "experiment $NAME: rows under $OUT"
