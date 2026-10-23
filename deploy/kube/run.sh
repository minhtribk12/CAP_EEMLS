#!/bin/sh
# Run one consumer service per consumer and collect the rows.
#
#   SCENARIO=recurring_ten_days DURATION_S=18000 sh deploy/kube/run.sh
#   SCENARIOS: the folder of scenario files; default experiments/scenarios
set -eu
. deploy/kube/common.sh
need_password
current_run
trap stop_forward EXIT
run_defaults
SEED="$(cat "$OUT/seed")"
SERVICE_PROFILES="$(cat "$OUT/profiles")"
IMAGES="$(cat "$OUT/images")"

poetry run eemls-deploy kube --config "$OUT/config" --mount-config "$OUT/config" \
  --profiles "$SERVICE_PROFILES" --images "$IMAGES" --run-id "$RUN_ID" --out "$OUT/kube" \
  --scenario "$SCENARIO" --scenarios "$SCENARIOS" --duration-s "$DURATION_S" --seed "$SEED" >/dev/null
kubectl -n eemls delete job --all >/dev/null
kubectl apply -k "$OUT/kube" >/dev/null
kubectl -n eemls wait --for=condition=complete job --all --timeout="$(( ${DURATION_S%.*} + 300 ))s" >/dev/null
for job in $(kubectl -n eemls get jobs -o name); do
  kubectl -n eemls logs "$job" | grep 'queries accepted' || true
done

# Stop the control plane and the inference services, then the Observation Service.
stop_deployment "$(service_name control_plane)"
stop_inference_services
stop_deployment "$(service_name observation_service)"
DATABASE="$(service_name metrics_database)"
forward "$DATABASE"
poetry run eemls-collect --config "$OUT/here" --run-id "$RUN_ID" --out "$OUT/rows"
echo "run $RUN_ID: rows under $OUT/rows"
