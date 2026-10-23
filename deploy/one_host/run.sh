#!/bin/sh
# Run one consumer service per consumer and wait for them.
#
#   SCENARIO=recurring_ten_days DURATION_S=18000 make run
#   SCENARIOS: the folder of scenario files; default experiments/scenarios
set -eu
. deploy/one_host/common.sh
need_password
run_defaults
case "$SCENARIOS" in /*) ;; *) SCENARIOS="$PWD/$SCENARIOS" ;; esac
OUT=runs/one_host
[ -f "$OUT/run_id" ] || { echo "no system is up; run make up first"; exit 1; }
RUN_ID="$(cat "$OUT/run_id")"
SERVICE_PROFILES="$(cat "$OUT/profiles")"
SEED="$(cat "$OUT/seed")"

stop_consumers
swarm_stack --scenario "$SCENARIO" --scenarios "$SCENARIOS" --duration-s "$DURATION_S"
docker stack deploy --detach=true --resolve-image never -c "$OUT/stack.yaml" "$STACK" >/dev/null

deadline=$(( $(date +%s) + ${DURATION_S%.*} + 300 ))
for service in $(docker service ls --filter "name=${STACK}_consumer-" --format '{{.Name}}'); do
  until docker service ps "$service" --format '{{.CurrentState}}' | grep -q '^Complete'; do
    if docker service ps "$service" --format '{{.CurrentState}}' | grep -q -E '^(Failed|Rejected)'; then
      echo "$service did not finish:"; docker service logs --tail 20 "$service"; exit 1
    fi
    [ "$(date +%s)" -lt "$deadline" ] || { echo "$service is still running"; exit 1; }
    sleep 5
  done
  docker service logs "$service" 2>/dev/null | grep 'queries accepted' || true
done
