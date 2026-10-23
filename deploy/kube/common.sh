# Helpers shared by the kube scripts.
#
#   . deploy/kube/common.sh
. deploy/common.sh

DATABASE_FORWARD_PORT=15432
READY_SECONDS="${READY_SECONDS:-600}"
CURRENT=runs/kube/current

current_run() {
  [ -f "$CURRENT" ] || { echo "no run is up; run deploy/kube/up.sh first"; exit 1; }
  RUN_ID="$(cat "$CURRENT")"
  OUT="runs/kube/$RUN_ID"
}

# The id itself, or <n>-<id> with the first n whose folder is free.
free_run_id() {
  [ -e "runs/kube/$1" ] || { echo "$1"; return; }
  n=1
  while [ -e "runs/kube/$n-$1" ]; do n=$((n + 1)); done
  echo "$n-$1"
}

# The host name of service $1 in the run's tree.
service_name() {
  poetry run eemls-deploy name --config "$OUT/config" "$1"
}

# Delete the deployment $1, if named, and wait until its pods have stopped.
stop_deployment() {
  [ -z "$1" ] || kubectl -n eemls delete deploy "$1" --ignore-not-found --cascade=foreground \
    --wait=true --timeout="${READY_SECONDS}s" >/dev/null
}

# Delete the run's inference services and wait until their pods have stopped.
stop_inference_services() {
  [ ! -d "$OUT/kube/inference" ] || kubectl -n eemls delete -f "$OUT/kube/inference" --ignore-not-found \
    --cascade=foreground --wait=true --timeout="${READY_SECONDS}s" >/dev/null
}

# Forward the run's database service $1 to 127.0.0.1:$DATABASE_FORWARD_PORT.
forwarded=""
forward() {
  kubectl -n eemls port-forward "svc/$1" "$DATABASE_FORWARD_PORT:5432" >"$OUT/port-forward.log" 2>&1 &
  forwarded=$!
  sleep 2
  kill -0 "$forwarded" 2>/dev/null || { echo "the database forward stopped:"; tail -2 "$OUT/port-forward.log"; exit 1; }
}

stop_forward() {
  [ -z "$forwarded" ] || kill "$forwarded" 2>/dev/null || true
}

# Run one command in a pod on a core node; print the log tail.
in_cluster() {
  name="$1"
  image="$2"
  shift 2
  poetry run -- eemls-deploy pod --config "$OUT/config" --name "$name" --image "$image" \
    --out "$OUT/$name.yaml" -- "$@" >/dev/null || return 1
  kubectl -n eemls delete pod "$name" --ignore-not-found >/dev/null
  kubectl -n eemls apply -f "$OUT/$name.yaml" >/dev/null
  deadline=$(( $(date +%s) + READY_SECONDS ))
  until phase="$(kubectl -n eemls get pod "$name" -o jsonpath='{.status.phase}')" \
      && { [ "$phase" = Succeeded ] || [ "$phase" = Failed ]; }; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "$name never finished:"; kubectl -n eemls describe pod "$name" | tail -10
      kubectl -n eemls delete pod "$name" --ignore-not-found >/dev/null
      rm -f "$OUT/$name.yaml"
      return 1
    fi
    sleep 2
  done
  kubectl -n eemls logs "$name" | tail -"${TAIL:-3}"
  kubectl -n eemls delete pod "$name" >/dev/null
  rm -f "$OUT/$name.yaml"
  [ "$phase" = Succeeded ]
}
