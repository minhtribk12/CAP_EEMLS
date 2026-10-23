# Helpers shared by the one-host and kube scripts.
#
#   . deploy/common.sh

need_password() {
  : "${EEMLS_DATABASE_PASSWORD:?is the metrics database password, and is never written to a file}"
}

# A run id names the run's database volume.
check_run_id() {
  case "$1" in
    "" | -* | *- | *[!a-z0-9-]*) echo "run id $1 must be lowercase letters, digits and hyphens, with no hyphen at either end"; exit 1 ;;
  esac
}

# Copy CONFIG_DIR to $1, or render the committed tree with the overrides $2 into it.
config_into() {
  if [ -n "${CONFIG_DIR:-}" ]; then
    cp -r "$CONFIG_DIR" "$1"
  else
    poetry run eemls-config render config --overrides "$2" --out "$1"
  fi
}

# Run the profile load ($3...) until the database takes it, by the time $2; its output goes to $1.
load_profiles_until() {
  log="$1"
  until_s="$2"
  shift 2
  until "$@" >"$log" 2>&1; do
    if [ "$(date +%s)" -ge "$until_s" ]; then
      echo "the profiles never went into the metrics database:"; tail -5 "$log"; exit 1
    fi
    sleep 2
  done
}

load_defaults() {
  RATES="${RATES:-10,20,50,100}"
  STEP_S="${STEP_S:-30}"
  CONSUMER="${CONSUMER:-c3_surveillance}"
}

run_defaults() {
  SCENARIO="${SCENARIO:-recurring_ten_days}"
  DURATION_S="${DURATION_S:-18000}"
  SCENARIOS="${SCENARIOS:-experiments/scenarios}"
}

# NAME, SCENARIOS and a full OUT for PLAN; OUT defaults to runs/$1_<time>_<name>.
experiment_folders() {
  : "${PLAN:?is the experiment plan to run, one of experiments/runs/}"
  NAME="$(basename "$PLAN" .yaml)"
  SCENARIOS="${SCENARIOS:-$(dirname "$(dirname "$PLAN")")/scenarios}"
  OUT="${OUT:-runs/$1_$(date +%Y%m%d_%H%M%S)_$NAME}"
  mkdir -p "$OUT"
  OUT="$(cd "$OUT" && pwd)"
}

# Plan PLAN against the tree $1 into $OUT/plan.txt.
experiment_plan() {
  poetry run eemls-exp plan "$PLAN" --config "$1" --scenarios "$SCENARIOS" \
    ${APPROACHES:+--approaches "$APPROACHES"} ${SEEDS:+--seeds "$SEEDS"} > "$OUT/plan.txt"
}

# Render approach $1 of PLAN from the tree $2 into $3.
experiment_render() {
  poetry run eemls-exp render "$PLAN" --approach "$1" --config "$2" \
    --scenarios "$SCENARIOS" --out "$3" >/dev/null
}
