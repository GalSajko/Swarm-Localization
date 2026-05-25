#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_CMD="${PYTHON_CMD:-$ROOT_DIR/swarm_loc_venv/bin/python}"
TRIALS_PER_TERRAIN="${TRIALS_PER_TERRAIN:-100}"
NUM_AGENTS="${NUM_AGENTS:-12}"
THRESHOLD_MIN="${THRESHOLD_MIN:-0.1}"
THRESHOLD_MAX="${THRESHOLD_MAX:-1.0}"
TOP_K="${TOP_K:-4}"
SINGLETON_WEIGHT="${SINGLETON_WEIGHT:-0.5}"
COMPATIBILITY_GAIN="${COMPATIBILITY_GAIN:-0.25}"
LOG_DIR="results/continuous_est/_logs"
TERRAINS=(city desert mountains rural)
THRESHOLD_VALUES=(0.5 0.6 0.7 0.8 0.9 1.0)

mkdir -p "$LOG_DIR"

format_threshold_suffix() {
  printf '%s' "$1" | tr '.' '_'
}

is_experiment_complete() {
  local experiment_name="$1"
  local summary_file="results/continuous_est/decision_threshold/${experiment_name}/trial_summary.csv"
  if [[ ! -f "$summary_file" ]]; then
    return 1
  fi

  local line_count
  line_count="$(tail -n +2 "$summary_file" | wc -l | tr -d ' ')"
  [[ "$line_count" -ge $((4 * TRIALS_PER_TERRAIN)) ]]
}

run_dynamic_batch() {
  local threshold_value="$1"
  local suffix
  suffix="$(format_threshold_suffix "$threshold_value")"
  local dynamic_experiment="decision_threshold_dynamic_${suffix}"

  if is_experiment_complete "$dynamic_experiment"; then
    echo "Skipping dynamic-threshold batch: threshold=${threshold_value}"
    return 0
  fi

  echo "Starting dynamic-threshold batch: threshold=${threshold_value}"
  local pids=()
  local terrain
  for terrain in "${TERRAINS[@]}"; do
    local dynamic_log="${LOG_DIR}/${dynamic_experiment}_${terrain}.log"

    "${PYTHON_CMD}" src/simulation.py "${dynamic_experiment}" "--${terrain}" \
      "--trials=${TRIALS_PER_TERRAIN}" \
      "--num_agents=${NUM_AGENTS}" \
      "--threshold_value=${threshold_value}" \
      "--threshold_min=${THRESHOLD_MIN}" \
      "--threshold_max=${THRESHOLD_MAX}" \
      "--top_k=${TOP_K}" \
      "--singleton_weight=${SINGLETON_WEIGHT}" \
      "--compatibility_gain=${COMPATIBILITY_GAIN}" > "${dynamic_log}" 2>&1 &
    pids+=("$!")
  done

  local pid
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  echo "Finished dynamic-threshold batch: threshold=${threshold_value}"
}

for threshold_value in "${THRESHOLD_VALUES[@]}"; do
  run_dynamic_batch "${threshold_value}"
done
