#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_CMD="${PYTHON_CMD:-$ROOT_DIR/swarm_loc_venv/bin/python}"
TRIALS_PER_TERRAIN="${TRIALS_PER_TERRAIN:-100}"
NUM_AGENTS="${NUM_AGENTS:-12}"
THRESHOLD_VALUE="${THRESHOLD_VALUE:-0.5}"
THRESHOLD_MIN="${THRESHOLD_MIN:-0.1}"
THRESHOLD_MAX="${THRESHOLD_MAX:-1.0}"
TOP_K="${TOP_K:-4}"
COMPATIBILITY_GAIN="${COMPATIBILITY_GAIN:-0.25}"
LOG_DIR="results/continuous_est/_logs"
TERRAINS=(city desert mountains rural)
SINGLETON_WEIGHT_VALUES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

mkdir -p "$LOG_DIR"

format_weight_suffix() {
  printf '%s' "$1" | tr '.' '_'
}

is_experiment_complete() {
  local experiment_name="$1"
  local summary_file="results/continuous_est/top_k_ds/${experiment_name}/trial_summary.csv"
  if [[ ! -f "$summary_file" ]]; then
    return 1
  fi

  local line_count
  line_count="$(tail -n +2 "$summary_file" | wc -l | tr -d ' ')"
  [[ "$line_count" -ge $((4 * TRIALS_PER_TERRAIN)) ]]
}

run_singleton_weight_batch_pair() {
  local first_weight="$1"
  local second_weight="${2:-}"
  local pids=()
  local terrain

  echo "Starting singleton-weight batch: ${first_weight}${second_weight:+ and ${second_weight}}"

  for singleton_weight in "$first_weight" ${second_weight:+"$second_weight"}; do
    local suffix
    suffix="$(format_weight_suffix "$singleton_weight")"
    local experiment_name="singleton_weight_${suffix}"

    if is_experiment_complete "$experiment_name"; then
      echo "Skipping singleton-weight experiment: weight=${singleton_weight}"
      continue
    fi

    for terrain in "${TERRAINS[@]}"; do
      local log_file="${LOG_DIR}/${experiment_name}_${terrain}.log"
      "${PYTHON_CMD}" src/simulation.py "${experiment_name}" "--${terrain}" \
        "--trials=${TRIALS_PER_TERRAIN}" \
        "--num_agents=${NUM_AGENTS}" \
        "--threshold_value=${THRESHOLD_VALUE}" \
        "--threshold_min=${THRESHOLD_MIN}" \
        "--threshold_max=${THRESHOLD_MAX}" \
        "--top_k=${TOP_K}" \
        "--singleton_weight=${singleton_weight}" \
        "--compatibility_gain=${COMPATIBILITY_GAIN}" > "${log_file}" 2>&1 &
      pids+=("$!")
    done
  done

  local pid
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  echo "Finished singleton-weight batch: ${first_weight}${second_weight:+ and ${second_weight}}"
}

index=0
while [[ $index -lt ${#SINGLETON_WEIGHT_VALUES[@]} ]]; do
  first_weight="${SINGLETON_WEIGHT_VALUES[$index]}"
  second_weight=""
  if [[ $((index + 1)) -lt ${#SINGLETON_WEIGHT_VALUES[@]} ]]; then
    second_weight="${SINGLETON_WEIGHT_VALUES[$((index + 1))]}"
  fi
  run_singleton_weight_batch_pair "$first_weight" "$second_weight"
  index=$((index + 2))
done
