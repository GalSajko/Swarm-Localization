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
SINGLETON_WEIGHT="${SINGLETON_WEIGHT:-0.5}"
COMPATIBILITY_GAIN="${COMPATIBILITY_GAIN:-0.25}"
LOG_DIR="results/continuous_est/_logs"
TERRAINS=(city desert mountains rural)
COMMUNICATION_RANGES=(50 100 150)

mkdir -p "$LOG_DIR"

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

run_communication_range_batch_pair() {
  local first_range="$1"
  local second_range="${2:-}"
  local pids=()
  local terrain

  echo "Starting low communication-range batch: ${first_range}${second_range:+ and ${second_range}}"

  for communication_range in "$first_range" ${second_range:+"$second_range"}; do
    local experiment_name="communication_range_${communication_range}"

    if is_experiment_complete "$experiment_name"; then
      echo "Skipping communication-range experiment: range=${communication_range}"
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
        "--singleton_weight=${SINGLETON_WEIGHT}" \
        "--compatibility_gain=${COMPATIBILITY_GAIN}" \
        "--communication_range=${communication_range}" > "${log_file}" 2>&1 &
      pids+=("$!")
    done
  done

  local pid
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  echo "Finished low communication-range batch: ${first_range}${second_range:+ and ${second_range}}"
}

index=0
while [[ $index -lt ${#COMMUNICATION_RANGES[@]} ]]; do
  first_range="${COMMUNICATION_RANGES[$index]}"
  second_range=""
  if [[ $((index + 1)) -lt ${#COMMUNICATION_RANGES[@]} ]]; then
    second_range="${COMMUNICATION_RANGES[$((index + 1))]}"
  fi
  run_communication_range_batch_pair "$first_range" "$second_range"
  index=$((index + 2))
done
