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
SINGLETON_WEIGHT="${SINGLETON_WEIGHT:-0.5}"
COMPATIBILITY_GAIN="${COMPATIBILITY_GAIN:-0.25}"
LOG_DIR="results/continuous_est/_logs"
TERRAINS=(city desert mountains rural)
TOPK_VALUES=(1 2 4 6 8 10)

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

run_topk_batch_pair() {
  local first_k="$1"
  local second_k="${2:-}"
  local pids=()
  local terrain

  echo "Starting top-k batch: ${first_k}${second_k:+ and ${second_k}}"

  for top_k in "$first_k" ${second_k:+"$second_k"}; do
    local experiment_name="topk_${top_k}"

    if is_experiment_complete "$experiment_name"; then
      echo "Skipping top-k experiment: k=${top_k}"
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
        "--top_k=${top_k}" \
        "--singleton_weight=${SINGLETON_WEIGHT}" \
        "--compatibility_gain=${COMPATIBILITY_GAIN}" > "${log_file}" 2>&1 &
      pids+=("$!")
    done
  done

  local pid
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  echo "Finished top-k batch: ${first_k}${second_k:+ and ${second_k}}"
}

index=0
while [[ $index -lt ${#TOPK_VALUES[@]} ]]; do
  first_k="${TOPK_VALUES[$index]}"
  second_k=""
  if [[ $((index + 1)) -lt ${#TOPK_VALUES[@]} ]]; then
    second_k="${TOPK_VALUES[$((index + 1))]}"
  fi
  run_topk_batch_pair "$first_k" "$second_k"
  index=$((index + 2))
done
