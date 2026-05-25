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
LOG_DIR="results/continuous_est/_logs"
TERRAINS=(city desert mountains rural)
GAIN_VALUES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

mkdir -p "$LOG_DIR"

format_gain_suffix() {
  printf '%s' "$1" | tr '.' '_'
}

is_experiment_complete() {
  local experiment_name="$1"
  local summary_file="results/continuous_est/social_fusion/${experiment_name}/trial_summary.csv"
  if [[ ! -f "$summary_file" ]]; then
    return 1
  fi

  local line_count
  line_count="$(tail -n +2 "$summary_file" | wc -l | tr -d ' ')"
  [[ "$line_count" -ge $((4 * TRIALS_PER_TERRAIN)) ]]
}

run_gain_batch_pair() {
  local first_gain="$1"
  local second_gain="${2:-}"
  local pids=()
  local terrain

  echo "Starting compatibility-gain batch: ${first_gain}${second_gain:+ and ${second_gain}}"

  for gain in "$first_gain" ${second_gain:+"$second_gain"}; do
    local suffix
    suffix="$(format_gain_suffix "$gain")"
    local experiment_name="compatibility_gain_${suffix}"

    if is_experiment_complete "$experiment_name"; then
      echo "Skipping compatibility-gain experiment: gain=${gain}"
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
        "--compatibility_gain=${gain}" > "${log_file}" 2>&1 &
      pids+=("$!")
    done
  done

  local pid
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  echo "Finished compatibility-gain batch: ${first_gain}${second_gain:+ and ${second_gain}}"
}

index=0
while [[ $index -lt ${#GAIN_VALUES[@]} ]]; do
  first_gain="${GAIN_VALUES[$index]}"
  second_gain=""
  if [[ $((index + 1)) -lt ${#GAIN_VALUES[@]} ]]; then
    second_gain="${GAIN_VALUES[$((index + 1))]}"
  fi
  run_gain_batch_pair "$first_gain" "$second_gain"
  index=$((index + 2))
done
