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
COMPATIBILITY_GAIN="${COMPATIBILITY_GAIN:-1.0}"
LOG_DIR="results/continuous_est/_logs"
TERRAINS=(city desert mountains rural)
EXPERIMENT_NAME="${EXPERIMENT_NAME:-laplacian_consensus_baseline}"

mkdir -p "$LOG_DIR"
mkdir -p "results/continuous_est/laplacian"

is_experiment_complete() {
  local experiment_name="$1"
  local summary_file="results/continuous_est/laplacian/${experiment_name}/trial_summary.csv"
  if [[ ! -f "$summary_file" ]]; then
    return 1
  fi

  local line_count
  line_count="$(tail -n +2 "$summary_file" | wc -l | tr -d ' ')"
  [[ "$line_count" -ge $((4 * TRIALS_PER_TERRAIN)) ]]
}

if is_experiment_complete "$EXPERIMENT_NAME"; then
  echo "Skipping ${EXPERIMENT_NAME}: trial_summary.csv already has full data."
  exit 0
fi

echo "Starting ${EXPERIMENT_NAME}"
pids=()
for terrain in "${TERRAINS[@]}"; do
  log_file="${LOG_DIR}/${EXPERIMENT_NAME}_${terrain}.log"
  "${PYTHON_CMD}" src/simulation.py "${EXPERIMENT_NAME}" "--${terrain}" \
    "--trials=${TRIALS_PER_TERRAIN}" \
    "--num_agents=${NUM_AGENTS}" \
    "--threshold_value=${THRESHOLD_VALUE}" \
    "--threshold_min=${THRESHOLD_MIN}" \
    "--threshold_max=${THRESHOLD_MAX}" \
    "--top_k=${TOP_K}" \
    "--singleton_weight=${SINGLETON_WEIGHT}" \
    "--compatibility_gain=${COMPATIBILITY_GAIN}" \
    --laplacian > "${log_file}" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "Finished ${EXPERIMENT_NAME}"
