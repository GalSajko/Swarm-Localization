#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
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

mkdir -p "$LOG_DIR"

run_social_fusion_batch() {
  local boost_experiment="social_fusion_compatibility_boost"
  local yager_experiment="social_fusion_yager"
  local pids=()
  local terrain

  echo "Starting social-fusion comparison batch"
  for terrain in "${TERRAINS[@]}"; do
    local boost_log="${LOG_DIR}/${boost_experiment}_${terrain}.log"
    local yager_log="${LOG_DIR}/${yager_experiment}_${terrain}.log"

    "${PYTHON_CMD}" src/simulation.py "${boost_experiment}" "--${terrain}" \
      "--trials=${TRIALS_PER_TERRAIN}" \
      "--num_agents=${NUM_AGENTS}" \
      "--threshold_value=${THRESHOLD_VALUE}" \
      "--threshold_min=${THRESHOLD_MIN}" \
      "--threshold_max=${THRESHOLD_MAX}" \
      "--top_k=${TOP_K}" \
      "--singleton_weight=${SINGLETON_WEIGHT}" \
      "--compatibility_gain=${COMPATIBILITY_GAIN}" > "${boost_log}" 2>&1 &
    pids+=("$!")

    "${PYTHON_CMD}" src/simulation.py "${yager_experiment}" "--${terrain}" \
      "--trials=${TRIALS_PER_TERRAIN}" \
      "--num_agents=${NUM_AGENTS}" \
      "--threshold_value=${THRESHOLD_VALUE}" \
      "--threshold_min=${THRESHOLD_MIN}" \
      "--threshold_max=${THRESHOLD_MAX}" \
      "--top_k=${TOP_K}" \
      "--singleton_weight=${SINGLETON_WEIGHT}" \
      "--compatibility_gain=${COMPATIBILITY_GAIN}" \
      --yager > "${yager_log}" 2>&1 &
    pids+=("$!")
  done

  local pid
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  echo "Finished social-fusion comparison batch"
}

run_social_fusion_batch
