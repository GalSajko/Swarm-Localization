#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

ensure_python_bin

RUNNER_NAME="incompatibility_penalty_sweep"
TARGET_RESULTS_ROOT="${REPO_ROOT}/results/social_fusion/${RUNNER_NAME}"

incompatibility_penalty_values=(0 0.25 0.5 0.75 1.0)
experiment_commands=()

for incompatibility_penalty in "${incompatibility_penalty_values[@]}"; do
    experiment_commands+=("$(simulation_command "${RUNNER_NAME}" "--desert" "${DEFAULT_THRESHOLD_TYPE}" "${DEFAULT_SOCIAL_FUSION}" "${DEFAULT_PERSONAL_FUSION}" "${DEFAULT_MASS_FUNCTION}" "${DEFAULT_TRIALS}" "${DEFAULT_SWARM_SIZE}" "${DEFAULT_DECISION_THRESHOLD}" "${DEFAULT_DS_TOP_K}" "${DEFAULT_DS_MAX_NEIGHBORS}" "${DEFAULT_DS_SINGLETON_WEIGHT}" "${DEFAULT_DS_NEIGHBOR_COMPATIBILITY_GAIN}" "${DEFAULT_DS_NEIGHBOR_SAME_TILE_SUPPORT}" "${DEFAULT_DS_NEIGHBOR_ADJACENT_TILE_SUPPORT}" "${incompatibility_penalty}")")
done

launch_parallel_commands "${RUNNER_NAME}" "${TARGET_RESULTS_ROOT}" "${experiment_commands[@]}"
