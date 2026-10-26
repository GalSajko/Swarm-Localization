#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

ensure_python_bin

RUNNER_NAME="mass_function_mode_sweep"
TARGET_RESULTS_ROOT="${REPO_ROOT}/results/personal_fusion/${RUNNER_NAME}"

terrains=(--city --desert --mountains --rural)
mass_function_modes=(spatial_topk topk)
experiment_commands=()

for mass_function_mode in "${mass_function_modes[@]}"; do
    for terrain_flag in "${terrains[@]}"; do
        experiment_commands+=("$(simulation_command "${RUNNER_NAME}" "${terrain_flag}" "${DEFAULT_THRESHOLD_TYPE}" "${DEFAULT_SOCIAL_FUSION}" "${DEFAULT_PERSONAL_FUSION}" "${mass_function_mode}")")
    done
done

launch_parallel_commands "${RUNNER_NAME}" "${TARGET_RESULTS_ROOT}" "${experiment_commands[@]}"
