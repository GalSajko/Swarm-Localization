#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

ensure_python_bin

RUNNER_NAME="social_fusion_rules_sweep"
TARGET_RESULTS_ROOT="${REPO_ROOT}/results/social_fusion/${RUNNER_NAME}"

terrains=(--city --desert --mountains --rural)
social_fusion_rules=(yager comp_boost disabled)
experiment_commands=()

for social_fusion_rule in "${social_fusion_rules[@]}"; do
    for terrain_flag in "${terrains[@]}"; do
        experiment_commands+=("$(simulation_command "${RUNNER_NAME}" "${terrain_flag}" "${DEFAULT_THRESHOLD_TYPE}" "${social_fusion_rule}" "${DEFAULT_PERSONAL_FUSION}")")
    done
done

launch_parallel_commands "${RUNNER_NAME}" "${TARGET_RESULTS_ROOT}" "${experiment_commands[@]}"
