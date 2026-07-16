#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

ensure_python_bin

RUNNER_NAME="personal_fusion_rules_sweep"
TARGET_RESULTS_ROOT="${REPO_ROOT}/results/personal_fusion/${RUNNER_NAME}"

terrains=(--city --desert --mountains --rural)
personal_fusion_rules=(dempster yager adaptive hybrid)
experiment_commands=()

for personal_fusion_rule in "${personal_fusion_rules[@]}"; do
    for terrain_flag in "${terrains[@]}"; do
        experiment_commands+=("$(simulation_command "${RUNNER_NAME}" "${terrain_flag}" "${DEFAULT_THRESHOLD_TYPE}" "${DEFAULT_SOCIAL_FUSION}" "${personal_fusion_rule}")")
    done
done

launch_parallel_commands "${RUNNER_NAME}" "${TARGET_RESULTS_ROOT}" "${experiment_commands[@]}"
