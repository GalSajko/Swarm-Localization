#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

ensure_python_bin

RUNNER_NAME="singleton_weight_sweep"
TARGET_RESULTS_ROOT="${REPO_ROOT}/results/personal_fusion/${RUNNER_NAME}"

personal_fusion_rules=(dempster yager adaptive hybrid)
singleton_weight_values=(0 0.25 0.5 0.75 1.0)
experiment_commands=()

for personal_fusion_rule in "${personal_fusion_rules[@]}"; do
    for ds_singleton_weight in "${singleton_weight_values[@]}"; do
        experiment_commands+=("$(simulation_command "${RUNNER_NAME}" "--desert" "${DEFAULT_THRESHOLD_TYPE}" "${DEFAULT_SOCIAL_FUSION}" "${personal_fusion_rule}" "${DEFAULT_MASS_FUNCTION}" "${DEFAULT_TRIALS}" "${DEFAULT_SWARM_SIZE}" "${DEFAULT_DECISION_THRESHOLD}" "${DEFAULT_DS_TOP_K}" "${DEFAULT_DS_MAX_NEIGHBORS}" "${ds_singleton_weight}")")
    done
done

launch_parallel_commands "${RUNNER_NAME}" "${TARGET_RESULTS_ROOT}" "${experiment_commands[@]}"
