#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

ensure_python_bin

RUNNER_NAME="topk_neighbors_sweep"
TARGET_RESULTS_ROOT="${REPO_ROOT}/results/personal_fusion/${RUNNER_NAME}"

personal_fusion_rules=(dempster yager adaptive hybrid)
top_k_values=(1 2 4 8)
max_neighbor_values=(0 1 2 4 8)
experiment_commands=()

for personal_fusion_rule in "${personal_fusion_rules[@]}"; do
    for ds_top_k in "${top_k_values[@]}"; do
        for ds_max_neighbors in "${max_neighbor_values[@]}"; do
            experiment_commands+=("$(simulation_command "${RUNNER_NAME}" "--desert" "${DEFAULT_THRESHOLD_TYPE}" "${DEFAULT_SOCIAL_FUSION}" "${personal_fusion_rule}" "${DEFAULT_MASS_FUNCTION}" "${DEFAULT_TRIALS}" "${DEFAULT_SWARM_SIZE}" "${DEFAULT_DECISION_THRESHOLD}" "${ds_top_k}" "${ds_max_neighbors}")")
        done
    done
done

launch_parallel_commands "${RUNNER_NAME}" "${TARGET_RESULTS_ROOT}" "${experiment_commands[@]}"
