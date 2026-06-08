#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_experiment_set_common.sh"

INCOMPATIBILITY_PENALTY_VALUES=(0.0 0.25 0.5 0.75 1.0)

run_incompatibility_penalty_condition() {
  local incompatibility_penalty="$1"
  local suffix
  suffix="$(format_suffix "$incompatibility_penalty")"
  local experiment_name="incompatibility_penalty_${suffix}"
  local terrain

  if is_experiment_complete "$experiment_name"; then
    echo "Skipping ${experiment_name}: full data already present."
    return 0
  fi

  for terrain in "${TERRAINS[@]}"; do
    run_four_parallel_trial_workers_for_terrain \
      "$experiment_name" \
      "$terrain" \
      "$PYTHON_CMD" src/simulation.py "$experiment_name" \
      "--num_agents=${NUMBER_OF_AGENTS_DEFAULT}" \
      "--stage_one=${STAGE_ONE_DEFAULT}" \
      "--social_fusion=comp_boost" \
      "--threshold_value=${THRESHOLD_VALUE_DEFAULT}" \
      "--top_k=${TOP_K_DEFAULT}" \
      "--singleton_weight=${SINGLETON_WEIGHT_DEFAULT}" \
      "--compatibility_gain=${COMPATIBILITY_GAIN_DEFAULT}" \
      "--incompatibility_penalty=${incompatibility_penalty}" \
      "--neighbor_adjacent_tile_support=${NEIGHBOR_ADJACENT_TILE_SUPPORT_DEFAULT}" \
      "--max_experiment_time=${MAX_EXPERIMENT_TIME_DEFAULT}"
  done
}

for incompatibility_penalty in "${INCOMPATIBILITY_PENALTY_VALUES[@]}"; do
  run_incompatibility_penalty_condition "$incompatibility_penalty"
done
