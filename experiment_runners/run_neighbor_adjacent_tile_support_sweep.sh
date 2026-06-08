#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_experiment_set_common.sh"

NEIGHBOR_ADJACENT_TILE_SUPPORT_VALUES=(0.0 0.25 0.5 0.75 1.0)

run_neighbor_adjacent_tile_support_condition() {
  local neighbor_adjacent_tile_support="$1"
  local suffix
  suffix="$(format_suffix "$neighbor_adjacent_tile_support")"
  local experiment_name="neighbor_adjacent_tile_support_${suffix}"
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
      "--incompatibility_penalty=${INCOMPATIBILITY_PENALTY_DEFAULT}" \
      "--neighbor_adjacent_tile_support=${neighbor_adjacent_tile_support}" \
      "--max_experiment_time=${MAX_EXPERIMENT_TIME_DEFAULT}"
  done
}

for neighbor_adjacent_tile_support in "${NEIGHBOR_ADJACENT_TILE_SUPPORT_VALUES[@]}"; do
  run_neighbor_adjacent_tile_support_condition "$neighbor_adjacent_tile_support"
done
