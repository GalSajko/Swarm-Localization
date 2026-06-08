#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_experiment_set_common.sh"

SWARM_SIZES=(2 4 9 12 16 20 25)
SOCIAL_FUSION_VALUES=(comp_boost yager)

run_swarm_size_condition() {
  local swarm_size="$1"
  local social_fusion="$2"
  local experiment_name="swarm_size_${swarm_size}_${social_fusion}"
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
      "--num_agents=${swarm_size}" \
      "--stage_one=${STAGE_ONE_DEFAULT}" \
      "--social_fusion=${social_fusion}" \
      "--threshold_value=${THRESHOLD_VALUE_DEFAULT}" \
      "--top_k=${TOP_K_DEFAULT}" \
      "--singleton_weight=${SINGLETON_WEIGHT_DEFAULT}" \
      "--compatibility_gain=${COMPATIBILITY_GAIN_DEFAULT}" \
      "--incompatibility_penalty=${INCOMPATIBILITY_PENALTY_DEFAULT}" \
      "--neighbor_adjacent_tile_support=${NEIGHBOR_ADJACENT_TILE_SUPPORT_DEFAULT}" \
      "--max_experiment_time=${MAX_EXPERIMENT_TIME_DEFAULT}"
  done
}

for swarm_size in "${SWARM_SIZES[@]}"; do
  for social_fusion in "${SOCIAL_FUSION_VALUES[@]}"; do
    run_swarm_size_condition "$swarm_size" "$social_fusion"
  done
done
