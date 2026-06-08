#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_experiment_set_common.sh"

SINGLETON_WEIGHT_VALUES=(0.0 0.25 0.5 0.75 1.0)
SOCIAL_FUSION_VALUES=(comp_boost yager)

run_singleton_weight_condition() {
  local singleton_weight="$1"
  local social_fusion="$2"
  local suffix
  suffix="$(format_suffix "$singleton_weight")"
  local experiment_name="singleton_weight_${suffix}_${social_fusion}"
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
      "--social_fusion=${social_fusion}" \
      "--threshold_value=${THRESHOLD_VALUE_DEFAULT}" \
      "--top_k=${TOP_K_DEFAULT}" \
      "--singleton_weight=${singleton_weight}" \
      "--compatibility_gain=${COMPATIBILITY_GAIN_DEFAULT}" \
      "--incompatibility_penalty=${INCOMPATIBILITY_PENALTY_DEFAULT}" \
      "--neighbor_adjacent_tile_support=${NEIGHBOR_ADJACENT_TILE_SUPPORT_DEFAULT}" \
      "--max_experiment_time=${MAX_EXPERIMENT_TIME_DEFAULT}"
  done
}

for singleton_weight in "${SINGLETON_WEIGHT_VALUES[@]}"; do
  for social_fusion in "${SOCIAL_FUSION_VALUES[@]}"; do
    run_singleton_weight_condition "$singleton_weight" "$social_fusion"
  done
done
