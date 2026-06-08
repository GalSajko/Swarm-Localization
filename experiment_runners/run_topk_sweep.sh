#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_experiment_set_common.sh"

TOPK_VALUES=(1 2 4 8)
SOCIAL_FUSION_VALUES=(comp_boost yager)

run_topk_condition() {
  local top_k="$1"
  local social_fusion="$2"
  local experiment_name="topk_${top_k}_${social_fusion}"
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
      "--top_k=${top_k}" \
      "--singleton_weight=${SINGLETON_WEIGHT_DEFAULT}" \
      "--compatibility_gain=${COMPATIBILITY_GAIN_DEFAULT}" \
      "--incompatibility_penalty=${INCOMPATIBILITY_PENALTY_DEFAULT}" \
      "--neighbor_adjacent_tile_support=${NEIGHBOR_ADJACENT_TILE_SUPPORT_DEFAULT}" \
      "--max_experiment_time=${MAX_EXPERIMENT_TIME_DEFAULT}"
  done
}

for top_k in "${TOPK_VALUES[@]}"; do
  for social_fusion in "${SOCIAL_FUSION_VALUES[@]}"; do
    run_topk_condition "$top_k" "$social_fusion"
  done
done
