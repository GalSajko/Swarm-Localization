#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

TRIALS_PER_TERRAIN="${TRIALS_PER_TERRAIN:-100}"
POLL_SECONDS="${POLL_SECONDS:-15}"

is_experiment_complete() {
  local experiment_name="$1"
  local summary_file="results/continuous_est/social_fusion/${experiment_name}/trial_summary.csv"
  if [[ ! -f "$summary_file" ]]; then
    return 1
  fi

  local line_count
  line_count="$(tail -n +2 "$summary_file" | wc -l | tr -d ' ')"
  [[ "$line_count" -ge $((4 * TRIALS_PER_TERRAIN)) ]]
}

echo "Waiting for social_fusion_compatibility_boost and social_fusion_yager to complete..."
until is_experiment_complete "social_fusion_compatibility_boost" && is_experiment_complete "social_fusion_yager"; do
  sleep "$POLL_SECONDS"
done

echo "Social-fusion comparison complete. Starting compatibility-gain sweep."
exec "${SCRIPT_DIR}/run_compatibility_gain_sweep.sh"
