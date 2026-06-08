#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_CMD="${PYTHON_CMD:-$ROOT_DIR/swarm_loc_venv/bin/python}"
TRIALS_PER_TERRAIN="${TRIALS_PER_TERRAIN:-100}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-4}"
NUMBER_OF_AGENTS_DEFAULT="${NUMBER_OF_AGENTS_DEFAULT:-16}"
THRESHOLD_VALUE_DEFAULT="${THRESHOLD_VALUE_DEFAULT:-0.5}"
TOP_K_DEFAULT="${TOP_K_DEFAULT:-4}"
SINGLETON_WEIGHT_DEFAULT="${SINGLETON_WEIGHT_DEFAULT:-0.5}"
COMPATIBILITY_GAIN_DEFAULT="${COMPATIBILITY_GAIN_DEFAULT:-1.0}"
INCOMPATIBILITY_PENALTY_DEFAULT="${INCOMPATIBILITY_PENALTY_DEFAULT:-0.1}"
NEIGHBOR_ADJACENT_TILE_SUPPORT_DEFAULT="${NEIGHBOR_ADJACENT_TILE_SUPPORT_DEFAULT:-0.8}"
MAX_EXPERIMENT_TIME_DEFAULT="${MAX_EXPERIMENT_TIME_DEFAULT:-400}"
STAGE_ONE_DEFAULT="${STAGE_ONE_DEFAULT:-fix_threshold}"
LOG_DIR="${LOG_DIR:-results/continuous_est/_logs/plan_set_sweeps}"
WORKER_STATE_DIR="${WORKER_STATE_DIR:-results/continuous_est/_worker_state}"
TERRAINS=(city desert mountains rural)

mkdir -p "$LOG_DIR" "$WORKER_STATE_DIR"

format_suffix() {
  printf '%s' "$1" | tr '.' '_'
}

get_results_category() {
  local experiment_name="$1"
  if [[ "$experiment_name" == swarm_size* ]]; then
    printf '%s\n' "swarm_size"
  elif [[ "$experiment_name" == decision_threshold* || "$experiment_name" == threshold_* ]]; then
    printf '%s\n' "decision_threshold"
  elif [[ "$experiment_name" == laplacian* ]]; then
    printf '%s\n' "laplacian"
  elif [[ "$experiment_name" == social_fusion* || "$experiment_name" == compatibility_gain* || "$experiment_name" == incompatibility_penalty* || "$experiment_name" == neighbor_adjacent_tile_support* ]]; then
    printf '%s\n' "social_fusion"
  else
    printf '%s\n' "top_k_ds"
  fi
}

get_trial_summary_path() {
  local experiment_name="$1"
  local category
  category="$(get_results_category "$experiment_name")"
  printf '%s\n' "results/continuous_est/${category}/${experiment_name}/trial_summary.csv"
}

count_unique_trials_for_terrain() {
  local experiment_name="$1"
  local terrain="$2"
  local summary_file
  summary_file="$(get_trial_summary_path "$experiment_name")"

  if [[ ! -f "$summary_file" ]]; then
    printf '0\n'
    return
  fi

  awk -F, -v terrain="$terrain" '
    NR > 1 && $3 == terrain {
      seen[$2] = 1
    }
    END {
      count = 0
      for (trial_key in seen) {
        count++
      }
      print count + 0
    }
  ' "$summary_file"
}

count_unique_trials_in_range_for_terrain() {
  local experiment_name="$1"
  local terrain="$2"
  local start_trial_index="$3"
  local end_trial_index="$4"
  local summary_file
  summary_file="$(get_trial_summary_path "$experiment_name")"

  if [[ ! -f "$summary_file" ]]; then
    printf '0\n'
    return
  fi

  awk -F, -v terrain="$terrain" -v start_idx="$start_trial_index" -v end_idx="$end_trial_index" '
    NR > 1 && $3 == terrain {
      trial_idx = $2 + 0
      if (trial_idx >= start_idx && trial_idx <= end_idx) {
        seen[$2] = 1
      }
    }
    END {
      count = 0
      for (trial_key in seen) {
        count++
      }
      print count + 0
    }
  ' "$summary_file"
}

is_experiment_complete() {
  local experiment_name="$1"
  local terrain
  for terrain in "${TERRAINS[@]}"; do
    if [[ "$(count_unique_trials_for_terrain "$experiment_name" "$terrain")" -lt "$TRIALS_PER_TERRAIN" ]]; then
      return 1
    fi
  done
  return 0
}

split_trials_into_chunks() {
  local total_trials="$1"
  local worker_count="$2"
  local base_chunk remainder index start_index chunk_size

  if (( worker_count < 1 )); then
    worker_count=1
  fi
  if (( worker_count > total_trials )); then
    worker_count="$total_trials"
  fi

  base_chunk=$((total_trials / worker_count))
  remainder=$((total_trials % worker_count))
  start_index=0

  for ((index = 0; index < worker_count; index++)); do
    chunk_size="$base_chunk"
    if (( index < remainder )); then
      chunk_size=$((chunk_size + 1))
    fi
    if (( chunk_size <= 0 )); then
      continue
    fi
    printf '%s %s\n' "$start_index" "$chunk_size"
    start_index=$((start_index + chunk_size))
  done
}

launch_terminal_worker() {
  local worker_label="$1"
  local log_file="$2"
  local done_file="$3"
  shift 3

  local command_string wrapper_script
  command_string="$(printf '%q ' "$@")"
  command_string="${command_string% }"
  wrapper_script="${WORKER_STATE_DIR}/${worker_label}.command.sh"

  cat > "$wrapper_script" <<EOF
#!/usr/bin/env zsh
set -euo pipefail
cd $(printf '%q' "$ROOT_DIR")
rm -f $(printf '%q' "$done_file") $(printf '%q' "${done_file}.exit")
echo "[${worker_label}] starting"
if ${command_string} > $(printf '%q' "$log_file") 2>&1; then
  printf '%s\n' "0" > $(printf '%q' "${done_file}.exit")
  touch $(printf '%q' "$done_file")
  echo "[${worker_label}] finished"
else
  status=\$?
  printf '%s\n' "\$status" > $(printf '%q' "${done_file}.exit")
  touch $(printf '%q' "$done_file")
  echo "[${worker_label}] failed with exit \$status"
  exit \$status
fi
EOF
  chmod +x "$wrapper_script"

  local quoted_root quoted_label quoted_wrapper
  quoted_root="$(printf '%q' "$ROOT_DIR")"
  quoted_label="$(printf '%q' "$worker_label")"
  quoted_wrapper="$(printf '%q' "$wrapper_script")"

  osascript <<EOF
tell application "Terminal"
  activate
  do script "cd ${quoted_root}; ${quoted_wrapper}"
end tell
EOF
}

wait_for_worker_group() {
  local done_files=("$@")
  local done_file
  local sleep_seconds="${WORKER_POLL_SECONDS:-5}"

  while true; do
    local all_done=1
    for done_file in "${done_files[@]}"; do
      if [[ ! -f "$done_file" ]]; then
        all_done=0
        break
      fi
    done
    if (( all_done == 1 )); then
      break
    fi
    sleep "$sleep_seconds"
  done

  local failed=0
  for done_file in "${done_files[@]}"; do
    local exit_file="${done_file}.exit"
    if [[ ! -f "$exit_file" ]]; then
      echo "Missing worker exit file: $exit_file" >&2
      failed=1
      continue
    fi

    local exit_code
    exit_code="$(tr -d '[:space:]' < "$exit_file")"
    if [[ "$exit_code" != "0" ]]; then
      echo "Worker failed: $done_file (exit ${exit_code})" >&2
      failed=1
    fi
  done

  return "$failed"
}

run_four_parallel_trial_workers_for_terrain() {
  local experiment_name="$1"
  local terrain="$2"
  shift 2
  local base_command=("$@")

  local completed_trials
  completed_trials="$(count_unique_trials_for_terrain "$experiment_name" "$terrain")"
  if [[ "$completed_trials" -ge "$TRIALS_PER_TERRAIN" ]]; then
    echo "Skipping ${experiment_name} ${terrain}: ${completed_trials}/${TRIALS_PER_TERRAIN} trials already present."
    return 0
  fi

  echo "Starting ${experiment_name} on ${terrain} with ${PARALLEL_WORKERS} parallel workers"
  local done_files=()
  local worker_index=0

  while read -r trial_start_index trial_count; do
    worker_index=$((worker_index + 1))
    local worker_label="${experiment_name}_${terrain}_worker_${worker_index}"
    local log_file="${LOG_DIR}/${worker_label}_trials_$((trial_start_index + 1))_$((trial_start_index + trial_count)).log"
    local done_file="${WORKER_STATE_DIR}/${worker_label}.done"
    local chunk_start_trial=$((trial_start_index + 1))
    local chunk_end_trial=$((trial_start_index + trial_count))
    local completed_chunk_trials
    completed_chunk_trials="$(count_unique_trials_in_range_for_terrain "$experiment_name" "$terrain" "$chunk_start_trial" "$chunk_end_trial")"

    if [[ "$completed_chunk_trials" -ge "$trial_count" ]]; then
      echo "Skipping ${worker_label}: ${completed_chunk_trials}/${trial_count} trials already present."
      continue
    fi

    done_files+=("$done_file")

    launch_terminal_worker \
      "$worker_label" \
      "$log_file" \
      "$done_file" \
      "${base_command[@]}" \
      "--${terrain}" \
      "--trials=${trial_count}" \
      "--trial_start_index=${trial_start_index}"
  done < <(split_trials_into_chunks "$TRIALS_PER_TERRAIN" "$PARALLEL_WORKERS")

  if [[ "${#done_files[@]}" -gt 0 ]]; then
    echo "Waiting for ${experiment_name} on ${terrain} worker group to finish"
    wait_for_worker_group "${done_files[@]}"
  else
    echo "No missing worker chunks for ${experiment_name} on ${terrain}"
  fi

  local final_completed_trials
  final_completed_trials="$(count_unique_trials_for_terrain "$experiment_name" "$terrain")"
  if [[ "$final_completed_trials" -lt "$TRIALS_PER_TERRAIN" ]]; then
    echo "Incomplete terrain output for ${experiment_name} ${terrain}: ${final_completed_trials}/${TRIALS_PER_TERRAIN}" >&2
    return 1
  fi

  echo "Finished ${experiment_name} on ${terrain}"
}
