#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

ensure_python_bin

RUNNER_NAME="same_and_adjacent_tile_support_sweep"
TARGET_RESULTS_ROOT="${REPO_ROOT}/results/social_fusion/${RUNNER_NAME}"
TRIAL_SUMMARY_PATH="${TARGET_RESULTS_ROOT}/trial_summary.csv"

same_tile_support_values=(0 0.25 0.5 0.75 1.0)
adjacent_tile_support_values=(0 0.25 0.5 0.75 1.0)
experiment_commands=()

launch_parallel_commands_preserving_results() {
    local runner_name="$1"
    local target_dir="$2"
    shift 2

    local -a experiment_commands=("$@")
    local worker_count=4
    local temp_root="${TMPDIR:-/tmp}/${runner_name}_$$"

    mkdir -p "${target_dir}"
    mkdir -p "${temp_root}"

    if [[ "${#experiment_commands[@]}" -eq 0 ]]; then
        echo "No remaining experiment batches to run for ${runner_name}."
        return 0
    fi

    local -a worker_scripts
    local -a worker_has_commands

    local worker
    for worker in {1..4}; do
        worker_scripts[$worker]="${temp_root}/worker_${worker}.sh"
        worker_has_commands[$worker]=0
        cat > "${worker_scripts[$worker]}" <<EOF
#!/bin/zsh
set -euo pipefail
cd '${REPO_ROOT}'
EOF
        chmod +x "${worker_scripts[$worker]}"
    done

    local index=0
    local command
    for command in "${experiment_commands[@]}"; do
        worker=$(( (index % worker_count) + 1 ))
        print -- "${command}" >> "${worker_scripts[$worker]}"
        worker_has_commands[$worker]=1
        index=$(( index + 1 ))
    done

    for worker in {1..4}; do
        if [[ "${worker_has_commands[$worker]}" -eq 0 ]]; then
            continue
        fi
        print -- "echo ''" >> "${worker_scripts[$worker]}"
        print -- "echo 'Finished ${runner_name} worker ${worker}'" >> "${worker_scripts[$worker]}"
        print -- "exec zsh" >> "${worker_scripts[$worker]}"
        osascript -e "tell application \"Terminal\" to do script \"zsh '${worker_scripts[$worker]}'\""
    done
}

existing_counts="$("${PYTHON_BIN}" - "${TRIAL_SUMMARY_PATH}" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path

csv_path = Path(sys.argv[1])
counts = Counter()
if csv_path.exists():
    with csv_path.open("r", encoding = "utf-8", newline = "") as file_handle:
        for row in csv.DictReader(file_handle):
            key = (row["same_tile_support"], row["neighbor_adjacent_tile_support"])
            counts[key] += 1

for (same_tile_support, adjacent_tile_support), count in sorted(
    counts.items(),
    key = lambda item: (float(item[0][0]), float(item[0][1])),
):
    print(f"{float(same_tile_support):g} {float(adjacent_tile_support):g} {count}")
PY
)"

typeset -A recorded_counts
if [[ -n "${existing_counts}" ]]; then
    while read -r same_tile_support adjacent_tile_support count; do
        [[ -z "${same_tile_support}" ]] && continue
        key="${same_tile_support}|${adjacent_tile_support}"
        recorded_counts[${(q)key}]="${count}"
    done <<< "${existing_counts}"
fi

all_complete=1
for same_tile_support in "${same_tile_support_values[@]}"; do
    for adjacent_tile_support in "${adjacent_tile_support_values[@]}"; do
        key="${same_tile_support}|${adjacent_tile_support}"
        recorded_count="${recorded_counts[${(q)key}]-0}"
        if (( recorded_count < DEFAULT_TRIALS )); then
            all_complete=0
            break 2
        fi
    done
done

if (( all_complete == 1 )); then
    rm -rf "${TARGET_RESULTS_ROOT}"
    mkdir -p "${TARGET_RESULTS_ROOT}"
    typeset -A recorded_counts=()
fi

for same_tile_support in "${same_tile_support_values[@]}"; do
    for adjacent_tile_support in "${adjacent_tile_support_values[@]}"; do
        key="${same_tile_support}|${adjacent_tile_support}"
        recorded_count="${recorded_counts[${(q)key}]-0}"
        remaining_trials=$(( DEFAULT_TRIALS - recorded_count ))
        if (( remaining_trials <= 0 )); then
            continue
        fi

        experiment_commands+=("$(simulation_command "${RUNNER_NAME}" "--desert" "${DEFAULT_THRESHOLD_TYPE}" "${DEFAULT_SOCIAL_FUSION}" "${DEFAULT_PERSONAL_FUSION}" "${DEFAULT_MASS_FUNCTION}" "${remaining_trials}" "${DEFAULT_SWARM_SIZE}" "${DEFAULT_DECISION_THRESHOLD}" "${DEFAULT_DS_TOP_K}" "${DEFAULT_DS_MAX_NEIGHBORS}" "${DEFAULT_DS_SINGLETON_WEIGHT}" "${DEFAULT_DS_NEIGHBOR_COMPATIBILITY_GAIN}" "${same_tile_support}" "${adjacent_tile_support}")")
    done
done

launch_parallel_commands_preserving_results "${RUNNER_NAME}" "${TARGET_RESULTS_ROOT}" "${experiment_commands[@]}"
