#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

ensure_python_bin

RUNNER_NAME="compatibility_gain_incompatibility_penalty_sweep"
TARGET_RESULTS_ROOT="${REPO_ROOT}/results/social_fusion/${RUNNER_NAME}"
TRIAL_SUMMARY_PATH="${TARGET_RESULTS_ROOT}/trial_summary.csv"
AGENT_FINAL_STATES_PATH="${TARGET_RESULTS_ROOT}/agent_final_states.csv"
CONSENSUS_DISPERSION_PATH="${TARGET_RESULTS_ROOT}/consensus_dispersion.csv"
DECISION_DYNAMICS_PATH="${TARGET_RESULTS_ROOT}/decision_dynamics.csv"

compatibility_gain_values=(0 0.25 0.5 0.75 1.0)
incompatibility_penalty_values=(0 0.25 0.5 0.75 1.0)
experiment_commands=()
experiment_descriptions=()
remaining_batch_count=0
complete_batch_count=0

trim_overflow_trials() {
    "${PYTHON_BIN}" - "${TRIAL_SUMMARY_PATH}" "${AGENT_FINAL_STATES_PATH}" "${CONSENSUS_DISPERSION_PATH}" "${DECISION_DYNAMICS_PATH}" "${DEFAULT_TRIALS}" <<'PY'
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

trial_summary_path = Path(sys.argv[1])
agent_final_states_path = Path(sys.argv[2])
consensus_dispersion_path = Path(sys.argv[3])
decision_dynamics_path = Path(sys.argv[4])
trial_limit = int(sys.argv[5])

if not trial_summary_path.exists():
    raise SystemExit(0)

PAIR_COLUMNS = ("compatibility_gain", "incompatibility_penalty")
TRIAL_ID_COLUMNS = (
    "experiment_name",
    "trial_index",
    "terrain",
    "number_of_agents",
    "threshold_type",
    "threshold_value",
    "social_fusion_type",
    "personal_fusion_type",
    "mass_function_type",
    "consensus_method",
    "communication_mode",
    "communication_ring_range",
    "top_k",
    "max_neighbors",
    "singleton_weight",
    "compatibility_gain",
    "same_tile_support",
    "incompatibility_penalty",
    "neighbor_adjacent_tile_support",
)

def normalized_pair_key(row):
    return (
        f"{float(row['compatibility_gain']):g}",
        f"{float(row['incompatibility_penalty']):g}",
    )

def trial_id_key(row):
    return tuple(row[column] for column in TRIAL_ID_COLUMNS)

with trial_summary_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    summary_rows = list(reader)
    summary_fieldnames = reader.fieldnames

pair_counts = Counter(normalized_pair_key(row) for row in summary_rows)
if all(count <= trial_limit for count in pair_counts.values()):
    raise SystemExit(0)

summary_seen_by_trial_id = Counter()
kept_instances = set()
kept_summary_rows = []
kept_pair_counts = Counter()
trimmed_pair_counts = Counter()

for row in summary_rows:
    pair_key = normalized_pair_key(row)
    trial_key = trial_id_key(row)
    summary_seen_by_trial_id[trial_key] += 1
    instance_key = (trial_key, summary_seen_by_trial_id[trial_key])
    if kept_pair_counts[pair_key] < trial_limit:
        kept_pair_counts[pair_key] += 1
        kept_instances.add(instance_key)
        kept_summary_rows.append(row)
    else:
        trimmed_pair_counts[pair_key] += 1

with trial_summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=summary_fieldnames)
    writer.writeheader()
    writer.writerows(kept_summary_rows)

def trim_block_file(path: Path):
    if not path.exists():
        return
    temp_path = path.with_suffix(path.suffix + ".tmp")
    seen_by_trial_id = Counter()
    previous_trial_key = None
    current_instance_key = None

    with path.open("r", encoding="utf-8", newline="") as input_handle, temp_path.open("w", encoding="utf-8", newline="") as output_handle:
        reader = csv.DictReader(input_handle)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            trial_key = trial_id_key(row)
            if trial_key != previous_trial_key:
                seen_by_trial_id[trial_key] += 1
                current_instance_key = (trial_key, seen_by_trial_id[trial_key])
                previous_trial_key = trial_key
            if current_instance_key in kept_instances:
                writer.writerow(row)

    temp_path.replace(path)

trim_block_file(agent_final_states_path)
trim_block_file(consensus_dispersion_path)
trim_block_file(decision_dynamics_path)

trimmed_description = ", ".join(
    f"({gain}, {penalty}) -{count}"
    for (gain, penalty), count in sorted(trimmed_pair_counts.items(), key=lambda item: (float(item[0][0]), float(item[0][1])))
)
print(f"Trimmed overflow trials for {trimmed_description}.")
PY
}

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

trim_overflow_trials

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
            key = (row["compatibility_gain"], row["incompatibility_penalty"])
            counts[key] += 1

for (compatibility_gain, incompatibility_penalty), count in sorted(
    counts.items(),
    key = lambda item: (float(item[0][0]), float(item[0][1])),
):
    print(f"{float(compatibility_gain):g} {float(incompatibility_penalty):g} {count}")
PY
)"

typeset -A recorded_counts
if [[ -n "${existing_counts}" ]]; then
    while read -r compatibility_gain incompatibility_penalty count; do
        [[ -z "${compatibility_gain}" ]] && continue
        key="${compatibility_gain}|${incompatibility_penalty}"
        recorded_counts[${(q)key}]="${count}"
    done <<< "${existing_counts}"
fi

for compatibility_gain in "${compatibility_gain_values[@]}"; do
    for incompatibility_penalty in "${incompatibility_penalty_values[@]}"; do
        normalized_compatibility_gain="$(printf '%g' "${compatibility_gain}")"
        normalized_incompatibility_penalty="$(printf '%g' "${incompatibility_penalty}")"
        key="${normalized_compatibility_gain}|${normalized_incompatibility_penalty}"
        recorded_count="${recorded_counts[${(q)key}]-0}"
        remaining_trials=$(( DEFAULT_TRIALS - recorded_count ))
        if (( remaining_trials <= 0 )); then
            complete_batch_count=$(( complete_batch_count + 1 ))
            continue
        fi

        remaining_batch_count=$(( remaining_batch_count + 1 ))
        experiment_descriptions+=("compatibility_gain=${normalized_compatibility_gain}, incompatibility_penalty=${normalized_incompatibility_penalty}, remaining_trials=${remaining_trials}")
        experiment_commands+=("$(simulation_command "${RUNNER_NAME}" "--desert" "${DEFAULT_THRESHOLD_TYPE}" "${DEFAULT_SOCIAL_FUSION}" "${DEFAULT_PERSONAL_FUSION}" "${DEFAULT_MASS_FUNCTION}" "${remaining_trials}" "${DEFAULT_SWARM_SIZE}" "${DEFAULT_DECISION_THRESHOLD}" "${DEFAULT_DS_TOP_K}" "${DEFAULT_DS_MAX_NEIGHBORS}" "${DEFAULT_DS_SINGLETON_WEIGHT}" "${compatibility_gain}" "${DEFAULT_DS_NEIGHBOR_SAME_TILE_SUPPORT}" "${DEFAULT_DS_NEIGHBOR_ADJACENT_TILE_SUPPORT}" "${incompatibility_penalty}")")
    done
done

echo "Completed parameter batches: ${complete_batch_count}"
echo "Remaining parameter batches: ${remaining_batch_count}"
if [[ "${#experiment_descriptions[@]}" -gt 0 ]]; then
    echo "Scheduled missing batches:"
    for description in "${experiment_descriptions[@]}"; do
        echo "  - ${description}"
    done
fi
echo "Prepared ${#experiment_commands[@]} remaining parameter batches for ${RUNNER_NAME}."

launch_parallel_commands_preserving_results "${RUNNER_NAME}" "${TARGET_RESULTS_ROOT}" "${experiment_commands[@]}"
