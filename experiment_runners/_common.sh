#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
DEFAULT_TRIALS=200
DEFAULT_SWARM_SIZE=25
DEFAULT_DECISION_THRESHOLD=1.0
DEFAULT_THRESHOLD_TYPE="dynamic"
DEFAULT_SOCIAL_FUSION="comp_boost"
DEFAULT_PERSONAL_FUSION="dempster"
DEFAULT_MASS_FUNCTION="spatial_topk"
DEFAULT_DS_TOP_K=4
DEFAULT_DS_MAX_NEIGHBORS=4
DEFAULT_DS_SINGLETON_WEIGHT=0.75
DEFAULT_DS_NEIGHBOR_COMPATIBILITY_GAIN=1.0
DEFAULT_DS_NEIGHBOR_SAME_TILE_SUPPORT=0.75
DEFAULT_DS_NEIGHBOR_ADJACENT_TILE_SUPPORT=1.0
DEFAULT_DS_NEIGHBOR_INCOMPATIBILITY_PENALTY=0.1

ensure_python_bin() {
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "Python environment not found at ${PYTHON_BIN}" >&2
        exit 1
    fi
}

simulation_command() {
    local experiment_name="$1"
    local terrain_flag="$2"
    local threshold_type="$3"
    local social_fusion="$4"
    local personal_fusion="$5"
    local mass_function="${6:-$DEFAULT_MASS_FUNCTION}"
    local trials="${7:-$DEFAULT_TRIALS}"
    local swarm_size="${8:-$DEFAULT_SWARM_SIZE}"
    local ds_decision_threshold="${9:-$DEFAULT_DECISION_THRESHOLD}"
    local ds_top_k="${10:-$DEFAULT_DS_TOP_K}"
    local ds_max_neighbors="${11:-$DEFAULT_DS_MAX_NEIGHBORS}"
    local ds_singleton_weight="${12:-$DEFAULT_DS_SINGLETON_WEIGHT}"
    local ds_neighbor_compatibility_gain="${13:-$DEFAULT_DS_NEIGHBOR_COMPATIBILITY_GAIN}"
    local ds_neighbor_same_tile_support="${14:-$DEFAULT_DS_NEIGHBOR_SAME_TILE_SUPPORT}"
    local ds_neighbor_adjacent_tile_support="${15:-$DEFAULT_DS_NEIGHBOR_ADJACENT_TILE_SUPPORT}"
    local ds_neighbor_incompatibility_penalty="${16:-$DEFAULT_DS_NEIGHBOR_INCOMPATIBILITY_PENALTY}"

    print -- "'${PYTHON_BIN}' '${REPO_ROOT}/src/simulation.py' '${experiment_name}' ${terrain_flag} --threshold_type=${threshold_type} --social_fusion=${social_fusion} --personal_fusion=${personal_fusion} --mass_function=${mass_function} --trials=${trials} --swarm_size=${swarm_size} --ds_decision_threshold=${ds_decision_threshold} --ds_top_k=${ds_top_k} --ds_max_neighbors=${ds_max_neighbors} --ds_singleton_weight=${ds_singleton_weight} --ds_neighbor_compatibility_gain=${ds_neighbor_compatibility_gain} --ds_neighbor_same_tile_support=${ds_neighbor_same_tile_support} --ds_neighbor_adjacent_tile_support=${ds_neighbor_adjacent_tile_support} --ds_neighbor_incompatibility_penalty=${ds_neighbor_incompatibility_penalty}"
}

launch_parallel_commands() {
    local runner_name="$1"
    local target_dir="$2"
    shift 2

    local -a experiment_commands=("$@")
    local worker_count=4
    local temp_root="${TMPDIR:-/tmp}/${runner_name}_$$"

    mkdir -p "${target_dir}"
    rm -rf "${target_dir}"
    mkdir -p "${target_dir}"
    mkdir -p "${temp_root}"

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
