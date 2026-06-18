#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
TARGET_RESULTS_ROOT="${REPO_ROOT}/results/continuous_est_resetting_vlad/personal_fusion"
TERRAINS=(city desert mountains rural)

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment not found at ${PYTHON_BIN}" >&2
    exit 1
fi

mkdir -p "${TARGET_RESULTS_ROOT}"

for personal_fusion_rule in dempster yager adaptive hybrid; do
    experiment_name="personal_fusion"
    target_experiment_dir="${TARGET_RESULTS_ROOT}/${personal_fusion_rule}"

    rm -rf "${target_experiment_dir}"

    terminal_command="cd '${REPO_ROOT}'; for terrain in ${TERRAINS[*]}; do '${PYTHON_BIN}' '${REPO_ROOT}/src/simulation.py' '${experiment_name}' --\${terrain} --threshold_type=dynamic --social_fusion=comp_boost --personal_fusion='${personal_fusion_rule}'; done; echo ''; echo 'Finished personal fusion run: ${personal_fusion_rule}'; exec zsh"
    osascript -e "tell application \"Terminal\" to do script \"${terminal_command}\""
done
