#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results" / "continuous_est"
SUMMARY_ROOT = RESULTS_ROOT / "plan_summaries" / "fixed_threshold_social_fusion_plan"
LOG_ROOT = RESULTS_ROOT / "_logs" / "fixed_threshold_social_fusion_plan"

TERRAINS = ("city", "desert", "mountains", "rural")
DEFAULTS = {
    "number_of_agents": 16,
    "stage_one": "fix_threshold",
    "social_fusion": "comp_boost",
    "threshold_value": 0.5,
    "top_k": 4,
    "singleton_weight": 0.5,
    "compatibility_gain": 1.0,
    "incompatibility_penalty": 0.1,
    "neighbor_adjacent_tile_support": 0.8,
    "max_experiment_time": 400,
    "trials_per_terrain": 100,
    "communication_model": "fixed_8_connected_search_cell_adjacency",
    "stage_two_method": "decentralized_weighted_average_laplacian_consensus",
}

GROUP_1_SWARM_SIZES = (2, 4, 9, 12, 16, 20, 25)
GROUP_1_TOP_K_VALUES = (1, 2, 4, 8)
GROUP_1_SINGLETON_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
GROUP_2_SWEEP_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
SOCIAL_FUSION_VALUES = ("comp_boost", "yager")
PARALLEL_TRIAL_WORKERS = 4


def format_suffix(value):
    return str(value).replace(".", "_")


def apply_overrides(condition, **overrides):
    updated = dict(condition)
    updated.update(overrides)
    return updated


def build_conditions():
    conditions = []
    base = dict(DEFAULTS)

    for swarm_size in GROUP_1_SWARM_SIZES:
        for social_fusion in SOCIAL_FUSION_VALUES:
            condition = apply_overrides(
                base,
                group="group1",
                set_name="swarm_size",
                experiment_name=f"swarm_size_{swarm_size}_{social_fusion}",
                number_of_agents=swarm_size,
                social_fusion=social_fusion,
            )
            conditions.append(condition)

    for top_k in GROUP_1_TOP_K_VALUES:
        for social_fusion in SOCIAL_FUSION_VALUES:
            condition = apply_overrides(
                base,
                group="group1",
                set_name="top_k",
                experiment_name=f"topk_{top_k}_{social_fusion}",
                top_k=top_k,
                social_fusion=social_fusion,
            )
            conditions.append(condition)

    for singleton_weight in GROUP_1_SINGLETON_WEIGHTS:
        suffix = format_suffix(singleton_weight)
        for social_fusion in SOCIAL_FUSION_VALUES:
            condition = apply_overrides(
                base,
                group="group1",
                set_name="singleton_weight",
                experiment_name=f"singleton_weight_{suffix}_{social_fusion}",
                singleton_weight=singleton_weight,
                social_fusion=social_fusion,
            )
            conditions.append(condition)

    for compatibility_gain in GROUP_2_SWEEP_VALUES:
        suffix = format_suffix(compatibility_gain)
        condition = apply_overrides(
            base,
            group="group2",
            set_name="compatibility_gain",
            experiment_name=f"compatibility_gain_{suffix}",
            compatibility_gain=compatibility_gain,
        )
        conditions.append(condition)

    for incompatibility_penalty in GROUP_2_SWEEP_VALUES:
        suffix = format_suffix(incompatibility_penalty)
        condition = apply_overrides(
            base,
            group="group2",
            set_name="incompatibility_penalty",
            experiment_name=f"incompatibility_penalty_{suffix}",
            incompatibility_penalty=incompatibility_penalty,
        )
        conditions.append(condition)

    for neighbor_adjacent_tile_support in GROUP_2_SWEEP_VALUES:
        suffix = format_suffix(neighbor_adjacent_tile_support)
        condition = apply_overrides(
            base,
            group="group2",
            set_name="neighbor_adjacent_tile_support",
            experiment_name=f"neighbor_adjacent_tile_support_{suffix}",
            neighbor_adjacent_tile_support=neighbor_adjacent_tile_support,
        )
        conditions.append(condition)

    for condition_index, condition in enumerate(conditions, start=1):
        condition["condition_index"] = condition_index
        condition["expected_total_trials"] = len(TERRAINS) * int(condition["trials_per_terrain"])

    return conditions


def filter_conditions(conditions, group=None, set_name=None):
    filtered = conditions
    if group is not None and group != "all":
        filtered = [condition for condition in filtered if condition["group"] == group]
    if set_name is not None and set_name != "all":
        filtered = [condition for condition in filtered if condition["set_name"] == set_name]
    return filtered


def get_results_category(experiment_name):
    if experiment_name.startswith("swarm_size"):
        return "swarm_size"
    if experiment_name.startswith("decision_threshold") or experiment_name.startswith("threshold_"):
        return "decision_threshold"
    if experiment_name.startswith("laplacian"):
        return "laplacian"
    if (
        experiment_name.startswith("social_fusion")
        or experiment_name.startswith("compatibility_gain")
        or experiment_name.startswith("incompatibility_penalty")
        or experiment_name.startswith("neighbor_adjacent_tile_support")
    ):
        return "social_fusion"
    return "top_k_ds"


def get_trial_summary_path(experiment_name):
    return RESULTS_ROOT / get_results_category(experiment_name) / experiment_name / "trial_summary.csv"

def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    normalized = str(value).strip().lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    return None

def load_trial_rows(experiment_name):
    summary_path = get_trial_summary_path(experiment_name)
    if not summary_path.exists():
        return []

    deduplicated_rows = {}
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dedup_key = (row.get("terrain"), row.get("trial_index"))
            deduplicated_rows[dedup_key] = row
    return list(deduplicated_rows.values())

def count_completed_trials(experiment_name):
    return len(load_trial_rows(experiment_name))


def is_condition_complete(condition):
    return count_completed_trials(condition["experiment_name"]) >= condition["expected_total_trials"]


def detect_python_executable(user_value):
    if user_value:
        return user_value

    venv_python = REPO_ROOT / "swarm_loc_venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def build_simulation_command(python_executable, condition, terrain, trial_count, trial_start_index):
    return [
        python_executable,
        "src/simulation.py",
        condition["experiment_name"],
        f"--{terrain}",
        f"--trials={trial_count}",
        f"--trial_start_index={trial_start_index}",
        f"--num_agents={condition['number_of_agents']}",
        f"--stage_one={condition['stage_one']}",
        f"--social_fusion={condition['social_fusion']}",
        f"--threshold_value={condition['threshold_value']}",
        f"--top_k={condition['top_k']}",
        f"--singleton_weight={condition['singleton_weight']}",
        f"--compatibility_gain={condition['compatibility_gain']}",
        f"--incompatibility_penalty={condition['incompatibility_penalty']}",
        f"--neighbor_adjacent_tile_support={condition['neighbor_adjacent_tile_support']}",
        f"--max_experiment_time={condition['max_experiment_time']}",
    ]


def split_trial_indices(total_trials, worker_count):
    worker_count = max(1, min(int(worker_count), int(total_trials)))
    base_chunk = int(total_trials) // worker_count
    remainder = int(total_trials) % worker_count
    chunks = []
    start_index = 0

    for worker_index in range(worker_count):
        chunk_size = base_chunk + (1 if worker_index < remainder else 0)
        if chunk_size <= 0:
            continue
        chunks.append((start_index, chunk_size))
        start_index += chunk_size

    return chunks


def run_condition(python_executable, condition, dry_run=False):
    if is_condition_complete(condition):
        print(f"Skipping {condition['experiment_name']}: already complete.")
        return

    print(
        f"Running {condition['condition_index']:02d}/77 {condition['experiment_name']} "
        f"[{condition['group']} / {condition['set_name']}]"
    )
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    commands = []
    for terrain in TERRAINS:
        for worker_index, (trial_start_index, trial_count) in enumerate(
            split_trial_indices(condition["trials_per_terrain"], PARALLEL_TRIAL_WORKERS),
            start=1,
        ):
            command = build_simulation_command(
                python_executable,
                condition,
                terrain,
                trial_count,
                trial_start_index,
            )
            commands.append((terrain, worker_index, trial_start_index, trial_count, command))
            if dry_run:
                print(" ".join(command))

    if dry_run:
        return

    processes = []
    for terrain, worker_index, trial_start_index, trial_count, command in commands:
        log_path = LOG_ROOT / (
            f"{condition['experiment_name']}_{terrain}_worker_{worker_index:02d}"
            f"_trials_{trial_start_index + 1:03d}_{trial_start_index + trial_count:03d}.log"
        )
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((terrain, worker_index, process, log_handle, log_path))

    failed = []
    for terrain, worker_index, process, log_handle, log_path in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            failed.append((terrain, worker_index, return_code, log_path))

    if failed:
        failure_summary = ", ".join(
            f"{terrain}/worker{worker_index} (exit {return_code}, log {log_path})"
            for terrain, worker_index, return_code, log_path in failed
        )
        raise RuntimeError(f"Condition {condition['experiment_name']} failed: {failure_summary}")


def parse_float(value):
    if value in (None, ""):
        return None
    return float(value)


def compute_summary_rows(condition):
    rows = load_trial_rows(condition["experiment_name"])
    by_terrain = []
    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[row["terrain"]].append(row)

    for terrain in TERRAINS:
        terrain_rows = grouped_rows.get(terrain, [])
        by_terrain.append(build_summary_row(condition, terrain, terrain_rows))

    overall_row = build_summary_row(condition, "overall", rows)
    return overall_row, by_terrain


def build_summary_row(condition, terrain_label, rows):
    position_errors = [parse_float(row["position_error_meters"]) for row in rows if parse_float(row["position_error_meters"]) is not None]
    consensus_times = [parse_float(row["consensus_time"]) for row in rows if parse_float(row["consensus_time"]) is not None]
    non_timeout_consensus_times = [
        parse_float(row["consensus_time"])
        for row in rows
        if parse_float(row["consensus_time"]) is not None and parse_bool(row.get("timeout_reached")) is False
    ]
    timeout_flags = [parse_bool(row.get("timeout_reached")) for row in rows if parse_bool(row.get("timeout_reached")) is not None]
    winning_tile_matches = [
        parse_bool(row.get("winning_tile_matches_target"))
        for row in rows
        if parse_bool(row.get("winning_tile_matches_target")) is not None
    ]
    final_position_matches = [
        parse_bool(row.get("final_position_matches_target"))
        for row in rows
        if parse_bool(row.get("final_position_matches_target")) is not None
    ]
    target_in_final_candidates = [
        parse_bool(row.get("target_in_final_candidate_tiles"))
        for row in rows
        if parse_bool(row.get("target_in_final_candidate_tiles")) is not None
    ]

    expected_rows = condition["expected_total_trials"] if terrain_label == "overall" else int(condition["trials_per_terrain"])
    completed_rows = len(rows)

    return {
        "condition_index": condition["condition_index"],
        "group": condition["group"],
        "set_name": condition["set_name"],
        "experiment_name": condition["experiment_name"],
        "terrain": terrain_label,
        "completed_trials": completed_rows,
        "expected_trials": expected_rows,
        "number_of_agents": condition["number_of_agents"],
        "stage_one": condition["stage_one"],
        "social_fusion": condition["social_fusion"],
        "threshold_value": condition["threshold_value"],
        "top_k": condition["top_k"],
        "singleton_weight": condition["singleton_weight"],
        "compatibility_gain": condition["compatibility_gain"],
        "incompatibility_penalty": condition["incompatibility_penalty"],
        "neighbor_adjacent_tile_support": condition["neighbor_adjacent_tile_support"],
        "communication_model": condition["communication_model"],
        "stage_two_method": condition["stage_two_method"],
        "average_position_error_meters": None if not position_errors else sum(position_errors) / len(position_errors),
        "average_consensus_time": None if not consensus_times else sum(consensus_times) / len(consensus_times),
        "average_consensus_time_non_timeout": None if not non_timeout_consensus_times else sum(non_timeout_consensus_times) / len(non_timeout_consensus_times),
        "timeout_percentage": None if not timeout_flags else 100.0 * sum(bool(flag) for flag in timeout_flags) / len(timeout_flags),
        "winning_tile_accuracy_percentage": None if not winning_tile_matches else 100.0 * sum(bool(flag) for flag in winning_tile_matches) / len(winning_tile_matches),
        "final_position_accuracy_percentage": None if not final_position_matches else 100.0 * sum(bool(flag) for flag in final_position_matches) / len(final_position_matches),
        "target_in_final_candidate_percentage": None if not target_in_final_candidates else 100.0 * sum(bool(flag) for flag in target_in_final_candidates) / len(target_in_final_candidates),
    }


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_outputs(conditions):
    overall_rows = []
    terrain_rows = []
    manifest_rows = []

    for condition in conditions:
        overall_row, condition_terrain_rows = compute_summary_rows(condition)
        overall_rows.append(overall_row)
        terrain_rows.extend(condition_terrain_rows)
        manifest_rows.append(
            {
                "condition_index": condition["condition_index"],
                "group": condition["group"],
                "set_name": condition["set_name"],
                "experiment_name": condition["experiment_name"],
                "number_of_agents": condition["number_of_agents"],
                "stage_one": condition["stage_one"],
                "social_fusion": condition["social_fusion"],
                "threshold_value": condition["threshold_value"],
                "top_k": condition["top_k"],
                "singleton_weight": condition["singleton_weight"],
                "compatibility_gain": condition["compatibility_gain"],
                "incompatibility_penalty": condition["incompatibility_penalty"],
                "neighbor_adjacent_tile_support": condition["neighbor_adjacent_tile_support"],
                "communication_model": condition["communication_model"],
                "stage_two_method": condition["stage_two_method"],
                "trials_per_terrain": condition["trials_per_terrain"],
                "expected_total_trials": condition["expected_total_trials"],
                "completed_total_trials": count_completed_trials(condition["experiment_name"]),
                "trial_summary_path": str(get_trial_summary_path(condition["experiment_name"]).relative_to(REPO_ROOT)),
            }
        )

    overall_fieldnames = [
        "condition_index",
        "group",
        "set_name",
        "experiment_name",
        "terrain",
        "completed_trials",
        "expected_trials",
        "number_of_agents",
        "stage_one",
        "social_fusion",
        "threshold_value",
        "top_k",
        "singleton_weight",
        "compatibility_gain",
        "incompatibility_penalty",
        "neighbor_adjacent_tile_support",
        "communication_model",
        "stage_two_method",
        "average_position_error_meters",
        "average_consensus_time",
        "average_consensus_time_non_timeout",
        "timeout_percentage",
        "winning_tile_accuracy_percentage",
        "final_position_accuracy_percentage",
        "target_in_final_candidate_percentage",
    ]

    manifest_fieldnames = [
        "condition_index",
        "group",
        "set_name",
        "experiment_name",
        "number_of_agents",
        "stage_one",
        "social_fusion",
        "threshold_value",
        "top_k",
        "singleton_weight",
        "compatibility_gain",
        "incompatibility_penalty",
        "neighbor_adjacent_tile_support",
        "communication_model",
        "stage_two_method",
        "trials_per_terrain",
        "expected_total_trials",
        "completed_total_trials",
        "trial_summary_path",
    ]

    write_csv(SUMMARY_ROOT / "condition_summary_overall.csv", overall_fieldnames, overall_rows)
    write_csv(SUMMARY_ROOT / "condition_summary_by_terrain.csv", overall_fieldnames, terrain_rows)
    write_csv(SUMMARY_ROOT / "condition_manifest.csv", manifest_fieldnames, manifest_rows)

    metadata = {
        "total_conditions": len(conditions),
        "terrains": list(TERRAINS),
        "defaults": DEFAULTS,
        "summary_root": str(SUMMARY_ROOT.relative_to(REPO_ROOT)),
    }
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    (SUMMARY_ROOT / "plan_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Run and summarize the fixed-threshold social-fusion experiment plan.")
    parser.add_argument("--group", choices=("all", "group1", "group2"), default="all")
    parser.add_argument(
        "--set-name",
        choices=(
            "all",
            "swarm_size",
            "top_k",
            "singleton_weight",
            "compatibility_gain",
            "incompatibility_penalty",
            "neighbor_adjacent_tile_support",
        ),
        default="all",
    )
    parser.add_argument("--python", dest="python_executable")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    conditions = filter_conditions(build_conditions(), group=args.group, set_name=args.set_name)
    python_executable = detect_python_executable(args.python_executable)

    if not args.summarize_only:
        for condition in conditions:
            run_condition(python_executable, condition, dry_run=args.dry_run)

    write_summary_outputs(conditions)
    print(f"Wrote summary files to {SUMMARY_ROOT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
