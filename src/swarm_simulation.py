"""Simulate the online stage of fuzzy pattern localization. Run from the project's root directory.

Each trial:
1. A search area of one cell per agent is centered on a random drop point (grid search of the main branch).
2. Every agent sweeps its cell and takes a photo every CAPTURE_PERIOD seconds. It stops when the strongest feature of a photo passes the stop rule
   ('response' or 'lowe'). An agent that does not pass within MAX_SEARCH_TIME uses its best photo.
3. Every agent re-centers on its selected feature. Its own position estimate is the map position of the map feature its descriptor matched.
4. The agents vote with the fuzzy memberships of their features and their relative positions (pattern_voting.py). Every agent's final position
   estimate is the vote's answer shifted by its relative position.

Results are appended to 'results/fcm_patterns/<experiment_name>/trial_summary.csv' (one row per trial) and 'agent_final_states.csv' (one row per
agent). Errors are distances between the estimated and the true current positions of the agents, in meters.
"""

import argparse
import os
import time

import numpy as np

import constants
import pattern_voting
import utils
from map_matching import TerrainMap
from search_agent import SearchAgent

TRIAL_SUMMARY_HEADER = [
    "experiment_name", "trial_index", "terrain", "number_of_agents", "stop_rule", "stop_threshold", "seed",
    "swarm_correct", "correct_agent_share", "median_agent_error_meters", "max_agent_error_meters", "mean_position_error_meters",
    "own_match_correct_share", "voting_agents", "agreeing_agents", "fallback_agents", "agents_without_feature",
    "first_decided_time", "half_decided_time", "all_decided_time", "timeout_reached", "median_centering_residual_meters",
]
AGENT_STATE_HEADER = [
    "experiment_name", "trial_index", "terrain", "agent_id", "stop_time", "decision_time", "fallback", "stop_response", "stop_lowe_ratio",
    "true_x", "true_y", "own_estimate_x", "own_estimate_y", "own_error_meters", "final_estimate_x", "final_estimate_y", "final_error_meters",
    "agrees_with_swarm", "centering_residual_meters",
]

def calculate_grid_shape(count):
    """Choose a near-square grid of cells for the given number of agents (an even number or a perfect square), like the main branch."""
    is_perfect_square = int(np.sqrt(count)) ** 2 == count
    if count % 2 != 0 and not is_perfect_square:
        raise ValueError("Grid search requires an even number of agents or a perfect square swarm size.")
    factor_pairs = [(rows, count // rows) for rows in range(1, int(np.sqrt(count)) + 1) if count % rows == 0]
    return min(factor_pairs, key = lambda pair: (abs(pair[1] - pair[0]), pair[0]))

def calculate_search_cells(map_shape, number_of_agents, random_generator):
    """Center a search area of one cell per agent on a random drop point that keeps the whole area inside the map.

    Returns:
        tuple[list[tuple[float, float, float, float]], numpy.ndarray]: Cell bounds (min x, max x, min y, max y) and the drop point.
    """
    map_height, map_width = map_shape
    rows, columns = calculate_grid_shape(number_of_agents)
    cell_width = map_width / constants.TILE_GRID_SIZE[0] / 2
    cell_height = map_height / constants.TILE_GRID_SIZE[1] / 2
    area_width, area_height = columns * cell_width, rows * cell_height
    drop_point = np.array([
        random_generator.uniform(area_width / 2, map_width - area_width / 2),
        random_generator.uniform(area_height / 2, map_height - area_height / 2),
    ])
    min_x, min_y = drop_point[0] - area_width / 2, drop_point[1] - area_height / 2
    cells = [
        (min_x + column * cell_width, min_x + (column + 1) * cell_width, min_y + row * cell_height, min_y + (row + 1) * cell_height)
        for row in range(rows)
        for column in range(columns)
    ]
    return cells, drop_point

def run_search(agents, terrain_map):
    """Let all agents search in synchronized photo steps until every agent has stopped."""
    while any(agent.is_searching for agent in agents):
        searching = [agent for agent in agents if agent.is_searching]
        captures = [(agent, agent.capture(terrain_map.image)) for agent in searching]
        detected = [(agent, feature) for agent, feature in captures if feature is not None]

        if detected and searching[0].stop_rule == "lowe":
            lowe_ratios, nearest_indices = terrain_map.match_descriptors(np.stack([feature["descriptor"] for _, feature in detected]))
            for (_, feature), lowe_ratio, nearest_index in zip(detected, lowe_ratios, nearest_indices):
                feature["lowe_ratio"], feature["nearest_index"] = float(lowe_ratio), int(nearest_index)

        for agent, feature in detected:
            agent.consider(feature)
        for agent in searching:
            if agent.is_searching:
                agent.advance()

def decision_times(agents):
    """First, half (ceil(n / 2)-th) and last decision time of the agents, like the main branch."""
    times = np.sort([agent.decision_time if agent.decision_time is not None else agent.stop_time for agent in agents])
    return float(times[0]), float(times[int(np.ceil(len(times) / 2)) - 1]), float(times[-1])

def run_trial(terrain_map, number_of_agents, stop_rule, stop_threshold, seed):
    """Run one trial.

    Returns:
        tuple[list, list[list]]: Trial summary row (without experiment name and trial index) and agent rows (without experiment name, trial index
            and terrain).
    """
    random_generator = np.random.default_rng(seed)
    np.random.seed(seed)
    cells, _ = calculate_search_cells(terrain_map.map_shape, number_of_agents, random_generator)
    agents = [
        SearchAgent(agent_id, cell, random_generator.choice(["horizontal", "vertical"]), stop_rule, stop_threshold)
        for agent_id, cell in enumerate(cells)
    ]

    run_search(agents, terrain_map)
    voting_agents = [agent for agent in agents if agent.recenter(terrain_map)]

    # Relative positions are measured directly between the agents; here they are exact.
    final_estimates = {}
    agreeing = np.zeros(0, dtype = np.int64)
    if voting_agents:
        reference_position = voting_agents[0].true_position
        relative_positions = np.array([agent.true_position - reference_position for agent in voting_agents])
        result = pattern_voting.vote(relative_positions, np.stack([agent.memberships for agent in voting_agents]), terrain_map)
        if result["reference_position"] is not None:
            agreeing = result["agreeing_agents"]
            for agent, relative_position in zip(voting_agents, relative_positions):
                final_estimates[agent.agent_id] = result["reference_position"] + relative_position

    agent_rows = []
    final_errors, own_errors = [], []
    for index, agent in enumerate(agents):
        own_error = final_error = np.nan
        if agent.true_position is not None:
            own_error = float(np.linalg.norm(agent.own_estimate - agent.true_position) * constants.METERS_PER_PIXEL)
            own_errors.append(own_error)
        final_estimate = final_estimates.get(agent.agent_id)
        if final_estimate is not None:
            final_error = float(np.linalg.norm(final_estimate - agent.true_position) * constants.METERS_PER_PIXEL)
        final_errors.append(final_error)
        voting_index = next((position for position, voting_agent in enumerate(voting_agents) if voting_agent is agent), None)
        agent_rows.append([
            agent.agent_id,
            agent.stop_time,
            agent.decision_time,
            agent.is_fallback,
            None if agent.selected is None else agent.selected["response"],
            None if agent.selected is None else agent.selected["lowe_ratio"],
            *(agent.true_position if agent.true_position is not None else (None, None)),
            *(agent.own_estimate if agent.own_estimate is not None else (None, None)),
            own_error,
            *(final_estimate if final_estimate is not None else (None, None)),
            final_error,
            voting_index is not None and voting_index in agreeing,
            agent.centering_residual,
        ])

    final_errors = np.array(final_errors, dtype = np.float64)
    tolerance = constants.CORRECT_POSITION_TOLERANCE
    is_correct = final_errors < tolerance  # NaN (no estimate) counts as incorrect
    true_positions = np.array([agent.true_position for agent in voting_agents]) if voting_agents else np.zeros((0, 2))
    estimates = np.array([final_estimates[agent.agent_id] for agent in voting_agents if agent.agent_id in final_estimates])
    mean_position_error = (
        float(np.linalg.norm(estimates.mean(axis = 0) - true_positions.mean(axis = 0)) * constants.METERS_PER_PIXEL)
        if len(estimates) == len(voting_agents) and len(estimates) else np.nan
    )
    first_time, half_time, all_time = decision_times(agents)
    residuals = [agent.centering_residual for agent in agents if agent.centering_residual is not None]

    summary = [
        number_of_agents, stop_rule, stop_threshold, seed,
        bool(is_correct.all()), float(is_correct.mean()),
        float(np.nanmedian(final_errors)) if np.isfinite(final_errors).any() else np.nan,
        float(np.nanmax(final_errors)) if np.isfinite(final_errors).any() else np.nan,
        mean_position_error,
        float(np.mean(np.array(own_errors) < tolerance)) if own_errors else np.nan,
        len(voting_agents), len(agreeing),
        sum(agent.is_fallback for agent in agents), len(agents) - len(voting_agents),
        first_time, half_time, all_time, any(agent.is_fallback for agent in agents),
        float(np.median(residuals)) if residuals else np.nan,
    ]
    return summary, agent_rows

def run_experiment(experiment_name, terrains, number_of_agents, trials, stop_rule, stop_threshold, seed):
    """Run all trials of an experiment and append the results to its csv files."""
    output_dir = os.path.join(constants.FCM_RESULTS_DIR, experiment_name)
    summary_path = os.path.join(output_dir, "trial_summary.csv")
    agent_path = os.path.join(output_dir, "agent_final_states.csv")

    for terrain_name in terrains:
        terrain_map = TerrainMap(terrain_name)
        start_time = time.time()
        summaries = []
        for trial_index in range(trials):
            trial_seed = seed + trial_index
            summary, agent_rows = run_trial(terrain_map, number_of_agents, stop_rule, stop_threshold, trial_seed)
            summaries.append(summary)
            utils.write_results_to_csv(summary_path, [experiment_name, trial_index, terrain_name] + summary, TRIAL_SUMMARY_HEADER)
            for agent_row in agent_rows:
                utils.write_results_to_csv(agent_path, [experiment_name, trial_index, terrain_name] + agent_row, AGENT_STATE_HEADER)

        correct = np.mean([summary[4] for summary in summaries])
        median_error = np.nanmedian([summary[6] for summary in summaries])
        half_time = np.median([summary[15] for summary in summaries])
        all_time = np.median([summary[16] for summary in summaries])
        print(
            f"{terrain_name}: {trials} trials with {number_of_agents} agents in {time.time() - start_time:.0f} s. Swarm correct in {100 * correct:.0f}% "
            f"of trials, median agent error {median_error:.1f} m, half of the agents decided after {half_time:.1f} s, all after {all_time:.1f} s.",
            flush = True,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Simulate fuzzy pattern localization of a swarm. Run from the project's root directory.")
    parser.add_argument("--experiment_name", default = "default", help = f"Results go to {constants.FCM_RESULTS_DIR}/<experiment_name>/ (default 'default').")
    parser.add_argument("--terrains", nargs = "+", choices = constants.TERRAIN_NAMES, default = list(constants.TERRAIN_NAMES), help = "Terrains to simulate (default all).")
    parser.add_argument("--number_of_agents", type = int, default = constants.NUMBER_OF_AGENTS, help = f"Swarm size, an even number or a perfect square (default {constants.NUMBER_OF_AGENTS}).")
    parser.add_argument("--trials", type = int, default = 100, help = "Trials per terrain (default 100).")
    parser.add_argument("--stop_rule", choices = constants.STOP_RULES, default = constants.STOP_RULE, help = f"Stop rule (default '{constants.STOP_RULE}').")
    parser.add_argument("--stop_threshold", type = float, default = None, help = f"Response threshold (default {constants.RESPONSE_THRESHOLD}) or Lowe ratio threshold (default {constants.LOWE_RATIO_THRESHOLD}).")
    parser.add_argument("--seed", type = int, default = 0, help = "Seed of the first trial; trial i uses seed + i (default 0).")
    arguments = parser.parse_args()

    default_threshold = constants.RESPONSE_THRESHOLD if arguments.stop_rule == "response" else constants.LOWE_RATIO_THRESHOLD
    run_experiment(
        arguments.experiment_name,
        arguments.terrains,
        arguments.number_of_agents,
        arguments.trials,
        arguments.stop_rule,
        default_threshold if arguments.stop_threshold is None else arguments.stop_threshold,
        arguments.seed,
    )
