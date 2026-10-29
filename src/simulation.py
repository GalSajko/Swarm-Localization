import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

from agent import Agent
import constants
import decision_making as dm
import utils

NUMBER_OF_TRIALS = 200
LEGACY_DS_METHOD_ALIAS = "ds"
MAX_EXPERIMENT_TIME = 400

SIMULATION_SPEED_UP_FACTOR = 5.0
SIMULATION_LOOP_PERIOD = 0.01

RESULTS_DIR = "results/continuous_est_resetting_vlad"
PERSONAL_FUSION_RESULTS_DIR = "results/personal_fusion"
SOCIAL_FUSION_RESULTS_DIR = "results/social_fusion"
DECISION_THRESHOLD_RESULTS_DIR = "results/decision_threshold"
SWARM_SIZE_RESULTS_DIR = "results/swarm_size"
TERRAIN_FLAGS = {
    "--city": "city",
    "--desert": "desert",
    "--mountains": "mountains",
    "--rural": "rural",
}

AGENT_COLOR_NO_DECISION = (255, 0, 0)
AGENT_COLOR_CORRECT_DECISION = (0, 255, 0)
AGENT_COLOR_WRONG_DECISION = (0, 0, 255)
SEARCH_AREA_COLOR = (0, 255, 255)
SUBAREA_GRID_SHADOW_COLOR = (30, 30, 30)
SUBAREA_GRID_COLOR = (245, 245, 245)
START_POSITION_COLOR = (0, 0, 255)
WINNER_TILE_COLOR = (0, 0, 255)
FINAL_POSITION_COLOR = (0, 255, 0)
CONSENSUS_POSITION_COLOR = (0, 255, 255)
METERS_PER_PIXEL = 2.0

consensus_times = []


def show_dynamic_threshold_debug_plot(agents, trial_index, terrain):
    """Debug-only plot of each agent's dynamic threshold over one completed trial."""
    dynamic_agents = [agent for agent in agents if not agent.use_fixed_threshold]
    if not dynamic_agents:
        return

    fig, ax = plt.subplots(figsize = (10, 6))
    for agent in sorted(dynamic_agents, key = lambda current_agent: int(current_agent.id)):
        elapsed_times = [0.0]
        threshold_values = [float(agent.decision_threshold)]
        for snapshot in agent.decision_dynamics:
            elapsed_times.append(float(snapshot["elapsed_time"]))
            threshold_values.append(float(snapshot["threshold"]))

        ax.plot(
            elapsed_times,
            threshold_values,
            linewidth = 1.2,
            alpha = 0.9,
            label = f"Agent {agent.id}",
        )

    ax.set_title(f"Dynamic Threshold Progress - Trial {trial_index + 1} ({terrain})")
    ax.set_xlabel("Elapsed simulation time [s]")
    ax.set_ylabel("Decision threshold")
    ax.set_ylim(constants.DS_THRESHOLD_MIN - 0.02, constants.DS_THRESHOLD_MAX + 0.02)
    ax.grid(True, linestyle = ":", linewidth = 0.5)
    ax.legend(loc = "center left", bbox_to_anchor = (1.02, 0.5), fontsize = 8)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def parse_choice_argument(raw_extra_args, prefix, valid_choices, default_value):
    """Extract a string choice argument and validate it."""
    value = None
    for argument in raw_extra_args:
        if argument.startswith(prefix):
            value = argument.split("=", 1)[1]
            break
    if value is None:
        return default_value
    if value not in valid_choices:
        raise ValueError(f"Invalid value '{value}' for {prefix}. Expected one of: {', '.join(valid_choices)}")
    return value

def parse_required_choice_argument(raw_extra_args, prefix, valid_choices):
    """Extract one required string choice argument and validate it."""
    value = None
    for argument in raw_extra_args:
        if argument.startswith(prefix):
            value = argument.split("=", 1)[1]
            break
    if value is None:
        raise ValueError(f"Missing required argument {prefix}{{...}}. Expected one of: {', '.join(valid_choices)}")
    if value not in valid_choices:
        raise ValueError(f"Invalid value '{value}' for {prefix}. Expected one of: {', '.join(valid_choices)}")
    return value

def parse_optional_int_argument(raw_extra_args, prefix, default_value = None, minimum = None):
    """Extract one optional integer CLI argument and validate it."""
    value = None
    for argument in raw_extra_args:
        if argument.startswith(prefix):
            value = argument.split("=", 1)[1]
            break
    if value is None:
        return default_value
    try:
        parsed_value = int(value)
    except ValueError as error:
        raise ValueError(f"Invalid integer value '{value}' for {prefix}.") from error
    if minimum is not None and parsed_value < minimum:
        raise ValueError(f"Invalid value '{value}' for {prefix}. Expected an integer >= {minimum}.")
    return parsed_value

def parse_optional_float_argument(raw_extra_args, prefix, default_value = None, minimum = None, maximum = None):
    """Extract one optional float CLI argument and validate it."""
    value = None
    for argument in raw_extra_args:
        if argument.startswith(prefix):
            value = argument.split("=", 1)[1]
            break
    if value is None:
        return default_value
    try:
        parsed_value = float(value)
    except ValueError as error:
        raise ValueError(f"Invalid float value '{value}' for {prefix}.") from error
    if minimum is not None and parsed_value < minimum:
        raise ValueError(f"Invalid value '{value}' for {prefix}. Expected a float >= {minimum}.")
    if maximum is not None and parsed_value > maximum:
        raise ValueError(f"Invalid value '{value}' for {prefix}. Expected a float <= {maximum}.")
    return parsed_value

def build_results_paths(experiment_name):
    """Build output CSV paths using the experiment-family directories from experiments_readme.md."""
    experiment_stem = os.path.splitext(os.path.basename(experiment_name))[0]
    personal_fusion_named_sets = {
        "personal_fusion_rules_sweep",
        "personal_fusion_rules_sweep_disabled_social",
        "mass_function_mode_sweep",
        "topk_neighbors_sweep",
        "singleton_weight_sweep",
    }
    social_fusion_named_sets = {
        "social_fusion_rules_sweep",
        "compatibility_gain_sweep",
        "compatibility_gain_incompatibility_penalty_sweep",
        "same_and_adjacent_tile_support_sweep",
        "incompatibility_penalty_sweep",
    }
    if experiment_stem in personal_fusion_named_sets:
        experiment_dir = os.path.join(PERSONAL_FUSION_RESULTS_DIR, experiment_stem)
    elif experiment_stem in social_fusion_named_sets:
        experiment_dir = os.path.join(SOCIAL_FUSION_RESULTS_DIR, experiment_stem)
    elif experiment_stem == "decision_threshold" or experiment_stem.startswith("threshold_"):
        experiment_dir = os.path.join(DECISION_THRESHOLD_RESULTS_DIR, experiment_stem)
    elif experiment_stem == "swarm_size":
        experiment_dir = os.path.join(SWARM_SIZE_RESULTS_DIR, experiment_stem)
    elif experiment_stem.startswith("social_fusion_"):
        experiment_stem = experiment_stem.removeprefix("social_fusion_")
        experiment_dir = os.path.join(SOCIAL_FUSION_RESULTS_DIR, experiment_stem)
    elif experiment_stem.startswith("personal_fusion_"):
        experiment_stem = experiment_stem.removeprefix("personal_fusion_")
        experiment_dir = os.path.join(PERSONAL_FUSION_RESULTS_DIR, experiment_stem)
    elif experiment_stem.startswith("personal_fusion_test_"):
        experiment_stem = experiment_stem.removeprefix("personal_fusion_test_")
        experiment_dir = os.path.join("results/personal_fusion_test", experiment_stem)
    elif experiment_stem.startswith("continuous"):
        experiment_dir = os.path.join(RESULTS_DIR, "continuous", experiment_stem)
    elif (
        experiment_stem.startswith("social_fusion")
        or experiment_stem.startswith("compatibility_gain")
        or experiment_stem.startswith("incompatibility_penalty")
        or experiment_stem.startswith("neighbor_adjacent_tile_support")
    ):
        experiment_dir = os.path.join(SOCIAL_FUSION_RESULTS_DIR, experiment_stem)
    else:
        experiment_dir = os.path.join(RESULTS_DIR, "top_k_ds", experiment_stem)

    return {
        "trial_summary": os.path.join(experiment_dir, "trial_summary.csv"),
        "agent_final_states": os.path.join(experiment_dir, "agent_final_states.csv"),
        "decision_dynamics": os.path.join(experiment_dir, "decision_dynamics.csv"),
        "consensus_dispersion": os.path.join(experiment_dir, "consensus_dispersion.csv"),
    }

def calculate_random_position(image, min_x = None, max_x = None, min_y = None, max_y = None):
    """Calculate one random center position that keeps the whole agent inside the image.

    Args:
        image (numpy.ndarray): Original image.
        min_x (float, optional): Minimum x-coordinate of the agent center.
        max_x (float, optional): Maximum x-coordinate of the agent center.
        min_y (float, optional): Minimum y-coordinate of the agent center.
        max_y (float, optional): Maximum y-coordinate of the agent center.

    Returns:
        numpy.ndarray: Random center position.
    """
    min_x = constants.AGENT_SIZE[0] / 2 if min_x is None else min_x
    max_x = image.shape[1] - constants.AGENT_SIZE[0] / 2 if max_x is None else max_x
    min_y = constants.AGENT_SIZE[1] / 2 if min_y is None else min_y
    max_y = image.shape[0] - constants.AGENT_SIZE[1] / 2 if max_y is None else max_y

    return np.array(
        [
            np.random.uniform(min_x, max_x),
            np.random.uniform(min_y, max_y),
        ],
        dtype = np.float32,
    )

def calculate_random_search_area_center_position(image):
    """Sample a shared swarm start position that stays at the true center of the search area.

    The sampled position is constrained so the full search area can be centered on it
    without clipping against the terrain border.
    """
    rows, cols = calculate_grid_shape(constants.NUMBER_OF_AGENTS)
    image_height, image_width = image.shape[:2]
    subimage_width = image_width / constants.SUBIMAGE_GRID_SIZE[0]
    subimage_height = image_height / constants.SUBIMAGE_GRID_SIZE[1]

    cell_width = subimage_width / 2
    cell_height = subimage_height / 2
    search_width = cols * cell_width
    search_height = rows * cell_height

    min_x = search_width / 2
    max_x = image_width - search_width / 2
    min_y = search_height / 2
    max_y = image_height - search_height / 2

    if min_x > max_x:
        min_x = max_x = image_width / 2
    if min_y > max_y:
        min_y = max_y = image_height / 2

    return calculate_random_position(
        image,
        min_x = min_x,
        max_x = max_x,
        min_y = min_y,
        max_y = max_y,
    )

def calculate_initial_positions_from_search_cells(search_cells):
    """Spawn each agent directly on a traversable corner of its assigned search cell.

    Args:
        search_cells (list[tuple[float, float, float, float]]): Cell bounds.

    Returns:
        numpy.ndarray: Array of per-agent start positions.
    """
    positions = []
    half_width = constants.AGENT_SIZE[0] / 2
    half_height = constants.AGENT_SIZE[1] / 2

    for min_x, max_x, min_y, max_y in search_cells:
        traversable_min_x = min_x + half_width
        traversable_max_x = max_x - half_width
        traversable_min_y = min_y + half_height
        traversable_max_y = max_y - half_height

        if traversable_min_x > traversable_max_x:
            traversable_min_x = traversable_max_x = (min_x + max_x) / 2
        if traversable_min_y > traversable_max_y:
            traversable_min_y = traversable_max_y = (min_y + max_y) / 2

        positions.append(
            np.array([traversable_min_x, traversable_min_y], dtype = np.float32)
        )

    return np.asarray(positions, dtype = np.float32)

def calculate_grid_shape(count):
    """Choose an exact near-square cell grid for supported swarm sizes.

    Args:
        count (int): Number of agents/cells.

    Returns:
        tuple[int, int]: Number of rows and columns.
    """
    sqrt_count = int(np.sqrt(count))
    is_perfect_square = sqrt_count * sqrt_count == count
    if count % 2 != 0 and not is_perfect_square:
        raise ValueError("Grid-search requires an even number of agents or a perfect square swarm size.")

    factor_pairs = [
        (rows, count // rows)
        for rows in range(1, int(np.sqrt(count)) + 1)
        if count % rows == 0
    ]

    rows, cols = min(factor_pairs, key = lambda pair: (abs(pair[1] - pair[0]), pair[0]))
    return rows, cols

def calculate_search_cells(image, initial_position):
    """Create a compact rectangular search area around the swarm spawn position.

    Args:
        image (numpy.ndarray): Original image.
        initial_position (numpy.ndarray): Shared swarm spawn position.

    Returns:
        tuple[list[tuple[float, float, float, float]], tuple[float, float, float, float]]:
            Cell bounds and the outer search-area boundary.
    """
    rows, cols = calculate_grid_shape(constants.NUMBER_OF_AGENTS)
    image_height, image_width = image.shape[:2]
    subimage_width = image_width / constants.SUBIMAGE_GRID_SIZE[0]
    subimage_height = image_height / constants.SUBIMAGE_GRID_SIZE[1]

    cell_width = subimage_width / 2
    cell_height = subimage_height / 2
    search_width = cols * cell_width
    search_height = rows * cell_height

    min_x = np.clip(initial_position[0] - search_width / 2, 0, image_width - search_width)
    min_y = np.clip(initial_position[1] - search_height / 2, 0, image_height - search_height)
    max_x = min_x + search_width
    max_y = min_y + search_height

    cells = []
    for row in range(rows):
        for col in range(cols):
            cells.append(
                (
                    float(min_x + col * cell_width),
                    float(min_x + (col + 1) * cell_width),
                    float(min_y + row * cell_height),
                    float(min_y + (row + 1) * cell_height),
                )
            )

    return cells, (float(min_x), float(max_x), float(min_y), float(max_y))

def calculate_fixed_adjacency_neighbors(agent_count, ring_range = None):
    """Build fixed neighbor lists over the search-cell grid up to the given ring range."""
    rows, cols = calculate_grid_shape(agent_count)
    ring_range = constants.COMMUNICATION_RING_RANGE if ring_range is None else int(ring_range)
    ring_range = max(1, ring_range)
    neighbor_ids = []

    for agent_index in range(agent_count):
        row_index = agent_index // cols
        col_index = agent_index % cols
        agent_neighbors = []

        for row_offset in range(-ring_range, ring_range + 1):
            for col_offset in range(-ring_range, ring_range + 1):
                if row_offset == 0 and col_offset == 0:
                    continue
                neighbor_row = row_index + row_offset
                neighbor_col = col_index + col_offset
                if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
                    agent_neighbors.append(neighbor_row * cols + neighbor_col)

        neighbor_ids.append(sorted(agent_neighbors))

    return neighbor_ids

def calculate_grid_matrix(agent_count):
    """Build a near-square row-major swarm grid matrix of agent ids."""
    rows, cols = calculate_grid_shape(agent_count)
    return np.arange(agent_count, dtype = np.int32).reshape(rows, cols)

def calculate_search_area_tile_count(search_area_bounds, image_shape):
    """Count how many subarea tiles intersect the rectangular search area."""
    min_x, max_x, min_y, max_y = search_area_bounds
    image_height, image_width = image_shape[:2]
    subimage_width = image_width / constants.SUBIMAGE_GRID_SIZE[0]
    subimage_height = image_height / constants.SUBIMAGE_GRID_SIZE[1]

    covered_tiles = 0
    for tile_x in range(constants.SUBIMAGE_GRID_SIZE[0]):
        tile_min_x = tile_x * subimage_width
        tile_max_x = (tile_x + 1) * subimage_width
        overlaps_x = tile_min_x < max_x and tile_max_x > min_x
        if not overlaps_x:
            continue

        for tile_y in range(constants.SUBIMAGE_GRID_SIZE[1]):
            tile_min_y = tile_y * subimage_height
            tile_max_y = (tile_y + 1) * subimage_height
            overlaps_y = tile_min_y < max_y and tile_max_y > min_y
            if overlaps_y:
                covered_tiles += 1

    return covered_tiles

def calculate_subarea_tile_size_square_meters(image_shape):
    """Calculate one terrain subarea tile area in square meters."""
    image_height, image_width = image_shape[:2]
    subimage_width_pixels = image_width / constants.SUBIMAGE_GRID_SIZE[0]
    subimage_height_pixels = image_height / constants.SUBIMAGE_GRID_SIZE[1]
    return float(subimage_width_pixels * subimage_height_pixels * (METERS_PER_PIXEL ** 2))

def format_focal_set_label(focal_set, image_shape):
    """Convert a focal set of tile indices to a stable human-readable label."""
    if not focal_set:
        return ""

    sorted_tiles = sorted(int(tile_index) for tile_index in focal_set)
    tile_names = [convert_decision_index_to_subimage_name(tile_index, image_shape) for tile_index in sorted_tiles]
    if len(tile_names) == 1:
        return tile_names[0]
    return "{" + "|".join(tile_names) + "}"

def calculate_decision_progress_times(agents):
    """Extract timing milestones from per-agent first-decision times."""
    decision_times = sorted(
        float(agent.first_decision_time)
        for agent in agents
        if agent.first_decision_time is not None
    )
    if not decision_times:
        return None, None, None, None, 0

    first_decided_time = decision_times[0]
    decided_agent_count = len(decision_times)
    half_index = max(0, int(np.ceil(len(agents) / 2)) - 1)
    half_decided_time = decision_times[half_index] if decided_agent_count > half_index else None
    all_decided_time = decision_times[-1] if decided_agent_count == len(agents) else None
    late_stage_waiting_gap = None if half_decided_time is None or all_decided_time is None else all_decided_time - half_decided_time
    return first_decided_time, half_decided_time, all_decided_time, late_stage_waiting_gap, decided_agent_count

def visualize_search_area(image, search_area_bounds):
    """Draw the outer boundary of the swarm search area.

    Args:
        image (numpy.ndarray): Copy of the original image.
        search_area_bounds (tuple[float, float, float, float]): Search area bounds.
    """
    min_x, max_x, min_y, max_y = search_area_bounds
    start_point = (int(round(min_x)), int(round(min_y)))
    end_point = (int(round(max_x)), int(round(max_y)))
    cv2.rectangle(image, start_point, end_point, SEARCH_AREA_COLOR, 6)

def visualize_subarea_grid(image):
    """Draw the terrain-wide 20x20 segmentation grid.

    Args:
        image (numpy.ndarray): Copy of the original image.
    """
    image_height, image_width = image.shape[:2]

    for col in range(1, constants.SUBIMAGE_GRID_SIZE[0]):
        x = int(round(col * image_width / constants.SUBIMAGE_GRID_SIZE[0]))
        cv2.line(image, (x, 0), (x, image_height), SUBAREA_GRID_SHADOW_COLOR, 3, cv2.LINE_AA)
        cv2.line(image, (x, 0), (x, image_height), SUBAREA_GRID_COLOR, 1, cv2.LINE_AA)

    for row in range(1, constants.SUBIMAGE_GRID_SIZE[1]):
        y = int(round(row * image_height / constants.SUBIMAGE_GRID_SIZE[1]))
        cv2.line(image, (0, y), (image_width, y), SUBAREA_GRID_SHADOW_COLOR, 3, cv2.LINE_AA)
        cv2.line(image, (0, y), (image_width, y), SUBAREA_GRID_COLOR, 1, cv2.LINE_AA)

def visualize_start_position(image, initial_position):
    """Draw the shared swarm starting position.

    Args:
        image (numpy.ndarray): Copy of the original image.
        initial_position (numpy.ndarray): Shared swarm spawn position.
    """
    center = tuple(np.rint(initial_position).astype(np.int32))
    cv2.circle(image, center, 12, START_POSITION_COLOR, -1, cv2.LINE_AA)

def visualize_start_position_cross(image, initial_position):
    """Draw a high-visibility red cross at the shared swarm starting position."""
    center = tuple(np.rint(initial_position).astype(np.int32))
    half_size = 22
    shadow_thickness = 8
    cross_thickness = 4
    shadow_color = (255, 255, 255)

    start_diag_a = (center[0] - half_size, center[1] - half_size)
    end_diag_a = (center[0] + half_size, center[1] + half_size)
    start_diag_b = (center[0] - half_size, center[1] + half_size)
    end_diag_b = (center[0] + half_size, center[1] - half_size)

    cv2.line(image, start_diag_a, end_diag_a, shadow_color, shadow_thickness, cv2.LINE_AA)
    cv2.line(image, start_diag_b, end_diag_b, shadow_color, shadow_thickness, cv2.LINE_AA)
    cv2.line(image, start_diag_a, end_diag_a, START_POSITION_COLOR, cross_thickness, cv2.LINE_AA)
    cv2.line(image, start_diag_b, end_diag_b, START_POSITION_COLOR, cross_thickness, cv2.LINE_AA)

def visualize_final_position_cross(image, final_position):
    """Draw a high-visibility green cross at the weighted-average final position."""
    center = tuple(np.rint(final_position).astype(np.int32))
    half_size = 18
    shadow_thickness = 8
    cross_thickness = 4
    shadow_color = (255, 255, 255)

    start_diag_a = (center[0] - half_size, center[1] - half_size)
    end_diag_a = (center[0] + half_size, center[1] + half_size)
    start_diag_b = (center[0] - half_size, center[1] + half_size)
    end_diag_b = (center[0] + half_size, center[1] - half_size)

    cv2.line(image, start_diag_a, end_diag_a, shadow_color, shadow_thickness, cv2.LINE_AA)
    cv2.line(image, start_diag_b, end_diag_b, shadow_color, shadow_thickness, cv2.LINE_AA)
    cv2.line(image, start_diag_a, end_diag_a, FINAL_POSITION_COLOR, cross_thickness, cv2.LINE_AA)
    cv2.line(image, start_diag_b, end_diag_b, FINAL_POSITION_COLOR, cross_thickness, cv2.LINE_AA)

def visualize_colored_cross(image, position, color, half_size = 14):
    """Draw a high-visibility cross in the requested color."""
    center = tuple(np.rint(position).astype(np.int32))
    shadow_thickness = 6
    cross_thickness = 3
    shadow_color = (255, 255, 255)
    start_diag_a = (center[0] - half_size, center[1] - half_size)
    end_diag_a = (center[0] + half_size, center[1] + half_size)
    start_diag_b = (center[0] - half_size, center[1] + half_size)
    end_diag_b = (center[0] + half_size, center[1] - half_size)

    cv2.line(image, start_diag_a, end_diag_a, shadow_color, shadow_thickness, cv2.LINE_AA)
    cv2.line(image, start_diag_b, end_diag_b, shadow_color, shadow_thickness, cv2.LINE_AA)
    cv2.line(image, start_diag_a, end_diag_a, color, cross_thickness, cv2.LINE_AA)
    cv2.line(image, start_diag_b, end_diag_b, color, cross_thickness, cv2.LINE_AA)

def visualize_live_white_cross(image, position, half_size = 16):
    """Draw a high-contrast white cross for live Laplacian consensus positions."""
    if position is None or not np.all(np.isfinite(position)):
        return
    center = tuple(np.rint(position).astype(np.int32))
    outline_color = (0, 0, 0)
    cross_color = (255, 255, 255)
    outline_thickness = 8
    cross_thickness = 5
    start_diag_a = (center[0] - half_size, center[1] - half_size)
    end_diag_a = (center[0] + half_size, center[1] + half_size)
    start_diag_b = (center[0] - half_size, center[1] + half_size)
    end_diag_b = (center[0] + half_size, center[1] - half_size)

    cv2.line(image, start_diag_a, end_diag_a, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.line(image, start_diag_b, end_diag_b, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.line(image, start_diag_a, end_diag_a, cross_color, cross_thickness, cv2.LINE_AA)
    cv2.line(image, start_diag_b, end_diag_b, cross_color, cross_thickness, cv2.LINE_AA)

def visualize_agent(agent, image, target_subimage_name, image_dir):
    """Visualize agent on the image.

    Args:
        agent (Agent): Agent to draw.
        image (numpy.ndarray): Copy of the original image.
        target_subimage_name (str): Ground-truth subimage label for the start position.
        image_dir (str): Directory containing candidate subimages.
    """
    position = agent.get_position()
    start_point = np.array(position - constants.AGENT_SIZE / 2).astype(np.int32)
    end_point = np.array(position + constants.AGENT_SIZE / 2).astype(np.int32)

    decision = agent.get_decision()
    if decision is None:
        color = AGENT_COLOR_NO_DECISION
    else:
        decided_image_name = utils.convert_decision_to_subimage_name(decision, image_dir)
        color = AGENT_COLOR_CORRECT_DECISION if decided_image_name == target_subimage_name else AGENT_COLOR_WRONG_DECISION

    cv2.rectangle(image, start_point, end_point, color, 5)

def convert_decision_to_tile_bounds(decision, image_shape):
    """Convert a decision index to the full-image bounds of the selected tile."""
    subimage_width = image_shape[1] / constants.SUBIMAGE_GRID_SIZE[0]
    subimage_height = image_shape[0] / constants.SUBIMAGE_GRID_SIZE[1]

    rows = constants.SUBIMAGE_GRID_SIZE[1]
    x_index = int(decision) // rows
    y_index = int(decision) % rows

    start_point = (
        int(round(x_index * subimage_width)),
        int(round(y_index * subimage_height)),
    )
    end_point = (
        int(round((x_index + 1) * subimage_width)),
        int(round((y_index + 1) * subimage_height)),
    )

    return start_point, end_point

def convert_decision_to_tile_center(decision, image_shape):
    """Convert a decision index to the center position of the selected tile."""
    start_point, end_point = convert_decision_to_tile_bounds(decision, image_shape)
    return np.array(
        [
            (start_point[0] + end_point[0]) / 2.0,
            (start_point[1] + end_point[1]) / 2.0,
        ],
        dtype = np.float32,
    )

def collect_final_decision_tiles(agents):
    """Collect the union of tiles contained in each agent's strongest focal set."""
    decision_tiles = {
        tile_index
        for agent in agents
        for tile_index in agent.get_best_focal_set()
    }
    return filter_to_largest_adjacent_component(decision_tiles)

def collect_final_decision_tiles_unfiltered(agents):
    """Collect the full union of tiles contained in each agent's strongest focal set."""
    decision_tiles = {
        tile_index
        for agent in agents
        for tile_index in agent.get_best_focal_set()
    }
    return sorted(decision_tiles)

def convert_decision_to_tile_indices(decision):
    """Convert a decision index to x/y tile indices in the terrain grid."""
    rows = constants.SUBIMAGE_GRID_SIZE[1]
    return int(decision) // rows, int(decision) % rows

def are_adjacent_tiles(first_decision, second_decision):
    """Return whether two tiles are adjacent, including diagonal adjacency."""
    first_x, first_y = convert_decision_to_tile_indices(first_decision)
    second_x, second_y = convert_decision_to_tile_indices(second_decision)
    return max(abs(first_x - second_x), abs(first_y - second_y)) == 1

def filter_to_largest_adjacent_component(decision_tiles):
    """Keep only the largest connected component of adjacent final decision tiles."""
    remaining_tiles = set(decision_tiles)
    if not remaining_tiles:
        return []

    connected_components = []
    while remaining_tiles:
        seed_tile = remaining_tiles.pop()
        component = {seed_tile}
        frontier = [seed_tile]

        while frontier:
            current_tile = frontier.pop()
            adjacent_tiles = {
                candidate
                for candidate in list(remaining_tiles)
                if are_adjacent_tiles(current_tile, candidate)
            }
            if adjacent_tiles:
                component.update(adjacent_tiles)
                frontier.extend(adjacent_tiles)
                remaining_tiles.difference_update(adjacent_tiles)

        connected_components.append(component)

    largest_component = max(connected_components, key = lambda component: (len(component), -min(component)))
    return sorted(largest_component)

def calculate_final_decision_scores(agents, decision_tiles):
    """Aggregate agent decision scores over the final decision tile set."""
    decision_scores = {}
    for decision in decision_tiles:
        decision_scores[decision] = sum(
            float(agent.ds_scores[decision])
            for agent in agents
            if agent.ds_scores is not None and len(agent.ds_scores) > decision
        )
    return decision_scores

def collect_current_best_hypothesis_tiles(agents):
    """Collect the unique set of current best-hypothesis tiles across agents."""
    best_tiles = set()
    for agent in agents:
        best_hypothesis, _ = agent.get_best_decision_score()
        if best_hypothesis is not None:
            best_tiles.add(int(best_hypothesis))
    return sorted(best_tiles)

def calculate_current_best_hypothesis_scores(agents, decision_tiles):
    """Aggregate current best-decision scores over the live best-hypothesis tile set."""
    decision_scores = {}
    for decision in decision_tiles:
        decision_scores[decision] = sum(
            float(agent.get_best_decision_score()[1])
            for agent in agents
            if agent.get_best_decision_score()[0] == decision
        )
    return decision_scores

def calculate_position_error_meters(initial_position, final_position):
    """Convert the Euclidean final-position error from pixels to meters."""
    if initial_position is None or final_position is None:
        return None
    error_pixels = np.linalg.norm(np.asarray(final_position) - np.asarray(initial_position))
    return float(error_pixels * METERS_PER_PIXEL)

def get_heatmap_color(score, min_score, max_score):
    """Map a score to a red heat color where stronger support means deeper red."""
    if max_score <= min_score:
        normalized = 1.0
    else:
        normalized = (score - min_score) / (max_score - min_score)

    red_intensity = int(round(80 + normalized * 175))
    green_blue_intensity = int(round(40 * (1.0 - normalized)))
    return (green_blue_intensity, green_blue_intensity, red_intensity)

def get_final_decision_colors(decision_scores):
    """Assign a heat-style red color to each final decision tile."""
    if not decision_scores:
        return {}

    min_score = min(decision_scores.values())
    max_score = max(decision_scores.values())
    return {
        decision: get_heatmap_color(score, min_score, max_score)
        for decision, score in decision_scores.items()
    }

def visualize_final_decision_tiles(image, decision_tiles, decision_colors):
    """Draw colored boundaries around the unique set of final decision tiles."""
    for decision in decision_tiles:
        color = decision_colors[decision]
        start_point, end_point = convert_decision_to_tile_bounds(decision, image.shape)
        cv2.rectangle(image, start_point, end_point, color, 4, cv2.LINE_AA)

def visualize_live_decision_tiles(image, decision_tiles, decision_colors):
    """Draw thicker live best-hypothesis tile outlines for Laplacian visualization."""
    for decision in decision_tiles:
        color = decision_colors[decision]
        start_point, end_point = convert_decision_to_tile_bounds(decision, image.shape)
        cv2.rectangle(image, start_point, end_point, color, 7, cv2.LINE_AA)

def visualize_final_agents(image, agents, decision_colors):
    """Draw agents using the same color as their final decision tile."""
    for agent in agents:
        decision = agent.get_decision()
        if decision is None:
            continue

        position = agent.get_position()
        start_point = np.array(position - constants.AGENT_SIZE / 2).astype(np.int32)
        end_point = np.array(position + constants.AGENT_SIZE / 2).astype(np.int32)
        color = decision_colors.get(decision, WINNER_TILE_COLOR)
        cv2.rectangle(image, start_point, end_point, color, 5, cv2.LINE_AA)

def visualize_live_consensus_positions(image, agents):
    """Draw each frozen agent's current continuous consensus estimate as a small white cross."""
    for agent in agents:
        if not agent.is_continuous_frozen():
            continue
        consensus_position = agent.get_consensus_position()
        if consensus_position is None:
            continue
        visualize_live_white_cross(image, consensus_position, half_size = 16)

def visualize_live_best_hypothesis_tiles(image, agents):
    """Draw current best-hypothesis tiles using the same heat-style outline as the final overlay."""
    decision_tiles = collect_current_best_hypothesis_tiles(agents)
    if not decision_tiles:
        return
    decision_scores = calculate_current_best_hypothesis_scores(agents, decision_tiles)
    decision_colors = get_final_decision_colors(decision_scores)
    visualize_live_decision_tiles(image, decision_tiles, decision_colors)

def collect_final_consensus_positions(agents):
    """Collect all initialized per-agent consensus positions."""
    consensus_positions = []
    for agent in agents:
        consensus_position = agent.get_consensus_position()
        if consensus_position is not None:
            consensus_positions.append(np.asarray(consensus_position, dtype = np.float32))
    return consensus_positions

def calculate_continuous_final_position(consensus_positions):
    """Return the mean consensus position across all initialized agents."""
    if not consensus_positions:
        return None
    return np.mean(np.asarray(consensus_positions, dtype = np.float32), axis = 0)

def calculate_consensus_spread(consensus_positions, mean_position):
    """Return x/y standard deviation and mean per-agent error of final consensus estimates in meters."""
    if not consensus_positions or mean_position is None:
        return None, None, None

    position_array = np.asarray(consensus_positions, dtype = np.float32)
    standard_deviation_xy_pixels = np.std(position_array, axis = 0, dtype = np.float32)
    standard_deviation_xy_meters = standard_deviation_xy_pixels * METERS_PER_PIXEL
    mean_error_meters = float(
        np.mean(
            np.linalg.norm(position_array - np.asarray(mean_position, dtype = np.float32), axis = 1),
            dtype = np.float32,
        ) * METERS_PER_PIXEL
    )
    return float(standard_deviation_xy_meters[0]), float(standard_deviation_xy_meters[1]), mean_error_meters

def calculate_consensus_dispersion_snapshot(agents, cycle_index, elapsed_time):
    """Summarize the current spatial dispersion of agent consensus positions for one synchronized cycle."""
    consensus_positions = collect_final_consensus_positions(agents)
    if not consensus_positions:
        return {
            "cycle_index": cycle_index,
            "elapsed_time": float(elapsed_time),
            "num_agents_with_consensus": 0,
            "centroid_x": None,
            "centroid_y": None,
            "std_x_meters": None,
            "std_y_meters": None,
            "mean_distance_to_centroid_meters": None,
            "std_distance_to_centroid_meters": None,
            "average_pairwise_distance_meters": None,
            "max_pairwise_distance_meters": None,
        }

    position_array = np.asarray(consensus_positions, dtype = np.float32)
    centroid = np.mean(position_array, axis = 0, dtype = np.float32)
    std_xy_pixels = np.std(position_array, axis = 0, dtype = np.float32)
    distances_to_centroid_pixels = np.linalg.norm(position_array - centroid, axis = 1)

    max_pairwise_distance_pixels = 0.0
    pairwise_distances_pixels = []
    if len(position_array) >= 2:
        for left_index in range(len(position_array)):
            deltas = position_array[left_index + 1:] - position_array[left_index]
            if len(deltas):
                current_pairwise_distances = np.linalg.norm(deltas, axis = 1)
                pairwise_distances_pixels.extend(current_pairwise_distances.tolist())
                candidate_max = float(np.max(current_pairwise_distances))
                max_pairwise_distance_pixels = max(max_pairwise_distance_pixels, candidate_max)

    return {
        "cycle_index": cycle_index,
        "elapsed_time": float(elapsed_time),
        "num_agents_with_consensus": int(len(position_array)),
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "std_x_meters": float(std_xy_pixels[0] * METERS_PER_PIXEL),
        "std_y_meters": float(std_xy_pixels[1] * METERS_PER_PIXEL),
        "mean_distance_to_centroid_meters": float(np.mean(distances_to_centroid_pixels, dtype = np.float32) * METERS_PER_PIXEL),
        "std_distance_to_centroid_meters": float(np.std(distances_to_centroid_pixels, dtype = np.float32) * METERS_PER_PIXEL),
        "average_pairwise_distance_meters": None if not pairwise_distances_pixels else float(np.mean(pairwise_distances_pixels, dtype = np.float32) * METERS_PER_PIXEL),
        "max_pairwise_distance_meters": float(max_pairwise_distance_pixels * METERS_PER_PIXEL),
    }

def build_continuous_final_overlay(image, agents, initial_position = None):
    """Build the final continuous-mode visualization with per-agent consensus crosses."""
    overlay_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    decision_tiles = collect_final_decision_tiles_unfiltered(agents)
    decision_scores = calculate_final_decision_scores(agents, decision_tiles)
    decision_colors = get_final_decision_colors(decision_scores)
    visualize_final_decision_tiles(overlay_image, decision_tiles, decision_colors)
    visualize_final_agents(overlay_image, agents, decision_colors)
    if initial_position is not None:
        visualize_start_position_cross(overlay_image, initial_position)

    consensus_positions = collect_final_consensus_positions(agents)
    for consensus_position in consensus_positions:
        visualize_colored_cross(overlay_image, consensus_position, CONSENSUS_POSITION_COLOR)

    final_position = calculate_continuous_final_position(consensus_positions)
    if final_position is not None:
        visualize_final_position_cross(overlay_image, final_position)

    return overlay_image, final_position, consensus_positions

def wait_for_next_trial_key(window_name, overlay_image):
    """Show the final overlay persistently and wait for a fresh key press."""
    idle_polls_required = 6
    idle_polls = 0

    while idle_polls < idle_polls_required:
        cv2.imshow(window_name, overlay_image)
        pressed_key = cv2.waitKey(50)
        if pressed_key == -1:
            idle_polls += 1
        else:
            idle_polls = 0

    while True:
        cv2.imshow(window_name, overlay_image)
        pressed_key = cv2.waitKey(50)
        if pressed_key != -1:
            return

def check_for_continuous_termination(agents):
    """Check whether every agent has completed the post-freeze continuous consensus phase."""
    for agent in agents:
        if not agent.is_continuous_done():
            return False
    return True

def format_final_decision(decisions, image_dir):
    """Format one trial's final agent decisions for reporting."""
    decision_names = [
        utils.convert_decision_to_subimage_name(decision, image_dir)
        for decision in decisions
        if decision is not None
    ]
    unique_decisions = sorted(set(decision_names))
    if not unique_decisions:
        return "NO DECISION"
    if len(unique_decisions) == 1:
        return unique_decisions[0]
    return "|".join(unique_decisions)

def format_final_decision_with_support(decisions, agents, image_shape):
    """Format final unique decisions with per-decision agent counts and combined scores."""
    valid_decisions = sorted({int(decision) for decision in decisions if decision is not None})
    if not valid_decisions:
        return "NO DECISION"

    formatted_entries = []
    for decision in valid_decisions:
        decision_name = convert_decision_index_to_subimage_name(decision, image_shape)
        agreeing_agents = [
            agent
            for agent, agent_decision in zip(agents, decisions)
            if agent_decision is not None and int(agent_decision) == decision
        ]
        agreement_count = len(agreeing_agents)
        combined_score = sum(
            float(agent.ds_scores[decision])
            for agent in agreeing_agents
            if agent.ds_scores is not None and len(agent.ds_scores) > decision
        )
        formatted_entries.append(
            f"{decision_name} (agents={agreement_count}, score={combined_score:.3f})"
        )

    return " | ".join(formatted_entries)

def get_image_dir_for_terrain(terrain):
    """Return the subimage directory for the selected terrain.

    Args:
        terrain (str): Selected terrain or None.

    Returns:
        str: Directory containing subimages.
    """
    if terrain is None:
        return constants.SUBIMAGES_DIR + os.sep
    return os.path.join(constants.SUBIMAGES_DIR, terrain) + os.sep

def get_codebook_path_for_terrain(terrain):
    """Return the VLAD codebook path for the selected terrain.

    Args:
        terrain (str): Selected terrain or None.

    Returns:
        str: Path to the VLAD vectors file.
    """
    if terrain is None:
        return constants.DEFAULT_VLAD_VECTORS_PATH
    return os.path.join(constants.CODEBOOKS_DIR, f"vlad_vectors_{terrain}.npy")

def get_original_image_path_for_terrain(terrain):
    """Return the full terrain image path for the selected terrain.

    Args:
        terrain (str): Selected terrain.

    Returns:
        str: Path to the terrain image.
    """
    return os.path.join(constants.ORIGINAL_IMAGES_DIR, constants.TERRAIN_IMAGE_NAMES[terrain])

def convert_position_to_subimage_name(position, image_shape):
    """Map a full-image position to the corresponding generated subimage name.

    Args:
        position (numpy.ndarray): Agent center position in the full terrain image.
        image_shape (tuple[int, ...]): Terrain image shape.

    Returns:
        str: Subimage name in the same format as generated dataset tiles.
    """
    subimage_width = image_shape[1] // constants.SUBIMAGE_GRID_SIZE[0]
    subimage_height = image_shape[0] // constants.SUBIMAGE_GRID_SIZE[1]

    x_index = min(int(position[0] // subimage_width), constants.SUBIMAGE_GRID_SIZE[0] - 1)
    y_index = min(int(position[1] // subimage_height), constants.SUBIMAGE_GRID_SIZE[1] - 1)

    return f"{x_index + 1:02d}_{y_index + 1:02d}"

def convert_position_to_decision_index(position, image_shape):
    """Map a full-image position to the corresponding terrain-tile index."""
    subimage_width = image_shape[1] // constants.SUBIMAGE_GRID_SIZE[0]
    subimage_height = image_shape[0] // constants.SUBIMAGE_GRID_SIZE[1]

    x_index = min(int(position[0] // subimage_width), constants.SUBIMAGE_GRID_SIZE[0] - 1)
    y_index = min(int(position[1] // subimage_height), constants.SUBIMAGE_GRID_SIZE[1] - 1)
    return int(x_index * constants.SUBIMAGE_GRID_SIZE[1] + y_index)

def convert_decision_index_to_subimage_name(decision, image_shape):
    """Convert a decision index to the corresponding subimage name."""
    subimage_width = image_shape[1] // constants.SUBIMAGE_GRID_SIZE[0]
    subimage_height = image_shape[0] // constants.SUBIMAGE_GRID_SIZE[1]
    x_index = min(int((decision % (constants.SUBIMAGE_GRID_SIZE[1] * constants.SUBIMAGE_GRID_SIZE[0])) // constants.SUBIMAGE_GRID_SIZE[1]), constants.SUBIMAGE_GRID_SIZE[0] - 1)
    y_index = min(int(decision % constants.SUBIMAGE_GRID_SIZE[1]), constants.SUBIMAGE_GRID_SIZE[1] - 1)
    if subimage_width <= 0 or subimage_height <= 0:
        return f"{x_index + 1:02d}_{y_index + 1:02d}"
    return f"{x_index + 1:02d}_{y_index + 1:02d}"

def convert_decision_index_to_tile_center(decision, image_shape):
    """Return the center position of one terrain tile decision in image coordinates."""
    subimage_width = image_shape[1] / constants.SUBIMAGE_GRID_SIZE[0]
    subimage_height = image_shape[0] / constants.SUBIMAGE_GRID_SIZE[1]
    x_index = min(
        int((decision % (constants.SUBIMAGE_GRID_SIZE[1] * constants.SUBIMAGE_GRID_SIZE[0])) // constants.SUBIMAGE_GRID_SIZE[1]),
        constants.SUBIMAGE_GRID_SIZE[0] - 1,
    )
    y_index = min(int(decision % constants.SUBIMAGE_GRID_SIZE[1]), constants.SUBIMAGE_GRID_SIZE[1] - 1)
    return np.array(
        [
            (x_index + 0.5) * subimage_width,
            (y_index + 0.5) * subimage_height,
        ],
        dtype = np.float32,
    )

def calculate_stage1_decision_dispersion_metrics(decisions, image_shape):
    """Return spatial dispersion and cluster structure metrics for final Stage-1 decisions."""
    valid_decisions = [int(decision) for decision in decisions if decision is not None]
    if len(valid_decisions) <= 1:
        if not valid_decisions:
            return None, 0, None
        return 0.0, 1, 1.0

    decision_positions = np.asarray(
        [convert_decision_index_to_tile_center(decision, image_shape) for decision in valid_decisions],
        dtype = np.float32,
    )
    pairwise_distances_meters = []
    for left_index in range(len(decision_positions)):
        deltas = decision_positions[left_index + 1:] - decision_positions[left_index]
        if len(deltas):
            pairwise_distances_meters.extend(
                (np.linalg.norm(deltas, axis = 1) * METERS_PER_PIXEL).tolist()
            )
    mean_pairwise_distance_meters = (
        None if not pairwise_distances_meters else float(np.mean(pairwise_distances_meters, dtype = np.float32))
    )

    decision_counts = {}
    for decision in valid_decisions:
        decision_counts[decision] = decision_counts.get(decision, 0) + 1

    unique_decisions = sorted(decision_counts)
    decision_components = []
    visited_decisions = set()
    rows = constants.SUBIMAGE_GRID_SIZE[1]

    def are_adjacent_decisions(left_decision, right_decision):
        left_x = int(left_decision) // rows
        left_y = int(left_decision) % rows
        right_x = int(right_decision) // rows
        right_y = int(right_decision) % rows
        return max(abs(left_x - right_x), abs(left_y - right_y)) <= 1

    for decision in unique_decisions:
        if decision in visited_decisions:
            continue
        stack = [decision]
        component_decisions = set()
        component_agent_count = 0
        visited_decisions.add(decision)
        while stack:
            current_decision = stack.pop()
            component_decisions.add(current_decision)
            component_agent_count += decision_counts[current_decision]
            for neighbor_decision in unique_decisions:
                if neighbor_decision in visited_decisions:
                    continue
                if are_adjacent_decisions(current_decision, neighbor_decision):
                    visited_decisions.add(neighbor_decision)
                    stack.append(neighbor_decision)
        decision_components.append((component_decisions, component_agent_count))

    number_of_connected_decision_clusters = len(decision_components)
    largest_cluster_share = float(
        max(component_agent_count for _, component_agent_count in decision_components) / len(valid_decisions)
    )

    return (
        mean_pairwise_distance_meters,
        number_of_connected_decision_clusters,
        largest_cluster_share,
    )

def format_final_decision_scores(decision_scores, image_shape):
    """Format final decision-tile scores for terminal output."""
    formatted_scores = []
    for decision, score in sorted(decision_scores.items(), key = lambda item: item[1], reverse = True):
        formatted_scores.append(
            f"{convert_decision_index_to_subimage_name(decision, image_shape)}:{score:.6f}"
        )
    return formatted_scores

def experiment(
    trial_index,
    terrain,
    personal_fusion_rule,
    social_fusion_rule = "comp_boost",
    use_fixed_threshold = False,
    visualize = True,
    use_uncertainty_threshold = False,
    use_concentration_threshold = False,
):
    """Single run of the experiment.

    Args:
        terrain (str): Selected terrain.
    """
    image_dir = get_image_dir_for_terrain(terrain)
    codebook_path = get_codebook_path_for_terrain(terrain)
    test_image = cv2.imread(get_original_image_path_for_terrain(terrain), cv2.IMREAD_GRAYSCALE)
    initial_position = calculate_random_search_area_center_position(test_image)
    target_tile_index = convert_position_to_decision_index(initial_position, test_image.shape)
    target_subimage_name = convert_position_to_subimage_name(initial_position, test_image.shape)

    print(
        f"TRIAL: {trial_index + 1}, START POSITION: "
        f"({initial_position[0]:.2f}, {initial_position[1]:.2f})"
    )

    centroids = np.load(constants.TRAINED_CENTROIDS_PATH)
    codebook = np.load(codebook_path)

    search_cells, search_area_bounds = calculate_search_cells(test_image, initial_position)
    search_area_tile_count = calculate_search_area_tile_count(search_area_bounds, test_image.shape)
    subarea_tile_size_square_meters = calculate_subarea_tile_size_square_meters(test_image.shape)
    positions = calculate_initial_positions_from_search_cells(search_cells)
    fixed_adjacency_neighbors = calculate_fixed_adjacency_neighbors(len(positions))
    grid_matrix = calculate_grid_matrix(len(positions))
    trial_start_time = time.time()
    agents = [
        Agent(
            test_image,
            centroids,
            codebook,
            positions[i],
            str(i),
            SIMULATION_SPEED_UP_FACTOR,
            search_cells[i],
            use_fixed_threshold = use_fixed_threshold,
            social_fusion_rule = social_fusion_rule,
            personal_fusion_rule = personal_fusion_rule,
            fixed_neighbor_ids = fixed_adjacency_neighbors[i],
            grid_matrix = grid_matrix,
            trial_start_time = trial_start_time,
            use_uncertainty_threshold = use_uncertainty_threshold,
            use_concentration_threshold = use_concentration_threshold,
        )
        for i in range(len(positions))
    ]
    for agent in agents:
        agent.attach_swarm(agents)

    start_time = trial_start_time
    elapsed_time = 0
    max_experiment_time = MAX_EXPERIMENT_TIME / SIMULATION_SPEED_UP_FACTOR
    final_overlay_image = None
    consensus_dispersion_history = []
    communication_cycle_index = 0

    while True:
        loop_start_time = time.time()
        image_copy = np.copy(test_image) if visualize else None
        if visualize:
            visualize_subarea_grid(image_copy)
            visualize_search_area(image_copy, search_area_bounds)
            visualize_start_position(image_copy, initial_position)

        simulated_elapsed_time = elapsed_time * SIMULATION_SPEED_UP_FACTOR
        max_completed_cycle_index = max(agent.decision_cycle_index for agent in agents)
        while communication_cycle_index < max_completed_cycle_index:
            communication_cycle_index += 1
            consensus_dispersion_history.append(
                calculate_consensus_dispersion_snapshot(
                    agents,
                    communication_cycle_index,
                    simulated_elapsed_time,
                )
            )

        decisions = []
        for agent in agents:
            if visualize:
                visualize_agent(agent, image_copy, target_subimage_name, image_dir)
            decisions.append(agent.get_decision())

        if visualize:
            visualize_live_best_hypothesis_tiles(image_copy, agents)
            visualize_live_consensus_positions(image_copy, agents)

        termination_reached = check_for_continuous_termination(agents)

        if termination_reached or elapsed_time > max_experiment_time:
            consensus_times.append(elapsed_time)
            timeout_reached = elapsed_time > max_experiment_time
            decisions = []
            for agent in agents:
                best_hypothesis, _ = agent.get_best_decision_score()
                agent.set_decision(best_hypothesis)
                decisions.append(best_hypothesis)
            for agent in agents:
                agent.stop_threads()

            final_overlay_image, final_position, final_consensus_positions = build_continuous_final_overlay(
                test_image,
                agents,
                initial_position,
            )
            final_decision_tiles = collect_final_decision_tiles(agents)
            final_decision_scores = calculate_final_decision_scores(agents, final_decision_tiles)
            consensus_std_x_meters, consensus_std_y_meters, consensus_mean_error_meters = calculate_consensus_spread(
                final_consensus_positions,
                final_position,
            )
            winning_tile_index = None if not final_decision_scores else max(
                final_decision_scores.items(),
                key = lambda item: item[1],
            )[0]
            winning_tile_name = (
                ""
                if winning_tile_index is None
                else convert_decision_index_to_subimage_name(winning_tile_index, test_image.shape)
            )
            winning_tile_score = None if winning_tile_index is None else float(final_decision_scores.get(winning_tile_index, 0.0))
            winning_tile_agreement_count = 0 if winning_tile_index is None else sum(
                1
                for agent in agents
                if winning_tile_index in agent.get_best_focal_set()
            )
            final_position_tile_index = (
                None if final_position is None else convert_position_to_decision_index(final_position, test_image.shape)
            )
            final_position_tile_name = (
                ""
                if final_position_tile_index is None
                else convert_decision_index_to_subimage_name(final_position_tile_index, test_image.shape)
            )
            final_candidate_tile_labels = [convert_decision_index_to_subimage_name(tile_index, test_image.shape) for tile_index in final_decision_tiles]
            target_in_final_candidate_tiles = target_tile_index in set(final_decision_tiles)
            winning_tile_matches_target = winning_tile_index == target_tile_index if winning_tile_index is not None else False
            final_position_matches_target = (
                final_position_tile_index == target_tile_index if final_position_tile_index is not None else False
            )
            position_error_meters = calculate_position_error_meters(initial_position, final_position)
            number_of_different_final_decisions = len({decision for decision in decisions if decision is not None})
            number_of_final_candidate_tiles = len(final_decision_tiles)
            stage1_mean_pairwise_decision_distance_meters, stage1_number_of_connected_decision_clusters, stage1_largest_cluster_share = calculate_stage1_decision_dispersion_metrics(
                decisions,
                test_image.shape,
            )
            all_received_opinion_counts = [
                snapshot["unique_received_opinion_count"]
                for agent in agents
                for snapshot in agent.decision_dynamics
            ]
            average_unique_received_opinions_per_agent_per_cycle = (
                float(np.mean(all_received_opinion_counts, dtype = np.float32))
                if all_received_opinion_counts else 0.0
            )
            first_decided_time, half_decided_time, all_decided_time, late_stage_waiting_gap, decided_agent_count = calculate_decision_progress_times(agents)

            decision = format_final_decision(decisions, image_dir)
            decision_with_support = format_final_decision_with_support(decisions, agents, test_image.shape)

            elapsed_time_rescaled = elapsed_time * SIMULATION_SPEED_UP_FACTOR
            print(
                f'FINAL DECISION: {decision_with_support}, '
                f'ELAPSED TIME: {elapsed_time_rescaled}, '
                f'FINAL POSITION ERROR [m]: {position_error_meters}'
            )
            if personal_fusion_rule == "adaptive":
                adaptive_yager_count = sum(agent.personal_fusion_yager_count for agent in agents)
                adaptive_dempster_count = sum(agent.personal_fusion_dempster_count for agent in agents)
                adaptive_total_count = adaptive_yager_count + adaptive_dempster_count
                if adaptive_total_count > 0:
                    yager_percentage = 100.0 * adaptive_yager_count / adaptive_total_count
                    dempster_percentage = 100.0 * adaptive_dempster_count / adaptive_total_count
                    print(
                        "ADAPTIVE PERSONAL FUSION SELECTION: "
                        f"Yager={yager_percentage:.1f}% ({adaptive_yager_count}/{adaptive_total_count}), "
                        f"Dempster={dempster_percentage:.1f}% ({adaptive_dempster_count}/{adaptive_total_count})"
                    )
            # print(f"FINAL DECISION SCORES: {format_final_decision_scores(final_decision_scores, test_image.shape)}")
            # formatted_consensus_positions = [
            #     f"agent {agent.id}: ({consensus_position[0]:.2f}, {consensus_position[1]:.2f})"
            #     for agent in agents
            #     for consensus_position in [agent.get_consensus_position()]
            #     if consensus_position is not None
            # ]
            # print(f"CONTINUOUS FINAL CONSENSUS POSITIONS: {formatted_consensus_positions}")
            # print(
            #     "CONTINUOUS CONSENSUS SPREAD: "
            #     f"std_x_meters={consensus_std_x_meters}, std_y_meters={consensus_std_y_meters}, "
            #     f"mean_error_meters={consensus_mean_error_meters}"
            # )
            # print(f"FINAL POSITION ERROR [m]: {position_error_meters}")

            if visualize:
                cv2.imshow(f'{terrain}', final_overlay_image)
                wait_for_next_trial_key(f'{terrain}', final_overlay_image)
                cv2.destroyAllWindows()
            # show_dynamic_threshold_debug_plot(agents, trial_index, terrain)
            trial_summary = {
                "elapsed_time": elapsed_time_rescaled,
                "target_tile_index": int(target_tile_index),
                "target_tile_name": target_subimage_name,
                "winning_tile_index": None if winning_tile_index is None else int(winning_tile_index),
                "winning_tile_name": winning_tile_name,
                "winning_tile_matches_target": winning_tile_matches_target,
                "final_position_tile_index": None if final_position_tile_index is None else int(final_position_tile_index),
                "final_position_tile_name": final_position_tile_name,
                "final_position_matches_target": final_position_matches_target,
                "final_candidate_tiles": "|".join(final_candidate_tile_labels),
                "target_in_final_candidate_tiles": target_in_final_candidate_tiles,
                "position_error_meters": position_error_meters,
                "search_area_tile_count": search_area_tile_count,
                "subarea_tile_size_square_meters": subarea_tile_size_square_meters,
                "number_of_different_final_decisions": number_of_different_final_decisions,
                "number_of_final_candidate_tiles": number_of_final_candidate_tiles,
                "stage1_mean_pairwise_decision_distance_meters": stage1_mean_pairwise_decision_distance_meters,
                "stage1_number_of_connected_decision_clusters": stage1_number_of_connected_decision_clusters,
                "stage1_largest_cluster_share": stage1_largest_cluster_share,
                "timeout_reached": timeout_reached,
                "first_decided_time": first_decided_time,
                "half_decided_time": half_decided_time,
                "all_decided_time": all_decided_time,
                "late_stage_waiting_gap": late_stage_waiting_gap,
                "decided_agent_count": decided_agent_count,
                "average_unique_received_opinions_per_agent_per_cycle": average_unique_received_opinions_per_agent_per_cycle,
                "final_position_x": None if final_position is None else float(final_position[0]),
                "final_position_y": None if final_position is None else float(final_position[1]),
                "consensus_std_x_meters": consensus_std_x_meters,
                "consensus_std_y_meters": consensus_std_y_meters,
                "consensus_mean_error_meters": consensus_mean_error_meters,
            }
            return trial_summary, agents, test_image.shape, consensus_dispersion_history

        if visualize:
            cv2.imshow(f'{terrain}', image_copy)
            cv2.waitKey(1)

        elapsed_time = time.time() - start_time
        utils.sleep_with_adjustable_period(SIMULATION_LOOP_PERIOD / SIMULATION_SPEED_UP_FACTOR, loop_start_time)

def main(
    experiment_name,
    terrain,
    personal_fusion_rule,
    threshold_type,
    social_fusion_rule = "comp_boost",
    visualize = False,
    trials = None,
):
    """Main experiment entry.

    Args:
        experiment_name (str): Base name used for results files inside results/continuous_est.
    """
    sqrt_agent_count = int(np.sqrt(constants.NUMBER_OF_AGENTS))
    is_perfect_square = sqrt_agent_count * sqrt_agent_count == constants.NUMBER_OF_AGENTS
    if constants.NUMBER_OF_AGENTS % 2 != 0 and not is_perfect_square:
        raise ValueError("Grid-search requires an even number of agents or a perfect square swarm size.")

    social_fusion_type = social_fusion_rule
    results_experiment_name = experiment_name
    if experiment_name == "personal_fusion":
        results_experiment_name = f"{experiment_name}_{personal_fusion_rule}"
    elif experiment_name == "social_fusion":
        results_experiment_name = f"{experiment_name}_{social_fusion_type}"
    results_paths = build_results_paths(results_experiment_name)
    use_fixed_threshold = threshold_type == "fixed"
    use_uncertainty_threshold = threshold_type == "uncertainty"
    use_concentration_threshold = threshold_type == "concentration"
    effective_social_fusion_rule = social_fusion_rule
    if social_fusion_rule == "disabled":
        dm.configure_dempster_shafer(
            compatibility_gain = 0.0,
            incompatibility_penalty = 1.0,
        )
        effective_social_fusion_rule = "comp_boost"
    consensus_method = "continuous_two_stage"
    communication_mode = f"fixed_ring_neighbors_{constants.COMMUNICATION_RING_RANGE}"

    trial_count = NUMBER_OF_TRIALS if trials is None else max(1, int(trials))

    for trial in range(trial_count):
        absolute_trial_index = trial
        trial_summary, agents, image_shape, consensus_dispersion_history = experiment(
            absolute_trial_index,
            terrain,
            personal_fusion_rule,
            effective_social_fusion_rule,
            use_fixed_threshold,
            visualize,
            use_uncertainty_threshold = use_uncertainty_threshold,
            use_concentration_threshold = use_concentration_threshold,
        )
        trial_summary_row = [
            experiment_name,
            absolute_trial_index + 1,
            terrain,
            constants.NUMBER_OF_AGENTS,
            threshold_type,
            constants.DS_DECISION_THRESHOLD,
            social_fusion_type,
            personal_fusion_rule,
            dm.DS_MASS_FUNCTION_MODE,
            consensus_method,
            communication_mode,
            constants.COMMUNICATION_RING_RANGE,
            constants.DS_THRESHOLD_MIN,
            constants.DS_THRESHOLD_MAX,
            dm.DS_TOP_K,
            dm.DS_MAX_NEIGHBORS,
            dm.DS_SINGLETON_WEIGHT,
            dm.DS_NEIGHBOR_COMPATIBILITY_GAIN,
            dm.DS_NEIGHBOR_SAME_TILE_SUPPORT,
            dm.DS_NEIGHBOR_INCOMPATIBILITY_PENALTY,
            dm.DS_NEIGHBOR_ADJACENT_TILE_SUPPORT,
            trial_summary["target_tile_index"],
            trial_summary["target_tile_name"],
            trial_summary["winning_tile_index"],
            trial_summary["winning_tile_name"],
            trial_summary["winning_tile_matches_target"],
            trial_summary["final_position_tile_index"],
            trial_summary["final_position_tile_name"],
            trial_summary["final_position_matches_target"],
            trial_summary["final_candidate_tiles"],
            trial_summary["target_in_final_candidate_tiles"],
            trial_summary["search_area_tile_count"],
            trial_summary["subarea_tile_size_square_meters"],
            trial_summary["number_of_different_final_decisions"],
            trial_summary["number_of_final_candidate_tiles"],
            trial_summary["stage1_mean_pairwise_decision_distance_meters"],
            trial_summary["stage1_number_of_connected_decision_clusters"],
            trial_summary["stage1_largest_cluster_share"],
            trial_summary["timeout_reached"],
            trial_summary["decided_agent_count"],
            trial_summary["first_decided_time"],
            trial_summary["half_decided_time"],
            trial_summary["all_decided_time"],
            trial_summary["late_stage_waiting_gap"],
            trial_summary["average_unique_received_opinions_per_agent_per_cycle"],
            trial_summary["final_position_x"],
            trial_summary["final_position_y"],
            trial_summary["consensus_std_x_meters"],
            trial_summary["consensus_std_y_meters"],
            trial_summary["consensus_mean_error_meters"],
            trial_summary["position_error_meters"],
            trial_summary["elapsed_time"],
        ]
        trial_summary_header = [
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
            "threshold_min",
            "threshold_max",
            "top_k",
            "max_neighbors",
            "singleton_weight",
            "compatibility_gain",
            "same_tile_support",
            "incompatibility_penalty",
            "neighbor_adjacent_tile_support",
            "target_tile_index",
            "target_tile_name",
            "winning_tile_index",
            "winning_tile_name",
            "winning_tile_matches_target",
            "final_position_tile_index",
            "final_position_tile_name",
            "final_position_matches_target",
            "final_candidate_tiles",
            "target_in_final_candidate_tiles",
            "search_area_tile_count",
            "subarea_tile_size_square_meters",
            "number_of_different_final_decisions",
            "number_of_final_candidate_tiles",
            "stage1_mean_pairwise_decision_distance_meters",
            "stage1_number_of_connected_decision_clusters",
            "stage1_largest_cluster_share",
            "timeout_reached",
            "decided_agent_count",
            "first_decided_time",
            "half_decided_time",
            "all_decided_time",
            "late_stage_waiting_gap",
            "average_unique_received_opinions_per_agent_per_cycle",
            "final_position_x",
            "final_position_y",
            "consensus_std_x_meters",
            "consensus_std_y_meters",
            "consensus_mean_error_meters",
            "position_error_meters",
            "consensus_time",
        ]
        utils.write_results_to_csv(results_paths["trial_summary"], trial_summary_row, header = trial_summary_header)

        for agent in agents:
            best_hypothesis, best_score = agent.get_best_decision_score()
            best_focal_set, best_focal_mass = agent.get_best_focal_set_and_mass()
            final_decision = agent.get_decision()
            received_opinion_counts = [snapshot["unique_received_opinion_count"] for snapshot in agent.decision_dynamics]
            average_unique_received_opinions = float(np.mean(received_opinion_counts, dtype = np.float32)) if received_opinion_counts else 0.0
            final_consensus_position = agent.get_consensus_position()
            agent_final_row = [
                experiment_name,
                absolute_trial_index + 1,
                terrain,
                constants.NUMBER_OF_AGENTS,
                threshold_type,
                constants.DS_DECISION_THRESHOLD,
                social_fusion_type,
                personal_fusion_rule,
                dm.DS_MASS_FUNCTION_MODE,
                consensus_method,
                communication_mode,
                constants.COMMUNICATION_RING_RANGE,
                agent.id,
                final_decision,
                "" if final_decision is None else convert_decision_index_to_subimage_name(final_decision, image_shape),
                best_hypothesis,
                "" if best_hypothesis is None else convert_decision_index_to_subimage_name(best_hypothesis, image_shape),
                best_score,
                agent.get_current_threshold(),
                dm.DS_TOP_K,
                dm.DS_MAX_NEIGHBORS,
                dm.DS_SINGLETON_WEIGHT,
                dm.DS_NEIGHBOR_COMPATIBILITY_GAIN,
                dm.DS_NEIGHBOR_SAME_TILE_SUPPORT,
                dm.DS_NEIGHBOR_INCOMPATIBILITY_PENALTY,
                dm.DS_NEIGHBOR_ADJACENT_TILE_SUPPORT,
                format_focal_set_label(best_focal_set, image_shape),
                best_focal_mass,
                agent.first_decision_cycle,
                agent.first_decision_time,
                None if final_consensus_position is None else float(final_consensus_position[0]),
                None if final_consensus_position is None else float(final_consensus_position[1]),
                agent.get_consensus_stable_counter(),
                agent.is_continuous_frozen(),
                agent.is_continuous_done(),
                average_unique_received_opinions,
            ]
            agent_final_header = [
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
                "agent_id",
                "final_decision_index",
                "final_decision_name",
                "best_hypothesis_index",
                "best_hypothesis_name",
                "best_decision_score",
                "final_threshold",
                "top_k",
                "max_neighbors",
                "singleton_weight",
                "compatibility_gain",
                "same_tile_support",
                "incompatibility_penalty",
                "neighbor_adjacent_tile_support",
                "best_focal_set",
                "best_focal_mass",
                "first_decision_cycle",
                "first_decision_time",
                "final_consensus_position_x",
                "final_consensus_position_y",
                "consensus_stable_counter",
                "continuous_frozen",
                "continuous_done",
                "average_unique_received_opinions_per_cycle",
            ]
            utils.write_results_to_csv(results_paths["agent_final_states"], agent_final_row, header = agent_final_header)

            for snapshot in agent.decision_dynamics:
                dynamics_row = [
                    experiment_name,
                    absolute_trial_index + 1,
                    terrain,
                    constants.NUMBER_OF_AGENTS,
                    threshold_type,
                    constants.DS_DECISION_THRESHOLD,
                    social_fusion_type,
                    personal_fusion_rule,
                    dm.DS_MASS_FUNCTION_MODE,
                    consensus_method,
                    communication_mode,
                    constants.COMMUNICATION_RING_RANGE,
                    dm.DS_TOP_K,
                    dm.DS_MAX_NEIGHBORS,
                    dm.DS_SINGLETON_WEIGHT,
                    dm.DS_NEIGHBOR_COMPATIBILITY_GAIN,
                    dm.DS_NEIGHBOR_SAME_TILE_SUPPORT,
                    dm.DS_NEIGHBOR_INCOMPATIBILITY_PENALTY,
                    dm.DS_NEIGHBOR_ADJACENT_TILE_SUPPORT,
                    agent.id,
                    snapshot["cycle_index"],
                    snapshot["elapsed_time"],
                    snapshot["best_hypothesis"],
                    "" if snapshot["best_hypothesis"] is None else convert_decision_index_to_subimage_name(snapshot["best_hypothesis"], image_shape),
                    snapshot["best_score"],
                    snapshot["threshold"],
                    snapshot["unique_received_opinion_count"],
                    snapshot["consensus_position_x"],
                    snapshot["consensus_position_y"],
                    snapshot["consensus_position_change"],
                    snapshot["consensus_confidence"],
                    snapshot["consensus_initialized"],
                    snapshot["continuous_frozen"],
                    snapshot["continuous_done"],
                    snapshot["decision"],
                    "" if snapshot["decision"] is None else convert_decision_index_to_subimage_name(snapshot["decision"], image_shape),
                    snapshot["decision"] is not None,
                ]
                if constants.LOG_DECISION_SIGNALS:
                    dynamics_row += [snapshot["uncertainty"], snapshot["second_score"], snapshot["rival_score"]]
                dynamics_header = [
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
                    "agent_id",
                    "cycle_index",
                    "elapsed_time",
                    "best_hypothesis_index",
                    "best_hypothesis_name",
                    "best_decision_score",
                    "threshold",
                    "unique_received_opinion_count",
                    "consensus_position_x",
                    "consensus_position_y",
                    "consensus_position_change",
                    "consensus_confidence",
                    "consensus_initialized",
                    "continuous_frozen",
                    "continuous_done",
                    "decision_index",
                    "decision_name",
                    "is_decided",
                ]
                if constants.LOG_DECISION_SIGNALS:
                    dynamics_header += ["uncertainty", "second_score", "rival_score"]
                utils.write_results_to_csv(results_paths["decision_dynamics"], dynamics_row, header = dynamics_header)

        for snapshot in consensus_dispersion_history:
            dispersion_row = [
                experiment_name,
                absolute_trial_index + 1,
                terrain,
                constants.NUMBER_OF_AGENTS,
                threshold_type,
                constants.DS_DECISION_THRESHOLD,
                social_fusion_type,
                personal_fusion_rule,
                dm.DS_MASS_FUNCTION_MODE,
                consensus_method,
                communication_mode,
                constants.COMMUNICATION_RING_RANGE,
                dm.DS_TOP_K,
                dm.DS_MAX_NEIGHBORS,
                dm.DS_SINGLETON_WEIGHT,
                dm.DS_NEIGHBOR_COMPATIBILITY_GAIN,
                dm.DS_NEIGHBOR_SAME_TILE_SUPPORT,
                dm.DS_NEIGHBOR_INCOMPATIBILITY_PENALTY,
                dm.DS_NEIGHBOR_ADJACENT_TILE_SUPPORT,
                snapshot["cycle_index"],
                snapshot["elapsed_time"],
                snapshot["num_agents_with_consensus"],
                snapshot["centroid_x"],
                snapshot["centroid_y"],
                snapshot["std_x_meters"],
                snapshot["std_y_meters"],
                snapshot["mean_distance_to_centroid_meters"],
                snapshot["std_distance_to_centroid_meters"],
                snapshot["average_pairwise_distance_meters"],
                snapshot["max_pairwise_distance_meters"],
            ]
            dispersion_header = [
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
                "cycle_index",
                "elapsed_time",
                "num_agents_with_consensus",
                "centroid_x",
                "centroid_y",
                "std_x_meters",
                "std_y_meters",
                "mean_distance_to_centroid_meters",
                "std_distance_to_centroid_meters",
                "average_pairwise_distance_meters",
                "max_pairwise_distance_meters",
            ]
            utils.write_results_to_csv(results_paths["consensus_dispersion"], dispersion_row, header = dispersion_header)

if __name__ == "__main__":
    experiment_name = sys.argv[1]
    raw_extra_args = sys.argv[2:]
    if raw_extra_args and raw_extra_args[0] == LEGACY_DS_METHOD_ALIAS:
        raw_extra_args = raw_extra_args[1:]

    invalid_flags = [
        argument
        for argument in raw_extra_args
        if argument.startswith("--")
        and argument not in TERRAIN_FLAGS
        and argument != "--visualize"
        and not argument.startswith("--social_fusion=")
        and not argument.startswith("--personal_fusion=")
        and not argument.startswith("--mass_function=")
        and not argument.startswith("--threshold_type=")
        and not argument.startswith("--trials=")
        and not argument.startswith("--swarm_size=")
        and not argument.startswith("--ds_decision_threshold=")
        and not argument.startswith("--ds_top_k=")
        and not argument.startswith("--ds_max_neighbors=")
        and not argument.startswith("--ds_singleton_weight=")
        and not argument.startswith("--ds_neighbor_compatibility_gain=")
        and not argument.startswith("--ds_neighbor_same_tile_support=")
        and not argument.startswith("--ds_neighbor_adjacent_tile_support=")
        and not argument.startswith("--ds_neighbor_incompatibility_penalty=")
        and not argument.startswith("--softmax_tau=")
        and argument != "--log_signals"
    ]
    if invalid_flags:
        print(
            "Unsupported CLI flags: "
            + ", ".join(invalid_flags)
            + ". Supported flags are terrain flags, --visualize, required --threshold_type=fixed|dynamic|uncertainty|concentration, "
            + "--social_fusion=comp_boost|comp_boost_directed|disabled|yager, --personal_fusion=adaptive|dempster|hybrid|yager, "
            + "optional --mass_function=spatial_topk|topk, "
            + "and optional --trials=, --swarm_size=, --ds_decision_threshold=, --ds_top_k=, --ds_max_neighbors=, "
            + "--ds_singleton_weight=, --ds_neighbor_compatibility_gain=, --ds_neighbor_same_tile_support=, "
            + "--ds_neighbor_adjacent_tile_support=, --ds_neighbor_incompatibility_penalty=, --softmax_tau=, --log_signals."
        )
        sys.exit(1)

    extra_args = set(raw_extra_args)
    visualize = "--visualize" in extra_args
    constants.LOG_DECISION_SIGNALS = "--log_signals" in extra_args
    try:
        threshold_type = parse_required_choice_argument(
            raw_extra_args,
            "--threshold_type=",
            ("dynamic", "fixed", "uncertainty", "concentration"),
        )
        social_fusion = parse_required_choice_argument(
            raw_extra_args,
            "--social_fusion=",
            ("comp_boost", "comp_boost_directed", "disabled", "yager"),
        )
        personal_fusion_rule = parse_required_choice_argument(
            raw_extra_args,
            "--personal_fusion=",
            ("adaptive", "dempster", "hybrid", "yager"),
        )
        mass_function_type = parse_choice_argument(
            raw_extra_args,
            "--mass_function=",
            ("spatial_topk", "topk"),
            dm.DS_MASS_FUNCTION_MODE,
        )
        trials = parse_optional_int_argument(raw_extra_args, "--trials=", default_value = NUMBER_OF_TRIALS, minimum = 1)
        swarm_size = parse_optional_int_argument(raw_extra_args, "--swarm_size=", default_value = constants.NUMBER_OF_AGENTS, minimum = 1)
        decision_threshold = parse_optional_float_argument(raw_extra_args, "--ds_decision_threshold=", default_value = constants.DS_DECISION_THRESHOLD, minimum = 0.0)
        if threshold_type in ("uncertainty", "concentration"):
            if not any(argument.startswith("--ds_decision_threshold=") for argument in raw_extra_args):
                raise ValueError(f"--threshold_type={threshold_type} requires --ds_decision_threshold=<value in (0, 1]>.")
            if not 0.0 < decision_threshold <= 1.0:
                raise ValueError(f"With --threshold_type={threshold_type}, --ds_decision_threshold must be in (0, 1].")
        ds_top_k = parse_optional_int_argument(raw_extra_args, "--ds_top_k=", default_value = dm.DS_TOP_K, minimum = 1)
        ds_max_neighbors = parse_optional_int_argument(raw_extra_args, "--ds_max_neighbors=", default_value = dm.DS_MAX_NEIGHBORS, minimum = 0)
        ds_singleton_weight = parse_optional_float_argument(raw_extra_args, "--ds_singleton_weight=", default_value = dm.DS_SINGLETON_WEIGHT, minimum = 0.0, maximum = 1.0)
        compatibility_gain = parse_optional_float_argument(raw_extra_args, "--ds_neighbor_compatibility_gain=", default_value = dm.DS_NEIGHBOR_COMPATIBILITY_GAIN, minimum = 0.0)
        same_tile_support = parse_optional_float_argument(raw_extra_args, "--ds_neighbor_same_tile_support=", default_value = dm.DS_NEIGHBOR_SAME_TILE_SUPPORT, minimum = 0.0, maximum = 1.0)
        adjacent_tile_support = parse_optional_float_argument(raw_extra_args, "--ds_neighbor_adjacent_tile_support=", default_value = dm.DS_NEIGHBOR_ADJACENT_TILE_SUPPORT, minimum = 0.0, maximum = 1.0)
        incompatibility_penalty = parse_optional_float_argument(raw_extra_args, "--ds_neighbor_incompatibility_penalty=", default_value = dm.DS_NEIGHBOR_INCOMPATIBILITY_PENALTY, minimum = 0.0, maximum = 1.0)
        constants.SOFTMAX_TAU = parse_optional_float_argument(raw_extra_args, "--softmax_tau=", default_value = constants.SOFTMAX_TAU, minimum = 1e-6)
    except ValueError as error:
        print(str(error))
        sys.exit(1)
    terrain_flags = [TERRAIN_FLAGS[arg] for arg in extra_args if arg in TERRAIN_FLAGS]
    terrain = terrain_flags[0] if terrain_flags else None

    if len(terrain_flags) > 1:
        print("Please choose only one terrain flag.")
        sys.exit(1)
    if terrain is None:
        print("Please choose one terrain flag.")
        sys.exit(1)

    constants.NUMBER_OF_AGENTS = int(swarm_size)
    constants.DS_DECISION_THRESHOLD = float(decision_threshold)
    dm.configure_dempster_shafer(
        top_k = ds_top_k,
        max_neighbors = ds_max_neighbors,
        mass_function_mode = mass_function_type,
        singleton_weight = ds_singleton_weight,
        compatibility_gain = compatibility_gain,
        same_tile_support = same_tile_support,
        incompatibility_penalty = incompatibility_penalty,
        neighbor_adjacent_tile_support = adjacent_tile_support,
    )

    main(
        experiment_name,
        terrain,
        personal_fusion_rule,
        threshold_type,
        social_fusion,
        visualize,
        trials = trials,
    )

    
