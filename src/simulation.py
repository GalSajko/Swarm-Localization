import cv2
import numpy as np
import os
import sys
import time

from agent import Agent
import constants
import decision_making as dm
import utils

NUMBER_OF_TRIALS = 100

SIMULATION_SPEED_UP_FACTOR = 2.0
SIMULATION_LOOP_PERIOD = 0.01

IMAGE_DIR = "data/subimages/"
ORIGINAL_IMAGE_DIR = "data/original_images/"
CENTROIDS_PATH = "codebooks/trained_centroids.npy"
CODEBOOK_PATH = "codebooks/vlad_vectors.npy"
RESULTS_DIR = "results/continuous_est"
SUBIMAGE_GRID_SIZE = (20, 20)
TERRAIN_IMAGE_NAMES = {
    "city": "city_bw.png",
    "desert": "desert_bw.png",
    "mountains": "mountains_bw.png",
    "rural": "rural_bw.png",
}
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
METERS_PER_PIXEL = 2.0

consensus_times = []

def parse_optional_argument(raw_extra_args, prefix, cast = str):
    """Extract one optional CLI argument of the form --name=value."""
    for argument in raw_extra_args:
        if argument.startswith(prefix):
            return cast(argument.split("=", 1)[1])
    return None

def build_results_paths(experiment_name):
    """Build output CSV paths inside the appropriate results/continuous_est category."""
    experiment_stem = os.path.splitext(os.path.basename(experiment_name))[0]
    if experiment_stem.startswith("swarm_size"):
        category_dir = "swarm_size"
    elif experiment_stem.startswith("decision_threshold") or experiment_stem.startswith("threshold_"):
        category_dir = "decision_threshold"
    elif experiment_stem.startswith("social_fusion") or experiment_stem.startswith("compatibility_gain"):
        category_dir = "social_fusion"
    else:
        category_dir = "top_k_ds"

    experiment_dir = os.path.join(RESULTS_DIR, category_dir, experiment_stem)
    return {
        "trial_summary": os.path.join(experiment_dir, "trial_summary.csv"),
        "agent_final_states": os.path.join(experiment_dir, "agent_final_states.csv"),
        "decision_dynamics": os.path.join(experiment_dir, "decision_dynamics.csv"),
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
    subimage_width = image_width / SUBIMAGE_GRID_SIZE[0]
    subimage_height = image_height / SUBIMAGE_GRID_SIZE[1]

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

def calculate_search_area_tile_count(search_area_bounds, image_shape):
    """Count how many subarea tiles intersect the rectangular search area."""
    min_x, max_x, min_y, max_y = search_area_bounds
    image_height, image_width = image_shape[:2]
    subimage_width = image_width / SUBIMAGE_GRID_SIZE[0]
    subimage_height = image_height / SUBIMAGE_GRID_SIZE[1]

    covered_tiles = 0
    for tile_x in range(SUBIMAGE_GRID_SIZE[0]):
        tile_min_x = tile_x * subimage_width
        tile_max_x = (tile_x + 1) * subimage_width
        overlaps_x = tile_min_x < max_x and tile_max_x > min_x
        if not overlaps_x:
            continue

        for tile_y in range(SUBIMAGE_GRID_SIZE[1]):
            tile_min_y = tile_y * subimage_height
            tile_max_y = (tile_y + 1) * subimage_height
            overlaps_y = tile_min_y < max_y and tile_max_y > min_y
            if overlaps_y:
                covered_tiles += 1

    return covered_tiles

def calculate_subarea_tile_size_square_meters(image_shape):
    """Calculate one terrain subarea tile area in square meters."""
    image_height, image_width = image_shape[:2]
    subimage_width_pixels = image_width / SUBIMAGE_GRID_SIZE[0]
    subimage_height_pixels = image_height / SUBIMAGE_GRID_SIZE[1]
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

    for col in range(1, SUBIMAGE_GRID_SIZE[0]):
        x = int(round(col * image_width / SUBIMAGE_GRID_SIZE[0]))
        cv2.line(image, (x, 0), (x, image_height), SUBAREA_GRID_SHADOW_COLOR, 3, cv2.LINE_AA)
        cv2.line(image, (x, 0), (x, image_height), SUBAREA_GRID_COLOR, 1, cv2.LINE_AA)

    for row in range(1, SUBIMAGE_GRID_SIZE[1]):
        y = int(round(row * image_height / SUBIMAGE_GRID_SIZE[1]))
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
    # cv2.circle(image, position.astype(np.int32), constants.COMMUNICATION_RANGE, (255, 255, 0), 2)

def convert_decision_to_tile_bounds(decision, image_shape):
    """Convert a decision index to the full-image bounds of the selected tile."""
    subimage_width = image_shape[1] / SUBIMAGE_GRID_SIZE[0]
    subimage_height = image_shape[0] / SUBIMAGE_GRID_SIZE[1]

    rows = SUBIMAGE_GRID_SIZE[1]
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

def convert_decision_to_tile_indices(decision):
    """Convert a decision index to x/y tile indices in the terrain grid."""
    rows = SUBIMAGE_GRID_SIZE[1]
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

def calculate_final_weighted_position(decision_scores, image_shape):
    """Calculate a centralized weighted-average final position from decision-tile scores."""
    if not decision_scores:
        return None

    total_weight = sum(float(score) for score in decision_scores.values())
    if total_weight <= 0:
        return None

    weighted_position = np.zeros(2, dtype = np.float32)
    for decision, score in decision_scores.items():
        weighted_position += float(score) * convert_decision_to_tile_center(decision, image_shape)

    return weighted_position / total_weight

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

def build_final_winner_overlay(image, agents, initial_position = None):
    """Build the final visualization frame with decision tiles, agents and markers."""
    overlay_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    decision_tiles = collect_final_decision_tiles(agents)
    decision_scores = calculate_final_decision_scores(agents, decision_tiles)
    final_position = calculate_final_weighted_position(decision_scores, image.shape)
    decision_colors = get_final_decision_colors(decision_scores)
    visualize_final_decision_tiles(overlay_image, decision_tiles, decision_colors)
    visualize_final_agents(overlay_image, agents, decision_colors)
    if initial_position is not None:
        visualize_start_position_cross(overlay_image, initial_position)
    if final_position is not None:
        visualize_final_position_cross(overlay_image, final_position)

    return overlay_image

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

def check_for_termination(agents, use_fixed_threshold = False):
    """Check if every agent's best decision score has crossed threshold."""
    for agent in agents:
        best_hypothesis, best_score = agent.get_best_decision_score()
        threshold = agent.decision_threshold if use_fixed_threshold else agent.get_dynamic_threshold()
        if best_hypothesis is None or best_score < threshold:
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

def get_image_dir_for_terrain(terrain):
    """Return the subimage directory for the selected terrain.

    Args:
        terrain (str): Selected terrain or None.

    Returns:
        str: Directory containing subimages.
    """
    if terrain is None:
        return IMAGE_DIR
    return os.path.join(IMAGE_DIR, terrain) + os.sep

def get_codebook_path_for_terrain(terrain):
    """Return the VLAD codebook path for the selected terrain.

    Args:
        terrain (str): Selected terrain or None.

    Returns:
        str: Path to the VLAD vectors file.
    """
    if terrain is None:
        return CODEBOOK_PATH
    return os.path.join("codebooks", f"vlad_vectors_{terrain}.npy")

def get_original_image_path_for_terrain(terrain):
    """Return the full terrain image path for the selected terrain.

    Args:
        terrain (str): Selected terrain.

    Returns:
        str: Path to the terrain image.
    """
    return os.path.join(ORIGINAL_IMAGE_DIR, TERRAIN_IMAGE_NAMES[terrain])

def convert_position_to_subimage_name(position, image_shape):
    """Map a full-image position to the corresponding generated subimage name.

    Args:
        position (numpy.ndarray): Agent center position in the full terrain image.
        image_shape (tuple[int, ...]): Terrain image shape.

    Returns:
        str: Subimage name in the same format as generated dataset tiles.
    """
    subimage_width = image_shape[1] // SUBIMAGE_GRID_SIZE[0]
    subimage_height = image_shape[0] // SUBIMAGE_GRID_SIZE[1]

    x_index = min(int(position[0] // subimage_width), SUBIMAGE_GRID_SIZE[0] - 1)
    y_index = min(int(position[1] // subimage_height), SUBIMAGE_GRID_SIZE[1] - 1)

    return f"{x_index + 1:02d}_{y_index + 1:02d}"

def convert_decision_index_to_subimage_name(decision, image_shape):
    """Convert a decision index to the corresponding subimage name."""
    subimage_width = image_shape[1] // SUBIMAGE_GRID_SIZE[0]
    subimage_height = image_shape[0] // SUBIMAGE_GRID_SIZE[1]
    x_index = min(int((decision % (SUBIMAGE_GRID_SIZE[1] * SUBIMAGE_GRID_SIZE[0])) // SUBIMAGE_GRID_SIZE[1]), SUBIMAGE_GRID_SIZE[0] - 1)
    y_index = min(int(decision % SUBIMAGE_GRID_SIZE[1]), SUBIMAGE_GRID_SIZE[1] - 1)
    if subimage_width <= 0 or subimage_height <= 0:
        return f"{x_index + 1:02d}_{y_index + 1:02d}"
    return f"{x_index + 1:02d}_{y_index + 1:02d}"

def format_final_decision_scores(decision_scores, image_shape):
    """Format final decision-tile scores for terminal output."""
    formatted_scores = []
    for decision, score in sorted(decision_scores.items(), key = lambda item: item[1], reverse = True):
        formatted_scores.append(
            f"{convert_decision_index_to_subimage_name(decision, image_shape)}:{score:.6f}"
        )
    return formatted_scores

def experiment(
    trial,
    terrain,
    visualize = False,
    use_fixed_threshold = False,
    use_yager_social_fusion = False,
    threshold_value = None,
):
    """Single run of the experiment.

    Args:
        terrain (str): Selected terrain.
    """
    image_dir = get_image_dir_for_terrain(terrain)
    codebook_path = get_codebook_path_for_terrain(terrain)
    test_image = cv2.imread(get_original_image_path_for_terrain(terrain), cv2.IMREAD_GRAYSCALE)
    initial_position = calculate_random_position(test_image)
    target_subimage_name = convert_position_to_subimage_name(initial_position, test_image.shape)

    print(
        f"TRIAL: {trial + 1}, START POSITION: "
        f"({initial_position[0]:.2f}, {initial_position[1]:.2f})"
    )

    centroids = np.load(CENTROIDS_PATH)
    codebook = np.load(codebook_path)

    search_cells, search_area_bounds = calculate_search_cells(test_image, initial_position)
    search_area_tile_count = calculate_search_area_tile_count(search_area_bounds, test_image.shape)
    subarea_tile_size_square_meters = calculate_subarea_tile_size_square_meters(test_image.shape)
    positions = calculate_initial_positions_from_search_cells(search_cells)
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
            use_yager_social_fusion = use_yager_social_fusion,
            threshold_value = threshold_value,
            trial_start_time = trial_start_time,
        )
        for i in range(len(positions))
    ]

    start_time = trial_start_time
    elapsed_time = 0
    max_experiment_time = constants.MAX_EXPERIMENT_TIME / SIMULATION_SPEED_UP_FACTOR
    final_overlay_image = None

    while True:
        loop_start_time = time.time()
        image_copy = np.copy(test_image) if visualize else None
        if visualize:
            visualize_subarea_grid(image_copy)
            visualize_search_area(image_copy, search_area_bounds)
            visualize_start_position(image_copy, initial_position)

        decisions = []
        for agent in agents:
            if visualize:
                visualize_agent(agent, image_copy, target_subimage_name, image_dir)
            if agent.get_state() == constants.RECEIVING_STATE:
                agent.collect_opinions(agents)
            
            decisions.append(agent.get_decision())

        if check_for_termination(agents, use_fixed_threshold) or elapsed_time > max_experiment_time:
            consensus_times.append(elapsed_time)
            timeout_reached = elapsed_time > max_experiment_time
            decisions = []
            for agent in agents:
                best_hypothesis, _ = agent.get_best_decision_score()
                agent.set_decision(best_hypothesis)
                decisions.append(best_hypothesis)
            for agent in agents:
                agent.stop_threads()

            final_overlay_image = build_final_winner_overlay(
                test_image,
                agents,
                initial_position,
            )
            final_decision_tiles = collect_final_decision_tiles(agents)
            final_decision_scores = calculate_final_decision_scores(agents, final_decision_tiles)
            final_position = calculate_final_weighted_position(final_decision_scores, test_image.shape)
            position_error_meters = calculate_position_error_meters(initial_position, final_position)
            number_of_different_final_decisions = len({decision for decision in decisions if decision is not None})
            number_of_final_candidate_tiles = len(final_decision_tiles)
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

            elapsed_time_rescaled = elapsed_time * SIMULATION_SPEED_UP_FACTOR
            print(f'FINAL DECISION: {decision}, ELAPSED TIME: {elapsed_time_rescaled}')
            print(f"FINAL DECISION SCORES: {format_final_decision_scores(final_decision_scores, test_image.shape)}")
            print(f"FINAL POSITION ERROR [m]: {position_error_meters}")

            if visualize:
                cv2.imshow(f'{terrain}', final_overlay_image)
                wait_for_next_trial_key(f'{terrain}', final_overlay_image)
                cv2.destroyAllWindows()
            trial_summary = {
                "elapsed_time": elapsed_time_rescaled,
                "position_error_meters": position_error_meters,
                "search_area_tile_count": search_area_tile_count,
                "subarea_tile_size_square_meters": subarea_tile_size_square_meters,
                "number_of_different_final_decisions": number_of_different_final_decisions,
                "number_of_final_candidate_tiles": number_of_final_candidate_tiles,
                "timeout_reached": timeout_reached,
                "first_decided_time": first_decided_time,
                "half_decided_time": half_decided_time,
                "all_decided_time": all_decided_time,
                "late_stage_waiting_gap": late_stage_waiting_gap,
                "decided_agent_count": decided_agent_count,
                "average_unique_received_opinions_per_agent_per_cycle": average_unique_received_opinions_per_agent_per_cycle,
            }
            return trial_summary, agents, test_image.shape

        if visualize:
            cv2.imshow(f'{terrain}', image_copy)
            cv2.waitKey(1)

        elapsed_time = time.time() - start_time
        utils.sleep_with_adjustable_period(SIMULATION_LOOP_PERIOD / SIMULATION_SPEED_UP_FACTOR, loop_start_time)

def main(
    experiment_name,
    terrain,
    visualize = False,
    use_fixed_threshold = False,
    use_yager_social_fusion = False,
    num_trials = NUMBER_OF_TRIALS,
    num_agents = None,
    threshold_value = None,
    threshold_min = None,
    threshold_max = None,
    top_k = None,
    singleton_weight = None,
    compatibility_gain = None,
    communication_range = None,
):
    """Main experiment entry.

    Args:
        experiment_name (str): Base name used for results files inside results/continuous_est.
    """
    if num_agents is not None:
        constants.NUMBER_OF_AGENTS = int(num_agents)
    if threshold_value is not None:
        constants.DS_DECISION_THRESHOLD = float(threshold_value)
    if threshold_min is not None:
        constants.DS_THRESHOLD_MIN = float(threshold_min)
    if threshold_max is not None:
        constants.DS_THRESHOLD_MAX = float(threshold_max)
    if communication_range is not None:
        constants.COMMUNICATION_RANGE = float(communication_range)
    if threshold_value is not None:
        constants.DS_THRESHOLD_MIN = min(constants.DS_THRESHOLD_MIN, constants.DS_DECISION_THRESHOLD)
        constants.DS_THRESHOLD_MAX = max(constants.DS_THRESHOLD_MAX, constants.DS_DECISION_THRESHOLD)

    dm.configure_dempster_shafer(
        top_k = top_k,
        singleton_weight = singleton_weight,
        compatibility_gain = compatibility_gain,
    )

    sqrt_agent_count = int(np.sqrt(constants.NUMBER_OF_AGENTS))
    is_perfect_square = sqrt_agent_count * sqrt_agent_count == constants.NUMBER_OF_AGENTS
    if constants.NUMBER_OF_AGENTS % 2 != 0 and not is_perfect_square:
        raise ValueError("Grid-search requires an even number of agents or a perfect square swarm size.")

    results_paths = build_results_paths(experiment_name)
    threshold_type = constants.DS_DECISION_THRESHOLD if use_fixed_threshold else "dynamic"
    social_fusion_type = "yager" if use_yager_social_fusion else "compatibility_boost"

    for trial in range(num_trials):
        trial_summary, agents, image_shape = experiment(
            trial,
            terrain,
            visualize,
            use_fixed_threshold,
            use_yager_social_fusion,
            threshold_value,
        )
        trial_summary_row = [
            experiment_name,
            trial + 1,
            terrain,
            constants.NUMBER_OF_AGENTS,
            threshold_type,
            constants.DS_DECISION_THRESHOLD,
            social_fusion_type,
            constants.COMMUNICATION_RANGE,
            constants.DS_THRESHOLD_MIN,
            constants.DS_THRESHOLD_MAX,
            dm.DS_TOP_K,
            dm.DS_SINGLETON_WEIGHT,
            dm.DS_NEIGHBOR_COMPATIBILITY_GAIN,
            trial_summary["search_area_tile_count"],
            trial_summary["subarea_tile_size_square_meters"],
            trial_summary["number_of_different_final_decisions"],
            trial_summary["number_of_final_candidate_tiles"],
            trial_summary["timeout_reached"],
            trial_summary["decided_agent_count"],
            trial_summary["first_decided_time"],
            trial_summary["half_decided_time"],
            trial_summary["all_decided_time"],
            trial_summary["late_stage_waiting_gap"],
            trial_summary["average_unique_received_opinions_per_agent_per_cycle"],
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
            "communication_range",
            "threshold_min",
            "threshold_max",
            "top_k",
            "singleton_weight",
            "compatibility_gain",
            "search_area_tile_count",
            "subarea_tile_size_square_meters",
            "number_of_different_final_decisions",
            "number_of_final_candidate_tiles",
            "timeout_reached",
            "decided_agent_count",
            "first_decided_time",
            "half_decided_time",
            "all_decided_time",
            "late_stage_waiting_gap",
            "average_unique_received_opinions_per_agent_per_cycle",
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
            agent_final_row = [
                experiment_name,
                trial + 1,
                terrain,
                constants.NUMBER_OF_AGENTS,
                threshold_type,
                constants.DS_DECISION_THRESHOLD,
                social_fusion_type,
                constants.COMMUNICATION_RANGE,
                agent.id,
                final_decision,
                "" if final_decision is None else convert_decision_index_to_subimage_name(final_decision, image_shape),
                best_hypothesis,
                "" if best_hypothesis is None else convert_decision_index_to_subimage_name(best_hypothesis, image_shape),
                best_score,
                agent.get_current_threshold(),
                format_focal_set_label(best_focal_set, image_shape),
                best_focal_mass,
                agent.first_decision_cycle,
                agent.first_decision_time,
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
                "communication_range",
                "agent_id",
                "final_decision_index",
                "final_decision_name",
                "best_hypothesis_index",
                "best_hypothesis_name",
                "best_decision_score",
                "final_threshold",
                "best_focal_set",
                "best_focal_mass",
                "first_decision_cycle",
                "first_decision_time",
                "average_unique_received_opinions_per_cycle",
            ]
            utils.write_results_to_csv(results_paths["agent_final_states"], agent_final_row, header = agent_final_header)

            for snapshot in agent.decision_dynamics:
                dynamics_row = [
                    experiment_name,
                    trial + 1,
                    terrain,
                    constants.NUMBER_OF_AGENTS,
                    threshold_type,
                    constants.DS_DECISION_THRESHOLD,
                    social_fusion_type,
                    constants.COMMUNICATION_RANGE,
                    dm.DS_TOP_K,
                    dm.DS_SINGLETON_WEIGHT,
                    dm.DS_NEIGHBOR_COMPATIBILITY_GAIN,
                    agent.id,
                    snapshot["cycle_index"],
                    snapshot["elapsed_time"],
                    snapshot["best_hypothesis"],
                    "" if snapshot["best_hypothesis"] is None else convert_decision_index_to_subimage_name(snapshot["best_hypothesis"], image_shape),
                    snapshot["best_score"],
                    snapshot["threshold"],
                    snapshot["unique_received_opinion_count"],
                    snapshot["decision"],
                    "" if snapshot["decision"] is None else convert_decision_index_to_subimage_name(snapshot["decision"], image_shape),
                    snapshot["decision"] is not None,
                ]
                dynamics_header = [
                    "experiment_name",
                    "trial_index",
                    "terrain",
                    "number_of_agents",
                    "threshold_type",
                    "threshold_value",
                    "social_fusion_type",
                    "communication_range",
                    "top_k",
                    "singleton_weight",
                    "compatibility_gain",
                    "agent_id",
                    "cycle_index",
                    "elapsed_time",
                    "best_hypothesis_index",
                    "best_hypothesis_name",
                    "best_decision_score",
                    "threshold",
                    "unique_received_opinion_count",
                    "decision_index",
                    "decision_name",
                    "is_decided",
                ]
                utils.write_results_to_csv(results_paths["decision_dynamics"], dynamics_row, header = dynamics_header)

if __name__ == "__main__":
    experiment_name = sys.argv[1]
    raw_extra_args = sys.argv[2:]
    if raw_extra_args and raw_extra_args[0] == constants.DEMPSER_SHAFER_METHOD:
        raw_extra_args = raw_extra_args[1:]

    extra_args = set(raw_extra_args)
    visualize = "--visualize" in extra_args
    use_fixed_threshold = "--fixed_threshold" in extra_args
    use_yager_social_fusion = "--yager" in extra_args
    num_trials = parse_optional_argument(raw_extra_args, "--trials=", int)
    num_agents = parse_optional_argument(raw_extra_args, "--num_agents=", int)
    threshold_value = parse_optional_argument(raw_extra_args, "--threshold_value=", float)
    threshold_min = parse_optional_argument(raw_extra_args, "--threshold_min=", float)
    threshold_max = parse_optional_argument(raw_extra_args, "--threshold_max=", float)
    top_k = parse_optional_argument(raw_extra_args, "--top_k=", int)
    singleton_weight = parse_optional_argument(raw_extra_args, "--singleton_weight=", float)
    compatibility_gain = parse_optional_argument(raw_extra_args, "--compatibility_gain=", float)
    communication_range = parse_optional_argument(raw_extra_args, "--communication_range=", float)
    terrain_flags = [TERRAIN_FLAGS[arg] for arg in extra_args if arg in TERRAIN_FLAGS]
    terrain = terrain_flags[0] if terrain_flags else None

    if len(terrain_flags) > 1:
        print("Please choose only one terrain flag.")
        sys.exit(1)
    if terrain is None:
        print("Please choose one terrain flag.")
        sys.exit(1)

    main(
        experiment_name,
        terrain,
        visualize,
        use_fixed_threshold,
        use_yager_social_fusion,
        NUMBER_OF_TRIALS if num_trials is None else num_trials,
        num_agents,
        threshold_value,
        threshold_min,
        threshold_max,
        top_k,
        singleton_weight,
        compatibility_gain,
        communication_range,
    )

    
