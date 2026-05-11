from collections import Counter
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import threading
import time

from agent import Agent
import constants
import image_augmentation as ia
import utils

NUMBER_OF_TRIALS = 250

SIMULATION_SPEED_UP_FACTOR = 1
SIMULATION_LOOP_PERIOD = 0.01

IMAGE_DIR = "data/subimages/"
ORIGINAL_IMAGE_DIR = "data/original_images/"
CENTROIDS_PATH = "codebooks/trained_centroids.npy"
CODEBOOK_PATH = "codebooks/vlad_vectors.npy"
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

consensus_times = []
final_decisions = []
agents_data_locker = threading.Lock()

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

def calculate_initial_positions(image, initial_position = None):
    """Calculate shared random initial positions of agents on the image.

    Args:
        image (numpy.ndarray): Original image.
        initial_position (numpy.ndarray, optional): Shared initial position for all agents.

    Returns:
        numpy.ndarray: Array of positions.
    """
    if initial_position is None:
        initial_position = calculate_random_position(image)

    return np.asarray([initial_position.copy() for _ in range(constants.NUMBER_OF_AGENTS)])

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

def check_for_consenus(decisions, agents):
    """Check if agents have reached a consensus.

    Args:
        decisions (list): List of all agents decisions.
        agents (list): List of all agents.

    Returns:
        bool: True if consensus has been reached, False otherwise.
    """
    if len(decisions) != len(agents) or decisions[0] is None:
        return False
    return decisions.count(decisions[0]) == len(decisions)

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

def experiment(trial, dm_method, terrain, visualize = False):
    """Single run of the experiment.

    Args:
        dm_method (str): Decision making method.
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

    positions = calculate_initial_positions(test_image, initial_position)
    movement_type = constants.GRID_SEARCH_MOVEMENT
    search_cells, search_area_bounds = calculate_search_cells(test_image, initial_position)
    agents = [
        Agent(test_image, centroids, codebook, positions[i], str(i), dm_method, SIMULATION_SPEED_UP_FACTOR, movement_type, search_cells[i])
        for i in range(len(positions))
    ]

    start_time = time.time()
    elapsed_time = 0
    max_experiment_time = constants.MAX_EXPERIMENT_TIME / SIMULATION_SPEED_UP_FACTOR

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

        if check_for_consenus(decisions, agents) or elapsed_time > max_experiment_time:
            consensus_times.append(elapsed_time)
            max_opinions = []
            for agent in agents:
                max_opinions.append(agent.max_opinions)
                agent.stop_threads()

            decision = utils.convert_decision_to_subimage_name(decisions[0], image_dir) if elapsed_time < max_experiment_time else 'NO DECISION'

            elapsed_time_rescaled = elapsed_time * SIMULATION_SPEED_UP_FACTOR
            print(f'FINAL DECISION: {decision}, ELAPSED TIME: {elapsed_time_rescaled}')

            if visualize:
                cv2.destroyAllWindows()
            return target_subimage_name, decision, elapsed_time_rescaled

        if visualize:
            cv2.imshow(f'{terrain}', image_copy)
            cv2.waitKey(1)

        elapsed_time = time.time() - start_time
        utils.sleep_with_adjustable_period(SIMULATION_LOOP_PERIOD / SIMULATION_SPEED_UP_FACTOR, loop_start_time)

def main(results_filename, dm_method, terrain, visualize = False):
    """Main experiment entry.

    Args:
        results_filename (str): Output csv file.
    """
    sqrt_agent_count = int(np.sqrt(constants.NUMBER_OF_AGENTS))
    is_perfect_square = sqrt_agent_count * sqrt_agent_count == constants.NUMBER_OF_AGENTS
    if constants.NUMBER_OF_AGENTS % 2 != 0 and not is_perfect_square:
        raise ValueError("Grid-search requires an even number of agents or a perfect square swarm size.")

    for trial in range(NUMBER_OF_TRIALS):
        image, decision, elapsed_time = experiment(trial, dm_method, terrain, visualize)
        movement_type = constants.GRID_SEARCH_MOVEMENT
        data_to_write = [image, decision, elapsed_time, dm_method, terrain, movement_type]
        utils.write_results_to_csv(results_filename, data_to_write)

if __name__ == "__main__":
    results_filename = sys.argv[1]
    dm_method = sys.argv[2]
    extra_args = set(sys.argv[3:])
    visualize = "--visualize" in extra_args
    terrain_flags = [TERRAIN_FLAGS[arg] for arg in extra_args if arg in TERRAIN_FLAGS]
    terrain = terrain_flags[0] if terrain_flags else None

    if len(terrain_flags) > 1:
        print("Please choose only one terrain flag.")
        sys.exit(1)
    if terrain is None:
        print("Please choose one terrain flag.")
        sys.exit(1)

    if dm_method not in (constants.ASODM_METHOD, constants.DEMPSER_SHAFER_METHOD):
        print(f'Invalid decision making method: {dm_method}. Please choose either "{constants.ASODM_METHOD}" or "{constants.DEMPSER_SHAFER_METHOD}".')
        sys.exit(1)

    main(results_filename, dm_method, terrain, visualize)

    
