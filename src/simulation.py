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
CENTROIDS_PATH = "codebooks/trained_centroids.npy"
CODEBOOK_PATH = "codebooks/vlad_vectors.npy"
TERRAIN_FLAGS = {
    "--city": "city",
    "--desert": "desert",
    "--mountains": "mountains",
    "--rural": "rural",
}
MOVEMENT_FLAGS = {
    "--grid-search": constants.GRID_SEARCH_MOVEMENT,
}

AGENT_COLOR_NO_DECISION = (255, 0, 0)
AGENT_COLOR_CORRECT_DECISION = (0, 255, 0)
AGENT_COLOR_WRONG_DECISION = (0, 0, 255)

consensus_times = []
final_decisions = []
agents_data_locker = threading.Lock()

def calculate_initial_positions(image):
    """Calculate shared random initial positions of agents on the image.

    Args:
        image (numpy.ndarray): Original image.

    Returns:
        numpy.ndarray: Array of positions.
    """
    min_x = constants.AGENT_SIZE[0] / 2
    max_x = image.shape[1] - constants.AGENT_SIZE[0] / 2
    min_y = constants.AGENT_SIZE[1] / 2
    max_y = image.shape[0] - constants.AGENT_SIZE[1] / 2

    initial_position = np.array(
        [
            np.random.uniform(min_x, max_x),
            np.random.uniform(min_y, max_y),
        ]
    )

    return np.asarray([initial_position.copy() for _ in range(constants.NUMBER_OF_AGENTS)])

def calculate_grid_shape(count):
    """Choose an exact cell grid for the desired number of agents.

    Args:
        count (int): Number of agents/cells.

    Returns:
        tuple[int, int]: Number of rows and columns.
    """
    rows = int(np.floor(np.sqrt(count)))
    while rows > 1 and count % rows != 0:
        rows -= 1
    cols = count // rows

    return rows, cols

def calculate_search_cells(image):
    """Divide the full subimage into non-overlapping cells without border margins.

    Args:
        image (numpy.ndarray): Original image.

    Returns:
        list[tuple[float, float, float, float]]: Cell bounds as min x, max x, min y and max y.
    """
    rows, cols = calculate_grid_shape(constants.NUMBER_OF_AGENTS)
    image_height, image_width = image.shape[:2]
    x_edges = np.rint(np.linspace(0, image_width, cols + 1)).astype(int)
    y_edges = np.rint(np.linspace(0, image_height, rows + 1)).astype(int)
    x_edges[0], x_edges[-1] = 0, image_width
    y_edges[0], y_edges[-1] = 0, image_height

    cells = []
    for row in range(rows):
        for col in range(cols):
            cells.append(
                (
                    float(x_edges[col]),
                    float(x_edges[col + 1]),
                    float(y_edges[row]),
                    float(y_edges[row + 1]),
                )
            )

    return cells

def visualize_agent(agent, image, target_image_name, image_dir):
    """Visualize agent on the image.

    Args:
        agent (Agent): Agent to draw.
        image (numpy.ndarray): Copy of the original image.
        target_image_name (str): Name of the selected target image.
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
        color = AGENT_COLOR_CORRECT_DECISION if decided_image_name == target_image_name else AGENT_COLOR_WRONG_DECISION

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

def experiment(trial, image_code, dm_method, terrain = None, visualize = False, movement_type = constants.RANDOM_MOVEMENT):
    """Single run of the experiment.

    Args:
        test_image (string): Name of the testing image from subimages folder or 'r' for random.
    """
    image_dir = get_image_dir_for_terrain(terrain)
    codebook_path = get_codebook_path_for_terrain(terrain)

    if image_code == 'r':
        test_image, test_image_name = utils.select_random_image(image_dir)
    else:
        test_image = cv2.imread(image_dir + image_code + '.png')
        test_image_name = image_code 

    print(f'TRIAL: {trial + 1}, SELECTED IMAGE: {test_image_name}')

    centroids = np.load(CENTROIDS_PATH)
    codebook = np.load(codebook_path)

    positions = calculate_initial_positions(test_image)
    search_cells = calculate_search_cells(test_image) if movement_type == constants.GRID_SEARCH_MOVEMENT else [None] * len(positions)
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

        decisions = []
        for agent in agents:
            if visualize:
                visualize_agent(agent, image_copy, test_image_name, image_dir)
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
            return test_image_name, decision, elapsed_time_rescaled

        if visualize:
            cv2.imshow(f'{test_image_name}', image_copy)
            cv2.waitKey(1)

        elapsed_time = time.time() - start_time
        utils.sleep_with_adjustable_period(SIMULATION_LOOP_PERIOD / SIMULATION_SPEED_UP_FACTOR, loop_start_time)

def main(image_code, results_filename, dm_method, terrain = None, visualize = False, movement_type = constants.RANDOM_MOVEMENT):
    """Main experiment entry.

    Args:
        image_code (string): Name of the original image or 'r' for randomly selected image.
    """
    for trial in range(NUMBER_OF_TRIALS):
        image, decision, elapsed_time = experiment(trial, image_code, dm_method, terrain, visualize, movement_type)
        data_to_write = [image, decision, elapsed_time, dm_method, terrain, movement_type]
        utils.write_results_to_csv(results_filename, data_to_write)

if __name__ == "__main__":
    image_code = sys.argv[1]
    results_filename = sys.argv[2]
    dm_method = sys.argv[3]
    extra_args = set(sys.argv[4:])
    visualize = "--visualize" in extra_args
    terrain_flags = [TERRAIN_FLAGS[arg] for arg in extra_args if arg in TERRAIN_FLAGS]
    terrain = terrain_flags[0] if terrain_flags else None
    movement_flags = [MOVEMENT_FLAGS[arg] for arg in extra_args if arg in MOVEMENT_FLAGS]
    movement_type = movement_flags[0] if movement_flags else constants.RANDOM_MOVEMENT

    if len(terrain_flags) > 1:
        print("Please choose only one terrain flag.")
        sys.exit(1)
    if len(movement_flags) > 1:
        print("Please choose only one movement flag.")
        sys.exit(1)

    if dm_method not in (constants.ASODM_METHOD, constants.DEMPSER_SHAFER_METHOD):
        print(f'Invalid decision making method: {dm_method}. Please choose either "{constants.ASODM_METHOD}" or "{constants.DEMPSER_SHAFER_METHOD}".')
        sys.exit(1)

    main(image_code, results_filename, dm_method, terrain, visualize, movement_type)

    
