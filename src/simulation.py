from collections import Counter
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import threading
import time

from agent import Agent
import constants
import image_augmentation as ia
import utils

NUMBER_OF_TRIALS = 20
IMAGE_DIR = "data/subimages/"
CENTROIDS_PATH = "codebooks/trained_centroids.npy"
CODEBOOK_PATH = "codebooks/vlad_vectors.npy"

AGENT_COLOR_NO_DECISION = (0, 0, 255)
AGENT_COLOR_DECISION = (0, 255, 0)

consensus_times = []
final_decisions = []
agents_data_locker = threading.Lock()

def calculate_initial_positions(image):
    """Calculate initial positions of agents on the image.

    Args:
        image (numpy.ndarray): Original image.

    Returns:
        numpy.ndarray: Array of positions.
    """
    number_in_x = 3
    number_in_y = 3

    initial_padding = [image.shape[1] - constants.AGENT_SIZE[0], image.shape[0] - constants.AGENT_SIZE[1]]

    x_positions = np.linspace(initial_padding[0], image.shape[1] - initial_padding[0], number_in_x)
    y_positions = np.linspace(initial_padding[1], image.shape[0] - initial_padding[1], number_in_y)

    positions = []

    for x in x_positions:
        for y in y_positions:
            positions.append(np.array([x, y]))
    
    return np.asarray(positions)

def visualize_agent(agent, image):
    """Visualize agent on the image.

    Args:
        position (numpy.ndarray): Agent's position.
        image (numpy.ndarray): Copy of the original image.
    """
    position = agent.get_position()
    start_point = np.array(position - constants.AGENT_SIZE / 2).astype(np.int32)
    end_point = np.array(position + constants.AGENT_SIZE / 2).astype(np.int32)

    color = AGENT_COLOR_NO_DECISION if agent.get_decision() is None else AGENT_COLOR_DECISION
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

def experiment(trial, image_code, dm_method):
    """Single run of the experiment.

    Args:
        test_image (string): Name of the testing image from subimages folder or 'r' for random.
    """
    if image_code == 'r':
        test_image, test_image_name = utils.select_random_image(IMAGE_DIR)
    else:
        test_image = cv2.imread(IMAGE_DIR + image_code + '.png')
        test_image_name = image_code 
    
    number_of_different_opinions = []

    print(f'TRIAL: {trial + 1}, SELECTED IMAGE: {test_image_name}')

    centroids = np.load(CENTROIDS_PATH)
    codebook = np.load(CODEBOOK_PATH)

    positions = calculate_initial_positions(test_image)
    agents = [Agent(test_image, centroids, codebook, positions[i], str(i), dm_method) for i in range(len(positions))]

    start_time = time.time()
    elapsed_time = 0

    previous_decisions = [-1] * len(agents)
    while True:
        image_copy = np.copy(test_image)

        decisions = []
        for agent in agents:
            visualize_agent(agent, image_copy)
            if agent.get_state() == constants.RECEIVING_STATE:
                agent.collect_opinions(agents)
            
            decisions.append(agent.get_decision())
        
        # if decisions != previous_decisions:
        #     print(decisions)
        # previous_decisions = decisions.copy()

        if check_for_consenus(decisions, agents) or elapsed_time > constants.MAX_EXPERIMENT_TIME:
            consensus_times.append(elapsed_time)
            max_opinions = []
            for agent in agents:
                max_opinions.append(agent.max_opinions)
                agent.stop_threads()

            decision = utils.convert_decision_to_subimage_name(decisions[0], IMAGE_DIR) if elapsed_time < constants.MAX_EXPERIMENT_TIME else 'NO DECISION'
            number_of_different_opinions = utils.get_number_of_different_opinions(max_opinions)

            print(f'FINAL DECISION: {decision}, ELAPSED TIME: {elapsed_time}')

            cv2.destroyAllWindows()
            return test_image_name, decision, elapsed_time, number_of_different_opinions

        cv2.imshow(f'{test_image_name}', image_copy)
        cv2.waitKey(1)

        elapsed_time = time.time() - start_time
        time.sleep(0.01)

def main(image_code, results_filename, dm_method):
    """Main experiment entry.

    Args:
        image_code (string): Name of the original image or 'r' for randomly selected image.
    """
    for trial in range(NUMBER_OF_TRIALS):
        image, decision, elapsed_time, number_of_different_opinions = experiment(trial, image_code, dm_method)
        data_to_write = [image, decision, elapsed_time, dm_method, number_of_different_opinions]
        utils.write_results_to_csv(results_filename, data_to_write)

if __name__ == "__main__":
    image_code = sys.argv[1]
    results_filename = sys.argv[2]
    dm_method = sys.argv[3]

    if dm_method not in (constants.ASODM_METHOD, constants.DEMPSER_SHAFER_METHOD):
        print(f'Invalid decision making method: {dm_method}. Please choose either "{constants.ASODM_METHOD}" or "{constants.DEMPSER_SHAFER_METHOD}".')
        sys.exit(1)

    main(image_code, results_filename, dm_method)

    