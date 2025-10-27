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

NUMBER_OF_TRIALS = 100
IMAGE_DIR = "data/subimages/"

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
    number_in_x = 5
    number_in_y = 4

    initial_padding = [image.shape[1] - constants.AGENT_SIZE[0], image.shape[0] - constants.AGENT_SIZE[1]]

    x_positions = np.linspace(initial_padding[0], image.shape[1] - initial_padding[0], number_in_x)
    y_positions = np.linspace(initial_padding[1], image.shape[0] - initial_padding[1], number_in_y)

    positions = []

    for x in x_positions:
        for y in y_positions:
            positions.append(np.array([x, y]))
    
    return np.asarray(positions)

def visualize_agent(position, image, color):
    """Visualize agent on the image.

    Args:
        position (numpy.ndarray): Agent's position.
        image (numpy.ndarray): Copy of the original image.
        color (tuple): RGB color, representing agent's current opinion.
    """
    start_point = np.array(position - constants.AGENT_SIZE / 2).astype(np.int32)
    end_point = np.array(position + constants.AGENT_SIZE / 2).astype(np.int32)

    if color:
        color = tuple(reversed(color))
    cv2.rectangle(image, start_point, end_point, color, 5)
    # cv2.circle(image, position.astype(np.int32), constants.COMMUNICATION_RANGE, (255, 255, 0), 2)

def check_for_consenus(decisions, agents):
    """Check if agents have reached a consensus.

    Args:
        opinions (list): List of all agents opinions.
        agents (list): List of all agents.

    Returns:
        bool: True if consensus has been reached, False otherwise.
    """
    if len(decisions) != len(agents) or decisions[0] is None:
        return False
    return decisions.count(decisions[0]) == len(decisions)

def experiment(trial, image_code):
    """Single run of the experiment.

    Args:
        test_image (string): Name of the testing image from subimages folder or 'r' for random.
    """
    if image_code == 'r':
        test_image, test_image_name = utils.select_random_image(IMAGE_DIR)
    else:
        test_image = cv2.imread(IMAGE_DIR + image_code + '.png')
        test_image_name = image_code 

    print(f'TRIAL: {trial + 1}, SELECTED IMAGE: {test_image_name}')

    centroids = np.load('codebooks/trained_centroids.npy')
    codebook = np.load('codebooks/vlad_vectors.npy')

    positions = calculate_initial_positions(test_image)
    agents = [Agent(test_image, centroids, codebook, positions[i], str(i)) for i in range(len(positions))]

    start_time = time.time()
    elapsed_time = 0

    while True:
        image_copy = np.copy(test_image)

        decisions = []
        for agent in agents:
            visualize_agent(agent.get_position(), image_copy, (255, 0, 0))
            if agent.get_state() == constants.RECEIVING_STATE:
                agent.collect_opinions(agents)
            
            decisions.append(agent.calculate_decision())

        if check_for_consenus(decisions, agents) or elapsed_time > constants.MAX_EXPERIMENT_TIME:
            consensus_times.append(elapsed_time)
            for agent in agents:
                agent.stop_threads()

            decision = utils.convert_decision_to_subimage_name(decisions[0], IMAGE_DIR) if elapsed_time < constants.MAX_EXPERIMENT_TIME else 'NO DECISION'

            print(f'FINAL DECISION: {decision}, ELAPSED TIME: {elapsed_time}')

            cv2.destroyAllWindows()
            return test_image_name, decision, elapsed_time

        # cv2.imshow(f'{test_image_name}', image_copy)
        # cv2.waitKey(1)

        elapsed_time = time.time() - start_time
        time.sleep(0.01)

def main(image_code, results_filename):
    """Main experiment entry.

    Args:
        image_code (string): Name of the original image or 'r' for randomly selected image.
    """
    for trial in range(NUMBER_OF_TRIALS):
        image, decision, elapsed_time = experiment(trial, image_code)
        data_to_write = [image, decision, elapsed_time]
        utils.write_results_to_csv(results_filename, data_to_write)

if __name__ == "__main__":
    image_code = sys.argv[1]
    results_filename = sys.argv[2]

    main(image_code, results_filename)

    