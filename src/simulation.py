from collections import Counter
import cv2
import numpy as np
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

    initial_padding = 350

    x_positions = np.linspace(initial_padding, image.shape[1] - initial_padding, number_in_x)
    y_positions = np.linspace(initial_padding, image.shape[0] - initial_padding, number_in_y)

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
    cv2.circle(image, position.astype(np.int32), constants.COMMUNICATION_RANGE, (255, 255, 0), 2)

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

def experiment(test_image):
    """Single run of the experiment.

    Args:
        test_image (string): Name of the testing image from subimages folder.
    """
    centroids = np.load('codebooks/trained_centroids.npy')
    codebook = np.load('codebooks/vlad_vectors.npy')

    for i in range(NUMBER_OF_TRIALS):
        positions = calculate_initial_positions(test_image)
        agents = [Agent(test_image, centroids, codebook, positions[i], str(i)) for i in range(len(positions))]

        start_time = time.time()
        elapsed_time = 0

        while True:
            image_copy = np.copy(test_image)

            decisions = []
            # opinions = []
            for agent in agents:
                visualize_agent(agent.get_position(), image_copy, (255, 0, 0))
                if agent.get_state() == constants.RECEIVING_STATE:
                    agent.collect_opinions(agents)
                
                decisions.append(agent.calculate_decision())
                # opinions.append(agent.opinions)
            
            # print(dict(Counter(decisions)))

            if check_for_consenus(decisions, agents) or elapsed_time > constants.MAX_EXPERIMENT_TIME:
                consensus_times.append(elapsed_time)
                for agent in agents:
                    agent.stop_threads()
                break
            
            cv2.imshow(f"{test_image}", image_copy)
            cv2.waitKey(1)

            elapsed_time = time.time() - start_time

            time.sleep(0.01)

        decision = utils.convert_decision_to_subimage_name(decisions[0], IMAGE_DIR) if elapsed_time < constants.MAX_EXPERIMENT_TIME else 'NO DECISION'
        print(f'TRIAL: {i + 1}, DECISION ON: {decision}, CONSENSUS TIME: {elapsed_time}')

        consensus_times.append(elapsed_time)
        final_decisions.append(decision)

def main(image_name):
    """Main experiment entry.

    Args:
        image_name (string): Name of the original image.
    """
    subimage = cv2.imread(IMAGE_DIR + image_name + '.png')

    # position = [500, 500]
    # start_point = np.array(position - constants.AGENT_SIZE / 2).astype(np.int32)
    # end_point = np.array(position + constants.AGENT_SIZE / 2).astype(np.int32)

    # photo = subimage[
    #         int(position[1] - constants.AGENT_SIZE[0] / 2) : int(position[1] + constants.AGENT_SIZE[0] / 2),
    #         int(position[0] - constants.AGENT_SIZE[1] / 2) : int(position[0] + constants.AGENT_SIZE[1] / 2)
    #         ]

    # photo = ia.pixelate_image(photo, (100, 100))
    # photo = ia.add_gaussian_noise(photo)
    # photo = ia.augment_camera_effects(photo)

    # cv2.imshow('photo', photo)
    # cv2.rectangle(subimage, start_point, end_point, (0, 0, 255), 5)
    # cv2.imshow('area', subimage)
    # cv2.waitKey(0)


    experiment(subimage)
    
    consensus_time = np.mean(consensus_times)
    accuracy = final_decisions.count(image_name + '.png') / NUMBER_OF_TRIALS

    print(f'ACCURACY: {accuracy}, AVERAGE CONSENSUS TIME: {consensus_time}')

    utils.write_results_to_csv(accuracy, consensus_time)

if __name__ == "__main__":
    image_name = sys.argv[1]
    
    main(image_name)

    