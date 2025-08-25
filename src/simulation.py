import csv
import json
import threading
import time

from matplotlib import pyplot as plt
import constants
import cv2
import numpy as np
import sys


from agent import Agent
import utils

IMAGE_DIR = "data/subimages/"

consensus_times = []
final_opinions = []
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

def check_for_consenus(opinions, agents):
    """Check if agents have reached a consensus.

    Args:
        opinions (list): List of all agents opinions.
        agents (list): List of all agents.

    Returns:
        bool: True if consensus has been reached, False otherwise.
    """
    if len(opinions) != len(agents) or opinions[0] is None:
        return False
    return opinions.count(opinions[0]) == len(opinions)

def experiment(original_image, algorithm, is_forgetting, visualize = False, image_name = None):
    """Single run of the experiment.

    Args:
        original_image (numpy.ndarray): Original image.
        is_forgetting (bool): True if forgetting is active, False otherwise.
        visualize (bool, optional): True if visualization is active, False otherwise. Defaults to False.
        image_name (string, optional): Name of the original (sub)image. Defaults to None.
    """
    centroids = np.load('codebooks/visual_words.npy')
    weights = np.load('codebooks/visual_words_weights.npy')
    with open('codebooks/bow_codebook.json', 'r') as file:
        codebook = json.load(file)

    positions = calculate_initial_positions(original_image)
    agents = [Agent(original_image, algorithm, centroids, codebook, positions[i], str(i), is_forgetting = is_forgetting, weights = weights) for i in range(len(positions))]

    start_time = time.time()
    elapsed_time = 0
    while True:
        image_copy = np.copy(original_image)

        opinions = []
        for agent in agents:
            # visualize_agent(agent.get_position(), image_copy, agent.get_color())
            visualize_agent(agent.get_position(), image_copy, (255, 0, 0))
            if agent.get_state() == constants.RECEIVING_STATE:
                agent.collect_social_information(agents)
            
            opinions.append(agent.get_opinion())
        
        if check_for_consenus(opinions, agents) or elapsed_time > constants.MAX_EXPERIMENT_TIME:
            # consensus_times.append(elapsed_time)
            for agent in agents:
                agent.stop_threads()
            break
        
        if visualize:
            cv2.imshow(f"{image_name}", image_copy)
        cv2.waitKey(1)

        elapsed_time = time.time() - start_time

    with agents_data_locker:
        consensus_times.append(elapsed_time)
        final_opinions.append(opinions[0])
    
def run_experiment_in_threads(number_of_threads, original_image, algorithm, is_forgetting):
    """Run experiment in multiple threads.

    Args:
        number_of_threads (int): Number of threads.
        original_image (string): Name of the original (sub)image.
        is_forgetting (bool): True if forgetting is active, False otherwise.

    Returns:
        list: List of threads.
    """    
    threads = [threading.Thread(target = experiment, daemon = False, args = (original_image, algorithm, is_forgetting, )) for _ in range(number_of_threads)]
    [thread.start() for thread in threads]
    
    return threads

def main(image_name, algorithm, is_forgetting, number_of_threads, visualize):
    """Main experiment entry.

    Args:
        image_name (string): Name of the original image.
        is_forgetting (bool): True if forgetting is active, False otherwise.
        number_of_threads (int): Number of experiment threads.
        visualize (bool): True if visualization is active, False otherwise.
    """
    if visualize and number_of_threads != 1:
            print("Can't visualize in more than one thread.")
            return

    original_image = cv2.imread(IMAGE_DIR + image_name + '.png')

    if visualize:
        experiment(original_image, algorithm, is_forgetting, visualize, image_name)
    else:
        experiment_threads = run_experiment_in_threads(number_of_threads, original_image, algorithm, is_forgetting)
        [thread.join() for thread in experiment_threads]
    
    consensus_time = np.mean(consensus_times)
    accuracy = final_opinions.count(image_name + '.png') / number_of_threads if consensus_time < constants.MAX_EXPERIMENT_TIME else 0

    utils.print_results(accuracy, consensus_time, final_opinions[0])
    utils.write_results_to_csv(accuracy, consensus_time)

    # plt.hist(consensus_times, bins = 15)
    # plt.show()
    
    # for agent in agents:
    #     utils.draw_opinions(agent.aggregated_decision_vectors, agent.id)

if __name__ == "__main__":
    image_name = sys.argv[1]
    algorithm = sys.argv[2]
    # is_forgetting = bool(int(sys.argv[3]))
    # number_of_threads = int(sys.argv[4])
    # visualize = bool(int(sys.argv[5]))
    
    main(image_name, algorithm, False, 1, True)

    