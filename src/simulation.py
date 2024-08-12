import json
import time
import constants
import cv2
from drawnow import drawnow
import matplotlib.pyplot as plt
import numpy as np
import sys

from scipy.cluster.vq import vq

from agent import Agent
import image_augmentation as ia
import utils

IMAGE_DIR = "data/subimages/"

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

if __name__ == "__main__":
    image_name = sys.argv[1]

    original_image = cv2.imread(IMAGE_DIR + image_name + '.png')
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(original_image, None)
    original_image_with_keypoints = cv2.drawKeypoints(original_image, keypoints, 0, (0, 255, 0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    centroids = np.load('codebooks/visual_words.npy')
    weights = np.load('codebooks/visual_words_weights.npy')
    with open('codebooks/bow_codebook.json', 'r') as file:
        codebook = json.load(file)

    positions = calculate_initial_positions(original_image_with_keypoints)
    agents = [Agent(original_image, centroids, weights, codebook, positions[i], str(i)) for i in range(len(positions))]

    start_time = time.time()
    elapsed_time = 0
    while True:
        image_copy = np.copy(original_image_with_keypoints)

        opinions = []
        for agent in agents:
            visualize_agent(agent.get_position(), image_copy, agent.get_color())
            if agent.get_state() == constants.RECEIVING_STATE:
                agent.collect_social_information(agents)
            
            opinions.append(agent.opinion)
        
        if check_for_consenus(opinions, agents):
            print(f"CONSENSUS REACHED IN {elapsed_time}")
            break

        cv2.imshow(f"{image_name}", image_copy)
        cv2.waitKey(1)

        elapsed_time = time.time() - start_time
    
    for agent in agents:
        utils.draw_opinions(agent.all_collected_opinions)
