import json
import constants
import cv2
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

def visualize_agent(position, image, id, color):
    """Visualize agent on the image.

    Args:
        position (numpy.ndarray): Agent's position.
        image (numpy.ndarray): Copy of the original image.
    """
    start_point = np.array(position - constants.AGENT_SIZE / 2).astype(np.int32)
    end_point = np.array(position + constants.AGENT_SIZE / 2).astype(np.int32)

    # if id == '0':
    #     color = (255, 0, 0)
    # else:
    #     color = (0, 0, 255)

    cv2.rectangle(image, start_point, end_point, color, 5)    

def image_recognition_test_with_masked_image(original_image, sift, centroids, weights, codebook):
    masked_image = np.copy(original_image)

    number_of_boxes_x = int(masked_image.shape[1] / constants.AGENT_SIZE[0])
    boxes_coordinates_x = np.linspace(int(constants.AGENT_SIZE[0] / 2), masked_image.shape[1] - int(constants.AGENT_SIZE[0]), number_of_boxes_x)
    number_of_boxes_y = int(masked_image.shape[0] / constants.AGENT_SIZE[1])
    boxes_coordinates_y = np.linspace(int(constants.AGENT_SIZE[1] / 2), masked_image.shape[0] - int(constants.AGENT_SIZE[1]), number_of_boxes_y)

    desired_covered_percentage = 0.2
    needed_number_of_boxes = int(np.prod(masked_image.shape[:2]) * desired_covered_percentage / np.prod(constants.AGENT_SIZE))

    centers_of_boxes = [[np.random.choice(boxes_coordinates_x), np.random.choice(boxes_coordinates_y)] for _ in range(needed_number_of_boxes)]
    
    for center in centers_of_boxes:
        masked_image[
            int(center[1]) - int(constants.AGENT_SIZE[0] / 2) : int(center[1]) + int(constants.AGENT_SIZE[0] / 2),
            int(center[0]) - int(constants.AGENT_SIZE[1] / 2) : int(center[0]) + int(constants.AGENT_SIZE[1] / 2)
            ] = 0
    
    masked_keypoints, masked_descriptors = sift.detectAndCompute(masked_image, None)
    masked_image_with_keypoints = cv2.drawKeypoints(masked_image, masked_keypoints, 0, (0, 255, 0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    masked_histogram = np.zeros(len(centroids))
    masked_centroids_indexes, _ = vq(masked_descriptors, centroids)
    for index in masked_centroids_indexes:
        masked_histogram[index] += 1
    
 
    important_weights_indices = np.where(weights >= 2.5 * np.mean(weights))[0].astype(np.int16)

    weighted_masked_histogram = masked_histogram[important_weights_indices] * weights[important_weights_indices]

    cosine_similarities = []

    for weighted_histogram in codebook.values():
        cosine_similarities.append(np.round(utils.calculate_cosine_similarity(weighted_masked_histogram, np.array(weighted_histogram)[important_weights_indices]), 4))
    closest_image = list(codebook.keys())[np.argmax(cosine_similarities)]

    print(f"Closest image: {closest_image}, cosine similarity: {max(cosine_similarities)}")
    print(f"All cosine similarities: {cosine_similarities}")

    return masked_image_with_keypoints

if __name__ == "__main__":
    image_name = sys.argv[1]

    original_image = cv2.imread(IMAGE_DIR + image_name + '.png')
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(original_image, None)
    original_image_with_keypoints = cv2.drawKeypoints(original_image, keypoints, 0, (0, 255, 0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # photo_segment = original_image[500 : 700, 500 : 700]
    # augmented_photo_segment = ia.add_gaussian_noise(photo_segment)
    # # augmented_photo_segment = ia.pixelate_image(photo_segment, (20, 20))
    # print(augmented_photo_segment)

    # cv2.imshow("original photo segment", photo_segment)
    # cv2.imshow("augmented photo segment", augmented_photo_segment)
    # cv2.waitKey( )

    centroids = np.load('codebooks/visual_words.npy')
    visual_words_in_images = np.load('codebooks/visual_words_in_images.pkl', allow_pickle = True)
    visual_words_histograms = np.load('codebooks/visual_words_histograms.npy')
    weights = np.load('codebooks/visual_words_weights.npy')
    with open('codebooks/bow_codebook.json', 'r') as file:
        codebook = json.load(file)

    positions = calculate_initial_positions(original_image_with_keypoints)
    agents = [Agent(original_image, centroids, weights, codebook, positions[i], str(i)) for i in range(len(positions))]

    while True:
        image_copy = np.copy(original_image_with_keypoints)

        cosine_similarities = []
        for agent in agents:
            visualize_agent(agent.get_position(), image_copy, agent.id, agent.get_color())
            cosine_similarities.append(agent.get_cosine_similarities())

        if not None in cosine_similarities:
            average_cosine_similarities = np.mean(np.array(cosine_similarities), axis = 0)
            closest_image = list(codebook.keys())[np.argmax(average_cosine_similarities)]
            print(f"Closest image: {closest_image}, cosine similarities: {average_cosine_similarities}")
     
        cv2.imshow("img", image_copy)
        cv2.waitKey(1)    