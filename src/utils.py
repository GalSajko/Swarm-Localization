import csv
import cv2
import constants
import matplotlib.pyplot as plt
import numpy as np
import os
import time

OPINION_DYNAMICS_FILE = "opinion_dynamics.csv"

def set_timer(low, max):
    """Set random timer for uniformly distributed values.

    Args:
        low (float): Low value.
        max (float): High value

    Returns:
        tuple[float, float]: Start time and final time.
    """
    final_time = np.random.uniform(low = low, high = max)
    start_time = time.time()

    return start_time, final_time

def calculate_cosine_similarity(first_vector, second_vector):
    """Calculate cosine similarity between two vectors.

    Args:
        first_vector (numpy.ndarray): First vector.
        second_vector (numpy.ndarray): Second vector.

    Returns:
        float: Cosine similarity between first and second vector.
    """
    flat_first_vector = first_vector.flatten()
    flat_second_vector = second_vector.flatten()

    dot_product = np.dot(flat_first_vector, flat_second_vector)
    product_of_norms = np.linalg.norm(flat_first_vector) * np.linalg.norm(flat_second_vector)
    
    if product_of_norms.any():
        return dot_product / product_of_norms

    return -1

def evaluate_similarity(vlad_vectors, codebook):
    similarity_vector = []
    for codebook_vector in codebook:
        similarity_vector.append(np.round(calculate_cosine_similarity(vlad_vectors, codebook_vector), 3))
    return np.array(similarity_vector)

def softmax_over_vector(vector, tau = 0.05, gamma = 2.5):
    vector = np.asarray(vector, dtype = np.float32)
    nom = np.exp(vector / tau)
    nom_gamma = np.power(nom, gamma)
    den = nom_gamma.sum()

    return nom / den

def sleep_with_adjustable_period(loop_period, start_time):
    """Sleep for the remaining time, ensuring that the whole loop lasts for desired period.

    Args:
        max_period (float): Desired loop period.
        start_time (float): Time at which the loop begun.
    """
    sleep_time = loop_period - (time.time() - start_time)
    if sleep_time >= 0:
        time.sleep(sleep_time)
        return
    
    time.sleep(0)

def draw_opinions(all_collected_opinions, figure_name):
    """Draw the course of collected opinions through the localization process.

    Args:
        all_collected_opinions (list): List of combined collected opinions.
        figure_name (str): Name of the figure.
    """
    all_collected_opinions = np.array(all_collected_opinions)
    plt.figure(figure_name)
    time_steps = range(0, len(all_collected_opinions), 1)
    colors = list(constants.AGENT_COLOR_OPINION_DICT.values())
    for i in range(len(all_collected_opinions[0])):
        image_color = np.array(colors[i]) / 255
        plt.plot(time_steps, all_collected_opinions[:, i], color = image_color)
    plt.show()

def get_biggest_element_and_difference_to_second(list):
    """Select and return two biggest elements from the list.

    Args:
        list (list): List.

    Returns:
        tuple: Two biggest elements from the list and their indices.
    """
    sorted_indices = np.argsort(list)[::-1]
    max_value = list[sorted_indices[0]]
    second_max_value = list[sorted_indices[1]]

    return max_value, max_value - second_max_value, sorted_indices[0].astype(np.int8)

def print_results(accuracy, elapsed_time):
    """Print experiment results in the terminal.

    Args:
        accuracy (float): Accuracy.
        elapsed_time (float): Elapsed time.
    """
    if elapsed_time > constants.MAX_EXPERIMENT_TIME:
        print("NO CONSENSUS")
    else:
        print(f"AVERAGE CONSENSUS TIME: {elapsed_time}, ACCURACY: {accuracy}")

def write_results_to_csv(filename, data_to_write):
    with open(filename, 'a', newline = '', encoding = 'utf-8') as f:
        writer = csv.writer(f)
        data_to_write = [item for sublist in data_to_write for item in (sublist if isinstance(sublist, list) else [sublist])]
        writer.writerow(data_to_write)
        f.close()

def convert_decision_to_subimage_name(decision, subimage_dir):
    subimages = sorted([f for f in os.listdir(subimage_dir) if f.lower().endswith('.png')])
    return subimages[decision][:-4]

def root_sift(sift_descriptors):
    if sift_descriptors is None:
        return None
    
    descriptors = sift_descriptors.astype(np.float32)
    l1_norm = descriptors.sum(axis = 1, keepdims = True) + 1e-7
    descriptors /= l1_norm
    descriptors = np.sqrt(descriptors)

    l2_norm = np.linalg.norm(descriptors, axis = 1, keepdims = True) + 1e-7
    descriptors = descriptors / l2_norm

    return descriptors

def select_random_image(image_dir):
    subimages = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])
    random_index = np.random.randint(0, len(subimages))
    random_image_name = subimages[random_index]
    random_image = cv2.imread(os.path.join(image_dir, random_image_name))

    return random_image, random_image_name[:-4]

def get_number_of_different_opinions(all_max_opinions):
    max_length = max([len(max_opinions) for max_opinions in all_max_opinions])
    
    number_of_different_opinions = []
    for i in range(max_length):
        opinions_at_time_step = set()
        for max_opinions in all_max_opinions:
            if i < len(max_opinions):
                opinions_at_time_step.add(max_opinions[i])
        number_of_different_opinions.append(len(opinions_at_time_step))
    
    return number_of_different_opinions
