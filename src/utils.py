import csv
import constants
import matplotlib.pyplot as plt
import numpy as np
import time

RESULTS_FILE = "results.csv"

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
    dot_product = np.dot(first_vector, second_vector)
    product_of_norms = np.linalg.norm(first_vector) * np.linalg.norm(second_vector)
    
    if product_of_norms.any():
        return dot_product / product_of_norms

    return -1

def sleep_with_adjustable_period(loop_period, start_time):
    """Sleep for the remaining time, ensuring that the whole loop lasts for desired period.

    Args:
        max_period (float): Desired loop period.
        start_time (float): Time at which the loop begun.
    """
    if sleep_time := loop_period - (time.time() - start_time) >= 0:
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

def print_results(accuracy, elapsed_time, opinion):
    """Print experiment results in the terminal.

    Args:
        accuracy (float): Accuracy.
        elapsed_time (float): Elapsed time.
    """
    if elapsed_time > constants.MAX_EXPERIMENT_TIME:
        print("NO CONSENSUS")
    else:
        print(f"AVERAGE CONSENSUS TIME: {elapsed_time}, ACCURACY: {accuracy}, OPINION: {opinion}")

def write_results_to_csv(accuracy, elapsed_time):
    with open(RESULTS_FILE, 'a', newline = '', encoding = 'utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([elapsed_time, accuracy])
        f.close()