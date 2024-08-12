import matplotlib.pyplot as plt
import matplotlib.pyplot
import numpy as np
import time

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

    return 0

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

def draw_opinions(all_collected_opinions):
    """Draw the course of collected opinions through the localization process.

    Args:
        all_collected_opinions (list): List of combined collected opinions.
    """
    all_collected_opinions = np.array(all_collected_opinions)
    plt.figure()
    time_steps = range(0, len(all_collected_opinions))
    for i in range(len(all_collected_opinions[0])):
        plt.plot(time_steps, all_collected_opinions[:, i])
    plt.show()
