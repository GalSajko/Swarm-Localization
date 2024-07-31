import numpy as np
import time

AGENT_COLOR_OPINION_DICT = {
    '1_1.png' : (255, 51, 0),
    '1_2.png' : (255, 191, 0),
    '2_1.png' : (128, 255, 0),
    '2_2.png' : (0, 255, 255),
    '3_1.png' : (0, 64, 255),
    '3_2.png' : (128, 0, 255),
    '4_1.png' : (191, 0, 255),
    '4_2.png' : (255, 0, 255),
    '5_1.png' : (255, 128, 0),
    '5_2.png' : (255, 0, 128)
}

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

    return dot_product / product_of_norms
