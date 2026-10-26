import csv
import fcntl
import os
import time

import numpy as np

def calculate_cosine_similarity(first_vector, second_vector):
    """Calculate cosine similarity between two vectors, returning 0 if either has zero norm."""
    flat_first_vector = np.asarray(first_vector, dtype = np.float64).ravel()
    flat_second_vector = np.asarray(second_vector, dtype = np.float64).ravel()

    product_of_norms = np.linalg.norm(flat_first_vector) * np.linalg.norm(flat_second_vector)
    if product_of_norms == 0:
        return 0.0
    return float(np.dot(flat_first_vector, flat_second_vector) / product_of_norms)

def evaluate_similarity(vlad_vectors, codebook):
    """Evaluate one VLAD vector against every codebook entry."""
    flat_vlad = np.asarray(vlad_vectors, dtype = np.float64).ravel()
    flat_codebook = np.asarray(codebook, dtype = np.float64).reshape(len(codebook), -1)

    products_of_norms = np.linalg.norm(flat_codebook, axis = 1) * np.linalg.norm(flat_vlad)
    dot_products = flat_codebook @ flat_vlad
    return np.divide(dot_products, products_of_norms, out = np.zeros_like(dot_products), where = products_of_norms > 0)

def softmax_over_vector(vector, tau = 0.02):
    """Convert cosine similarities into normalized personal evidence."""
    scores = np.asarray(vector, dtype = np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("Expected a nonempty similarity vector.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Similarities must be finite.")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("Temperature must be finite and positive.")

    # Shift before exponentiation to avoid overflow without changing tile ratios.
    logits = (scores - scores.max()) / tau
    weights = np.exp(logits)
    return weights / weights.sum()

def sleep_with_adjustable_period(loop_period, start_time):
    """Sleep for the remaining time so the loop lasts for the desired period."""
    sleep_time = float(loop_period) - (time.time() - start_time)
    if sleep_time >= 0:
        time.sleep(float(sleep_time))
        return
    time.sleep(0)

def write_results_to_csv(filename, data_to_write, header = None):
    """Append one experiment result row to a csv file and add a header if needed."""
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok = True)
    with open(filename, "a", newline = "", encoding = "utf-8") as file_handle:
        fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX)
        writer = csv.writer(file_handle)
        file_handle.seek(0, os.SEEK_END)
        needs_header = header is not None and file_handle.tell() == 0
        if needs_header:
            writer.writerow(header)
        flattened = [
            item
            for sublist in data_to_write
            for item in (sublist if isinstance(sublist, list) else [sublist])
        ]
        writer.writerow(flattened)
        file_handle.flush()
        os.fsync(file_handle.fileno())
        fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)

def convert_decision_to_subimage_name(decision, subimage_dir):
    """Convert a codebook index to the corresponding subimage filename stem."""
    subimages = sorted(file_name for file_name in os.listdir(subimage_dir) if file_name.lower().endswith(".png"))
    return subimages[decision][:-4]

def root_sift(sift_descriptors):
    """Apply RootSIFT normalization to local descriptors."""
    if sift_descriptors is None:
        return None

    descriptors = sift_descriptors.astype(np.float32)
    l1_norm = descriptors.sum(axis = 1, keepdims = True) + 1e-7
    descriptors /= l1_norm
    descriptors = np.sqrt(descriptors)

    l2_norm = np.linalg.norm(descriptors, axis = 1, keepdims = True) + 1e-7
    return descriptors / l2_norm
