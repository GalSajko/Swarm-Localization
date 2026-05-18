import csv
import fcntl
import os
import time

import numpy as np

def calculate_cosine_similarity(first_vector, second_vector):
    """Calculate cosine similarity between two vectors."""
    flat_first_vector = first_vector.flatten()
    flat_second_vector = second_vector.flatten()

    dot_product = np.dot(flat_first_vector, flat_second_vector)
    product_of_norms = np.linalg.norm(flat_first_vector) * np.linalg.norm(flat_second_vector)
    if product_of_norms.any():
        return dot_product / product_of_norms
    return -1

def evaluate_similarity(vlad_vectors, codebook):
    """Evaluate one VLAD vector against every codebook entry."""
    similarity_vector = []
    for codebook_vector in codebook:
        similarity_vector.append(np.round(calculate_cosine_similarity(vlad_vectors, codebook_vector), 3))
    return np.array(similarity_vector)

def softmax_over_vector(vector, tau = 0.05, gamma = 2.5):
    """Compute the sharpened softmax used as personal DS evidence."""
    vector = np.asarray(vector, dtype = np.float32)
    nom = np.exp(vector / tau)
    nom_gamma = np.power(nom, gamma)
    den = nom_gamma.sum()
    return nom / den

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
