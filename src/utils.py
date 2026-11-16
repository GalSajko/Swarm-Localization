import csv
import fcntl
import os

import numpy as np

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
