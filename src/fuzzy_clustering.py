"""Fuzzy c-means clustering of local descriptors.

'calculate_memberships' is the only part needed online: it assigns descriptors to fixed centers with a given fuzzifier. 'fuzzy_c_means' trains the
centers offline.
"""

import numpy as np

# Descriptors are processed in chunks so the (descriptors x centers) distance matrix never has to fit in memory at once.
CHUNK_SIZE = 100_000
# Lower bound on squared distances, so a descriptor that lies exactly on a center does not produce log(0).
MIN_SQUARED_DISTANCE = 1e-12

def calculate_squared_distances(descriptors, centers):
    """Calculate squared Euclidean distances between descriptors and centers.

    Args:
        descriptors (numpy.ndarray): Descriptors of shape (N, D).
        centers (numpy.ndarray): Cluster centers of shape (C, D).

    Returns:
        numpy.ndarray: Squared distances of shape (N, C).
    """
    descriptors = np.asarray(descriptors, dtype = np.float32)
    centers = np.asarray(centers, dtype = np.float32)
    squared_distances = (
        np.einsum("ij,ij->i", descriptors, descriptors)[:, None]
        - 2.0 * descriptors @ centers.T
        + np.einsum("ij,ij->i", centers, centers)[None, :]
    )

    return np.maximum(squared_distances, MIN_SQUARED_DISTANCE)

def calculate_memberships(descriptors, centers, fuzzifier):
    """Calculate fuzzy memberships of descriptors in clusters with fixed centers.

    The membership of descriptor x in cluster k is d_k^(-2 / (m - 1)) / sum_j d_j^(-2 / (m - 1)), where d_k is the distance between x and center k
    and m is the fuzzifier. It is evaluated in log space, because the exponent is large when m is close to 1.

    Args:
        descriptors (numpy.ndarray): Descriptors of shape (N, D), or one descriptor of shape (D,).
        centers (numpy.ndarray): Cluster centers of shape (C, D).
        fuzzifier (float): Fuzzifier m > 1. Memberships get harder as m approaches 1.

    Returns:
        numpy.ndarray: Memberships of shape (N, C), or (C,) for one descriptor. Each row sums to 1.
    """
    if not fuzzifier > 1:
        raise ValueError("Fuzzifier must be greater than 1.")

    descriptors = np.asarray(descriptors, dtype = np.float32)
    is_single_descriptor = descriptors.ndim == 1
    descriptors = np.atleast_2d(descriptors)
    memberships = np.empty((len(descriptors), len(centers)), dtype = np.float32)

    for start in range(0, len(descriptors), CHUNK_SIZE):
        squared_distances = calculate_squared_distances(descriptors[start : start + CHUNK_SIZE], centers)
        # Shift before exponentiation to avoid underflow without changing membership ratios.
        logits = -np.log(squared_distances) / (fuzzifier - 1)
        logits -= logits.max(axis = 1, keepdims = True)
        weights = np.exp(logits)
        memberships[start : start + CHUNK_SIZE] = weights / weights.sum(axis = 1, keepdims = True)

    return memberships[0] if is_single_descriptor else memberships

def update_centers(descriptors, centers, fuzzifier):
    """Run one fuzzy c-means iteration: each new center is the mean of all descriptors, weighted by membership^m.

    Args:
        descriptors (numpy.ndarray): Descriptors of shape (N, D).
        centers (numpy.ndarray): Current cluster centers of shape (C, D).
        fuzzifier (float): Fuzzifier m > 1.

    Returns:
        numpy.ndarray: Updated cluster centers of shape (C, D).
    """
    weighted_sums = np.zeros(centers.shape, dtype = np.float64)
    weight_totals = np.zeros(len(centers), dtype = np.float64)

    for start in range(0, len(descriptors), CHUNK_SIZE):
        chunk = np.asarray(descriptors[start : start + CHUNK_SIZE], dtype = np.float32)
        weights = calculate_memberships(chunk, centers, fuzzifier) ** fuzzifier
        weighted_sums += weights.T @ chunk
        weight_totals += weights.sum(axis = 0)

    # A cluster without any weight keeps its previous center.
    has_weight = weight_totals > 0
    new_centers = np.array(centers, dtype = np.float64)
    new_centers[has_weight] = weighted_sums[has_weight] / weight_totals[has_weight, None]

    return new_centers.astype(np.float32)

def fuzzy_c_means(descriptors, initial_centers, fuzzifier, max_iterations = 100, tolerance = 1e-4):
    """Train fuzzy c-means centers, starting from the given centers.

    Args:
        descriptors (numpy.ndarray): Descriptors of shape (N, D).
        initial_centers (numpy.ndarray): Starting cluster centers of shape (C, D), e.g. from k-means.
        fuzzifier (float): Fuzzifier m > 1.
        max_iterations (int, optional): Maximal number of iterations. Defaults to 100.
        tolerance (float, optional): Stop when no center moves more than this distance in one iteration. Defaults to 1e-4.

    Returns:
        numpy.ndarray: Cluster centers of shape (C, D).
    """
    centers = np.asarray(initial_centers, dtype = np.float32)

    for iteration in range(max_iterations):
        new_centers = update_centers(descriptors, centers, fuzzifier)
        largest_shift = float(np.linalg.norm(new_centers - centers, axis = 1).max())
        centers = new_centers
        print(f"Fuzzy c-means iteration {iteration + 1}: largest center shift {largest_shift:.6f}.", flush = True)
        if largest_shift < tolerance:
            break

    return centers
