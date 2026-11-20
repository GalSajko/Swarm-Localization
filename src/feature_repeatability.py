"""Repeatability of map features under the simulated camera degradation.

The whole map is degraded several times with the same operations that are applied to an agent's photo. A map feature is repeatable if SIFT finds a
keypoint at the same place, with a similar size and orientation, in most of the degraded copies. Such features are the ones an agent is likely to
detect, so the codebook keeps the most repeatable feature of every agent-view window (prepare_image_dataset.select_window_features).
"""

import numpy as np
from scipy.spatial import cKDTree

import constants
import image_augmentation as ia

REPEATABILITY_SEED = 0
# A keypoint of a degraded copy is the same feature as a map keypoint if it lies within this distance (in map pixels), its size differs by less
# than this factor and its orientation by less than this angle (in degrees).
MAX_POSITION_DIFFERENCE = 2.5
MAX_SIZE_FACTOR = 1.3
MAX_ANGLE_DIFFERENCE = 20.0
# One location can hold several keypoints (different sizes or orientations), so more than one nearby keypoint is compared.
NEIGHBORS_TO_CHECK = 8

def calculate_degradation_parameters(number_of_copies, random_generator):
    """Spread the degradation strengths of the copies evenly over the ranges used for agents' photos.

    Each of the three ranges is covered from end to end, and the three lists are shuffled independently so every copy gets a different combination.

    Args:
        number_of_copies (int): Number of degraded copies.
        random_generator (numpy.random.Generator): Random generator used for shuffling.

    Returns:
        list[tuple[float, float, float]]: Zoom factor, resolution factor and noise sigma of every copy.
    """
    zoom_factors = random_generator.permutation(np.linspace(*ia.ZOOM_FACTOR_RANGE, number_of_copies))
    resolution_factors = random_generator.permutation(np.linspace(*ia.RESOLUTION_FACTOR_RANGE, number_of_copies))
    noise_sigmas = random_generator.permutation(np.linspace(*ia.NOISE_SIGMA_RANGE, number_of_copies))

    return [(float(zoom), float(resolution), float(sigma)) for zoom, resolution, sigma in zip(zoom_factors, resolution_factors, noise_sigmas)]

def find_repeated_keypoints(keypoints_xy, keypoints_meta, copy_keypoints_xy, copy_sizes, copy_angles):
    """Find which map keypoints have a matching keypoint in a degraded copy.

    Args:
        keypoints_xy (numpy.ndarray): Locations (x, y) of map keypoints, shape (N, 2).
        keypoints_meta (numpy.ndarray): SIFT size, angle and response of map keypoints, shape (N, 3).
        copy_keypoints_xy (numpy.ndarray): Locations of the copy's keypoints in map pixels, shape (M, 2).
        copy_sizes (numpy.ndarray): Sizes of the copy's keypoints in map pixels, shape (M,).
        copy_angles (numpy.ndarray): Orientations of the copy's keypoints in degrees, shape (M,).

    Returns:
        numpy.ndarray: Boolean array of shape (N,), True where the map keypoint was found in the copy.
    """
    if len(copy_keypoints_xy) == 0:
        return np.zeros(len(keypoints_xy), dtype = bool)

    neighbors_to_check = min(NEIGHBORS_TO_CHECK, len(copy_keypoints_xy))
    distances, indices = cKDTree(copy_keypoints_xy).query(
        keypoints_xy, k = neighbors_to_check, distance_upper_bound = MAX_POSITION_DIFFERENCE, workers = -1
    )
    distances = distances.reshape(len(keypoints_xy), neighbors_to_check)
    indices = indices.reshape(len(keypoints_xy), neighbors_to_check)

    # Missing neighbors have infinite distance and an index out of range.
    is_near = np.isfinite(distances)
    indices = np.where(is_near, indices, 0)
    size_factors = keypoints_meta[:, 0, None] / copy_sizes[indices]
    angle_differences = np.abs((keypoints_meta[:, 1, None] - copy_angles[indices] + 180.0) % 360.0 - 180.0)
    is_match = (
        is_near
        & (size_factors > 1.0 / MAX_SIZE_FACTOR)
        & (size_factors < MAX_SIZE_FACTOR)
        & (angle_differences < MAX_ANGLE_DIFFERENCE)
    )

    return is_match.any(axis = 1)

def calculate_repeat_counts(image, keypoints_xy, keypoints_meta, number_of_copies = None):
    """Count in how many degraded copies of the map every map keypoint is found again.

    Args:
        image (numpy.ndarray): Grayscale terrain image.
        keypoints_xy (numpy.ndarray): Locations (x, y) of map keypoints, shape (N, 2).
        keypoints_meta (numpy.ndarray): SIFT size, angle and response of map keypoints, shape (N, 3).
        number_of_copies (int, optional): Number of degraded copies. Defaults to constants.NUMBER_OF_DEGRADED_COPIES.

    Returns:
        numpy.ndarray: Number of copies in which each keypoint was found, shape (N,), values from 0 to number_of_copies.
    """
    number_of_copies = constants.NUMBER_OF_DEGRADED_COPIES if number_of_copies is None else number_of_copies
    random_generator = np.random.default_rng(REPEATABILITY_SEED)
    keypoints_xy = np.asarray(keypoints_xy, dtype = np.float64)
    repeat_counts = np.zeros(len(keypoints_xy), dtype = np.int16)

    degradation_parameters = calculate_degradation_parameters(number_of_copies, random_generator)
    for copy_index, (zoom_factor, resolution_factor, noise_sigma) in enumerate(degradation_parameters):
        degraded_image = ia.degrade_whole_image(image, zoom_factor, resolution_factor, noise_sigma, random_generator)
        scale_x = degraded_image.shape[1] / image.shape[1]
        scale_y = degraded_image.shape[0] / image.shape[0]
        copy_keypoints = constants.SIFT_EXTRACTOR.detect(degraded_image, None)
        del degraded_image

        # Convert the copy's keypoints back to map pixels by undoing the enlargement (pixel centers are at half-integer coordinates).
        copy_points = np.array([keypoint.pt for keypoint in copy_keypoints], dtype = np.float64).reshape(-1, 2)
        copy_keypoints_xy = np.column_stack(((copy_points[:, 0] + 0.5) / scale_x - 0.5, (copy_points[:, 1] + 0.5) / scale_y - 0.5))
        copy_sizes = np.array([keypoint.size for keypoint in copy_keypoints], dtype = np.float64) / scale_x
        copy_angles = np.array([keypoint.angle for keypoint in copy_keypoints], dtype = np.float64)

        is_repeated = find_repeated_keypoints(keypoints_xy, keypoints_meta, copy_keypoints_xy, copy_sizes, copy_angles)
        repeat_counts += is_repeated
        print(
            f"Degraded copy {copy_index + 1}/{number_of_copies} (zoom {zoom_factor:.3f}, resolution {resolution_factor:.2f}, noise {noise_sigma:.1f}): "
            f"found {100 * is_repeated.mean():.1f}% of map features.",
            flush = True,
        )

    return repeat_counts
