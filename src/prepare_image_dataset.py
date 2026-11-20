"""Prepare the data needed for fuzzy pattern matching. Run from the project's root directory.

1. Train the fuzzy c-means centers on 'data/training_data/ims_gray.npy' and save them in 'codebooks/fcm_centers.npz'. This runs only if the centers
are missing or if --train is given.
2. Calculate the RootSIFT features of each whole terrain image from 'data/original_images/' and save them with their locations in
'codebooks/map_features_<terrain>.npz'. This runs only if the features are missing or if --extract is given.
3. Count in how many degraded copies of the terrain image every map feature is found again and save the counts in
'codebooks/map_repeatability_<terrain>.npz'. This runs only if the counts are missing, if they do not belong to the saved features, or if
--extract or --repeatability is given.
4. Select the codebook features with a sliding window of the agent's view size (the most repeatable feature of every window), calculate their
fuzzy memberships and save them with the feature locations, descriptors and the centers in 'codebooks/fcm_codebook_<terrain>.npz'. This step is
fast and always runs, so the membership fuzzifier (--membership_m) and the window step (--window_steps) can be changed without repeating steps 1
to 3.
"""

import argparse
import cv2
import numpy as np
import os
from scipy.cluster.vq import kmeans

import constants
import feature_repeatability
import fuzzy_clustering
import utils

#region DATASET PREPARATION
TRAINING_DATASET_PATHS = (
    os.path.join("data", "training_data", "ims_gray.npy"),
    os.path.join("data", "training_data", "ims_gray.np"),
)
KMEANS_SEED = 0
SAMPLING_SEED = 0
# The centers are trained on a random sample of descriptors, because all descriptors of the training dataset do not fit in memory.
MAX_TRAINING_DESCRIPTORS_PER_IMAGE = 1000
# K-means only initializes fuzzy c-means, so it runs on a smaller random sample of the training descriptors.
KMEANS_INITIALIZATION_SAMPLE_SIZE = 200_000

def load_training_images():
    """Load the grayscale training dataset (array of shape (N, H, W)) from the first existing path in TRAINING_DATASET_PATHS."""
    for path in TRAINING_DATASET_PATHS:
        if os.path.exists(path):
            return np.load(path, mmap_mode = "r")

    raise FileNotFoundError(f"Training dataset not found. Expected one of: {', '.join(TRAINING_DATASET_PATHS)}")
# endregion

#region SIFT
def root_sift_visual_features(list_of_images):
    """Compute keypoints and descriptors with SIFT algorithm.

    Args:
        list_of_images (list): List of images, represented as numpy arrays.

    Returns:
        tuple[list, list]: Keypoints and descriptors.
    """
    sift_extractor = constants.SIFT_EXTRACTOR
    keypoints = []
    descriptors = []

    for image in list_of_images:
        img_keypoints, img_descriptors = sift_extractor.detectAndCompute(image, None)
        if img_descriptors is None:
            img_descriptors = np.empty((0, constants.SIFT_EXTRACTOR.descriptorSize()), dtype = np.float32)
        root_sift_descriptors = utils.root_sift(img_descriptors)
        keypoints.append(img_keypoints)
        descriptors.append(root_sift_descriptors)
    
    return keypoints, descriptors

def sample_training_descriptors(dataset):
    """Compute RootSIFT descriptors of the dataset image by image and keep a random sample of each image's descriptors.

    Args:
        dataset (Sequence[numpy.ndarray]): Grayscale images.

    Returns:
        numpy.ndarray: Sampled descriptors of shape (N, 128).
    """
    random_generator = np.random.default_rng(SAMPLING_SEED)
    sampled_descriptors = []

    for index, image in enumerate(dataset):
        _, (descriptors,) = root_sift_visual_features([np.ascontiguousarray(image)])
        if len(descriptors) > MAX_TRAINING_DESCRIPTORS_PER_IMAGE:
            descriptors = descriptors[random_generator.choice(len(descriptors), MAX_TRAINING_DESCRIPTORS_PER_IMAGE, replace = False)]
        if len(descriptors) > 0:
            sampled_descriptors.append(descriptors)
        if (index + 1) % 100 == 0 or index + 1 == len(dataset):
            print(f"Extracted descriptors from {index + 1}/{len(dataset)} training images.", flush = True)

    return np.vstack(sampled_descriptors)
#endregion

#region CLUSTERING
def kmeans_over_descriptors(stacked_descriptors):
    """Kmeans clustering over all descriptors from dataset.

    Args:
        all_descriptors (numpy.ndaray): Array of all descriptors.

    Returns:
        numpy.ndarray: Array of centroids.
    """
    iters = 20
    centroids, _ = kmeans(stacked_descriptors, constants.NUMBER_OF_CLUSTERS, iters, seed = KMEANS_SEED)

    return centroids.astype(np.float32, copy = False)

def calculate_centers(dataset, training_fuzzifier):
    """Cluster RootSIFT descriptors of the dataset with fuzzy c-means, initialized with k-means.

    Args:
        dataset (Sequence[numpy.ndarray]): Grayscale images.
        training_fuzzifier (float): Fuzzifier used to train the centers.

    Returns:
        numpy.ndarray: Array of cluster centers.
    """
    descriptors = sample_training_descriptors(dataset)

    random_generator = np.random.default_rng(SAMPLING_SEED)
    sample_size = min(KMEANS_INITIALIZATION_SAMPLE_SIZE, len(descriptors))
    initialization_sample = descriptors[random_generator.choice(len(descriptors), sample_size, replace = False)]
    print(f"Initializing centers with k-means on {sample_size} descriptors.", flush = True)
    initial_centers = kmeans_over_descriptors(initialization_sample)

    print(f"Running fuzzy c-means on {len(descriptors)} descriptors.", flush = True)
    return fuzzy_clustering.fuzzy_c_means(descriptors, initial_centers, training_fuzzifier)
#endregion

#region PREPARE AND SAVE
def train_and_save_centers(training_fuzzifier):
    """Train centers on the training dataset and save them together with the fuzzifier they were trained with."""
    training_images = load_training_images()
    print(f"Training {constants.NUMBER_OF_CLUSTERS} centers on {len(training_images)} images. This can take a long time.")
    centers = calculate_centers(training_images, training_fuzzifier)

    os.makedirs(constants.CODEBOOKS_DIR, exist_ok = True)
    np.savez(constants.FCM_CENTERS_PATH, centers = centers, training_fuzzifier = training_fuzzifier)

    return centers, training_fuzzifier

def load_centers():
    """Load the trained centers and the fuzzifier they were trained with."""
    with np.load(constants.FCM_CENTERS_PATH) as centers_file:
        return centers_file["centers"], float(centers_file["training_fuzzifier"])

def get_terrain_map_features_path(terrain_name):
    return os.path.join(constants.CODEBOOKS_DIR, f"map_features_{terrain_name}.npz")

def get_terrain_codebook_path(terrain_name):
    return os.path.join(constants.CODEBOOKS_DIR, f"fcm_codebook_{terrain_name}.npz")

def extract_and_save_map_features(terrain_name, image):
    """Calculate RootSIFT features of the whole terrain image and save them with their locations.

    Args:
        terrain_name (str): Name of the terrain.
        image (numpy.ndarray): Grayscale terrain image.

    Returns:
        dict: Saved arrays. 'keypoints_xy' (N, 2) are pixel locations (x, y) in the terrain image, 'keypoints_meta' (N, 3) are SIFT size, angle and
        response, 'descriptors' (N, 128) are RootSIFT descriptors and 'map_shape' is (height, width) of the terrain image.
    """
    (keypoints,), (descriptors,) = root_sift_visual_features([image])
    map_features = {
        "keypoints_xy": np.array([keypoint.pt for keypoint in keypoints], dtype = np.float32).reshape(-1, 2),
        "keypoints_meta": np.array([(keypoint.size, keypoint.angle, keypoint.response) for keypoint in keypoints], dtype = np.float32).reshape(-1, 3),
        "descriptors": descriptors,
        "map_shape": np.array(image.shape[:2]),
    }
    np.savez(get_terrain_map_features_path(terrain_name), **map_features)

    return map_features

def load_map_features(terrain_name):
    """Load the features saved by 'extract_and_save_map_features'."""
    with np.load(get_terrain_map_features_path(terrain_name)) as map_features_file:
        return {key: map_features_file[key] for key in map_features_file.files}

def get_terrain_repeatability_path(terrain_name):
    return os.path.join(constants.CODEBOOKS_DIR, f"map_repeatability_{terrain_name}.npz")

def load_terrain_image(terrain_name):
    """Load the grayscale terrain image, or return None if it does not exist."""
    image_path = os.path.join(constants.ORIGINAL_IMAGES_DIR, constants.TERRAIN_IMAGE_NAMES[terrain_name])
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Skipping terrain '{terrain_name}', image not found: {image_path}")

    return image

def load_repeat_counts(terrain_name, number_of_features):
    """Load the saved repeat counts, or return None if they are missing or do not belong to the current features and settings."""
    repeatability_path = get_terrain_repeatability_path(terrain_name)
    if not os.path.exists(repeatability_path):
        return None

    with np.load(repeatability_path) as repeatability_file:
        repeat_counts = repeatability_file["repeat_counts"]
        number_of_copies = int(repeatability_file["number_of_copies"])
    if len(repeat_counts) != number_of_features or number_of_copies != constants.NUMBER_OF_DEGRADED_COPIES:
        return None

    return repeat_counts

def calculate_and_save_repeat_counts(terrain_name, image, map_features):
    """Count in how many degraded copies of the terrain image every map feature is found again and save the counts."""
    print(f"Calculating repeatability of {len(map_features['keypoints_xy'])} features of terrain '{terrain_name}'.", flush = True)
    repeat_counts = feature_repeatability.calculate_repeat_counts(image, map_features["keypoints_xy"], map_features["keypoints_meta"])
    np.savez(
        get_terrain_repeatability_path(terrain_name),
        repeat_counts = repeat_counts,
        number_of_copies = constants.NUMBER_OF_DEGRADED_COPIES,
    )

    return repeat_counts

def select_window_features(keypoints_xy, scores, map_shape, window_steps):
    """Select the best feature of every window of the agent's view size that slides over the map.

    The map is divided into cells of 1/window_steps of the window size. Every window covers window_steps x window_steps cells and starts at a cell
    corner, so the best feature of a window is the best of its cells' best features.

    Args:
        keypoints_xy (numpy.ndarray): Feature locations (N, 2) in pixels.
        scores (numpy.ndarray): Feature scores (N,), higher is better.
        map_shape (Sequence[int]): Map height and width.
        window_steps (int): Number of steps per window size.

    Returns:
        tuple[numpy.ndarray, float]: Sorted indices of the features that win at least one window, and the share of windows without any feature.
    """
    height, width = (int(value) for value in map_shape)
    cell_width, cell_height = constants.AGENT_SIZE / window_steps
    cells_x, cells_y = int(np.ceil(width / cell_width)), int(np.ceil(height / cell_height))
    cell_x = np.minimum((keypoints_xy[:, 0] // cell_width).astype(int), cells_x - 1)
    cell_y = np.minimum((keypoints_xy[:, 1] // cell_height).astype(int), cells_y - 1)
    cell = cell_y * cells_x + cell_x

    # Best feature of every cell.
    order = np.lexsort((-scores, cell))
    is_cell_best = np.r_[True, cell[order][1:] != cell[order][:-1]]
    cell_best_feature = np.full(cells_x * cells_y, -1)
    cell_best_feature[cell[order][is_cell_best]] = order[is_cell_best]
    cell_best_score = np.full(cells_x * cells_y, -np.inf)
    cell_best_score[cell[order][is_cell_best]] = scores[order][is_cell_best]

    # Best cell of every window.
    windows = np.lib.stride_tricks.sliding_window_view(cell_best_score.reshape(cells_y, cells_x), (window_steps, window_steps))
    windows = windows.reshape(cells_y - window_steps + 1, cells_x - window_steps + 1, window_steps * window_steps)
    best_cell_in_window = windows.argmax(axis = 2)
    has_feature = np.isfinite(windows.max(axis = 2))
    window_y, window_x = np.mgrid[0:windows.shape[0], 0:windows.shape[1]]
    best_cell = (window_y + best_cell_in_window // window_steps) * cells_x + window_x + best_cell_in_window % window_steps

    return np.unique(cell_best_feature[best_cell[has_feature]]), 1.0 - has_feature.mean()

def prepare_terrain_codebooks(
    retrain = False,
    reextract = False,
    recalculate_repeatability = False,
    training_fuzzifier = None,
    membership_fuzzifier = None,
    window_steps = None,
):
    """Prepare terrain-specific fuzzy codebooks.

    Centers are trained first if they are missing or if retrain is True. Map features are calculated if they are missing or if reextract is True.
    Repeat counts are calculated if they are missing, if the map features were recalculated or if recalculate_repeatability is True. The window
    selection and the memberships are always recalculated from the saved centers, map features and repeat counts.

    Args:
        retrain (bool, optional): Retrain the centers. Defaults to False.
        reextract (bool, optional): Recalculate the map features. Defaults to False.
        recalculate_repeatability (bool, optional): Recalculate the repeat counts of map features. Defaults to False.
        training_fuzzifier (float, optional): Fuzzifier for training the centers. Defaults to constants.TRAINING_FUZZIFIER.
        membership_fuzzifier (float, optional): Fuzzifier for the memberships of map features. Defaults to constants.MEMBERSHIP_FUZZIFIER.
        window_steps (int, optional): The selection window moves by 1/window_steps of its size. Defaults to constants.WINDOW_STEPS.
    """
    training_fuzzifier = constants.TRAINING_FUZZIFIER if training_fuzzifier is None else training_fuzzifier
    membership_fuzzifier = constants.MEMBERSHIP_FUZZIFIER if membership_fuzzifier is None else membership_fuzzifier
    window_steps = constants.WINDOW_STEPS if window_steps is None else window_steps
    if window_steps < 1:
        raise ValueError("Window steps must be at least 1.")

    if retrain or not os.path.exists(constants.FCM_CENTERS_PATH):
        centers, training_fuzzifier = train_and_save_centers(training_fuzzifier)
    else:
        centers, training_fuzzifier = load_centers()
    print(f"Using {len(centers)} centers trained with fuzzifier {training_fuzzifier}. Membership fuzzifier is {membership_fuzzifier}.")

    for terrain_name in constants.TERRAIN_NAMES:
        image = None
        if reextract or not os.path.exists(get_terrain_map_features_path(terrain_name)):
            image = load_terrain_image(terrain_name)
            if image is None:
                continue
            map_features = extract_and_save_map_features(terrain_name, image)
            repeat_counts = None
        else:
            map_features = load_map_features(terrain_name)
            repeat_counts = None if recalculate_repeatability else load_repeat_counts(terrain_name, len(map_features["keypoints_xy"]))

        if repeat_counts is None:
            image = load_terrain_image(terrain_name) if image is None else image
            if image is None:
                continue
            repeat_counts = calculate_and_save_repeat_counts(terrain_name, image, map_features)
        del image

        # SIFT response is below 1, so the score ranks by repeat count first and by response second.
        scores = repeat_counts + map_features["keypoints_meta"][:, 2].astype(np.float64)
        kept, empty_window_share = select_window_features(map_features["keypoints_xy"], scores, map_features["map_shape"], window_steps)
        memberships = fuzzy_clustering.calculate_memberships(map_features["descriptors"][kept], centers, membership_fuzzifier)
        np.savez(
            get_terrain_codebook_path(terrain_name),
            keypoints_xy = map_features["keypoints_xy"][kept],
            descriptors = map_features["descriptors"][kept],
            memberships = memberships,
            repeat_counts = repeat_counts[kept],
            centers = centers,
            membership_fuzzifier = membership_fuzzifier,
            training_fuzzifier = training_fuzzifier,
            window_steps = window_steps,
            number_of_degraded_copies = constants.NUMBER_OF_DEGRADED_COPIES,
            map_shape = map_features["map_shape"],
        )

        cluster_shares = memberships.sum(axis = 0, dtype = np.float64) / len(memberships)
        print(
            f"Prepared terrain '{terrain_name}': kept {len(kept)} of {len(repeat_counts)} features ({100 * len(kept) / len(repeat_counts):.1f}%), "
            f"{100 * empty_window_share:.1f}% of windows without a feature, kept features found in {np.median(repeat_counts[kept]):.0f} of "
            f"{constants.NUMBER_OF_DEGRADED_COPIES} degraded copies (median), median top membership {np.median(memberships.max(axis = 1)):.3f}, "
            f"cluster shares between {100 * cluster_shares.min():.2f}% and {100 * cluster_shares.max():.2f}%."
        )
#endregion

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Prepare fuzzy c-means centers and terrain codebooks. Run from the project's root directory.")
    parser.add_argument("--train", action = "store_true", help = f"Retrain the centers even if {constants.FCM_CENTERS_PATH} already exists.")
    parser.add_argument("--extract", action = "store_true", help = "Recalculate the map features and their repeatability even if they already exist.")
    parser.add_argument("--repeatability", action = "store_true", help = "Recalculate the repeatability of map features even if it already exists.")
    parser.add_argument("--training_m", type = float, default = None, help = f"Fuzzifier for training the centers (default {constants.TRAINING_FUZZIFIER}). Only used when the centers are trained.")
    parser.add_argument("--membership_m", type = float, default = None, help = f"Fuzzifier for the memberships of map features (default {constants.MEMBERSHIP_FUZZIFIER}).")
    parser.add_argument("--window_steps", type = int, default = None, help = f"The selection window of the agent's view size moves by 1/window_steps of its size (default {constants.WINDOW_STEPS}).")
    arguments = parser.parse_args()
    prepare_terrain_codebooks(
        arguments.train,
        arguments.extract,
        arguments.repeatability,
        arguments.training_m,
        arguments.membership_m,
        arguments.window_steps,
    )
