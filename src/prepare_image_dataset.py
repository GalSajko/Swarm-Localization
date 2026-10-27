"""Prepare the data needed by the simulation. Run from the project's root directory.

1. Train the VLAD centroids on 'data/training_data/ims_gray.np' and save them in 'codebooks/trained_centroids.npy'. This runs only if the centroids
are missing or if --train is given.
2. Cut each terrain image from 'data/original_images/' into a grid of subimages and save them in 'data/subimages/<terrain>/'.
3. Calculate the VLAD vector of every subimage with the trained centroids and save them in 'codebooks/vlad_vectors_<terrain>.npy'.
"""

import argparse
import cv2
import glob
import math
import numpy as np
import os
from scipy.cluster.vq import kmeans, vq

import constants
import utils

#region DATASET PREPARATION
TRAINING_DATASET_PATHS = (
    os.path.join("data", "training_data", "ims_gray.npy"),
    os.path.join("data", "training_data", "ims_gray.np"),
)
KMEANS_SEED = 0

def divide_image_on_subimages_and_save(image, divide_on, subimages_dir = constants.SUBIMAGES_DIR, remove_old_files = True):
    """Divide original image on desired number of subimages and save them in subimages directory.

    Args:
        image (numpy.ndarray): Original image as numpy array.
        divide_on (tuple[int, int]): Number of subimages in horizontal and vertical direction.
        subimages_dir (string): Directory where subimages will be stored.
        remove_old_files(bool, optional): If True, remove previous image files in the directory. Defaults to True.
    """
    os.makedirs(subimages_dir, exist_ok = True)

    if remove_old_files:
        remove_files_from_dir(subimages_dir)

    shape = image.shape
    
    subimage_x_size = math.floor(shape[1] / divide_on[0])
    subimage_y_size = math.floor(shape[0] / divide_on[1])
    
    for y in range(divide_on[1]):
        for x in range(divide_on[0]):
            sub_image = image[y * subimage_y_size : (y + 1) * subimage_y_size, x * subimage_x_size : (x + 1) * subimage_x_size]
            first_index = x + 1 if x + 1 >= 10 else '0' + str(x + 1)
            second_index = y + 1 if y + 1 >= 10 else '0' + str(y + 1)
            cv2.imwrite(os.path.join(subimages_dir, f"{first_index}_{second_index}.png"), sub_image)

def get_dict_of_dataset_images(dir):
    """Save images and their names in dictionary.

    Args:
        dir (string): Directory with images.,

    Returns:
        dict: Dictionary with filenames as keys and images as numpy arrays as values.
    """
    filenames_and_images = {}

    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        if filename.endswith(('.png', '.jpg')):
            image = cv2.imread(dir + filename, cv2.IMREAD_GRAYSCALE)
            filenames_and_images[filename] = image

    keys = list(filenames_and_images.keys())
    keys.sort()
    filenames_and_images = {i: filenames_and_images[i] for i in keys}

    return filenames_and_images

def remove_files_from_dir(dir):
    """Remove files from given directory.

    Args:
        dir (string): Directory.
    """
    files = glob.glob(os.path.join(dir, '*'))
    for file in files:
        if os.path.basename(file) == '.gitkeep':
            continue
        if os.path.isdir(file):
            continue
        os.remove(file)

# endregion

#region SIFT 
NUMBER_OF_CLUSTERS = 256

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

def kmeans_over_descriptors(stacked_descriptors):
    """Kmeans clustering over all descriptors from dataset.

    Args:
        all_descriptors (numpy.ndaray): Array of all descriptors.

    Returns:
        numpy.ndarray: Array of centroids.
    """
    iters = 20
    centroids, _ = kmeans(stacked_descriptors, NUMBER_OF_CLUSTERS, iters, seed = KMEANS_SEED)

    return centroids.astype(np.float32, copy = False)

def calculate_centroids(dataset):
    """Cluster RootSIFT descriptors of all images in the dataset.

    Args:
        dataset (Sequence[numpy.ndarray]): Grayscale images.

    Returns:
        numpy.ndarray: Array of centroids.
    """
    _, descriptors_in_images = root_sift_visual_features(dataset)
    descriptors_in_images = [descriptors for descriptors in descriptors_in_images if len(descriptors) > 0]

    return kmeans_over_descriptors(np.vstack(descriptors_in_images))

def map_descriptors_to_centroids(descriptors, centroids):
    """Assign descriptors to the closest centroids and save them as list of visual words.

    Args:
        descriptors (numpy.ndarray): Arrays of descriptors of images.
        centroids (numpy.ndarray): Array of centroids of clusters.

    Returns:
        numpy.ndarray: List of visual words for each image.
    """
    mapped_descriptors = []
    for descriptor in descriptors:
        if len(descriptor) == 0:
            mapped_descriptors.append(np.empty(0, dtype = np.int64))
            continue

        # For each descriptor return index of centroid the descriptor belongs to.
        visual_words_in_image, _ = vq(descriptor, centroids)
        mapped_descriptors.append(visual_words_in_image)
    
    return mapped_descriptors
#endregion

#region VLAD
TRAINED_CENTROIDS_NPY = constants.TRAINED_CENTROIDS_PATH

def calculate_vlad_vectors(descriptors, mapped_descriptors, centroids, normalize = True):
    """Calculate vlad descriptors for all images.

    Args:
        descriptors (numpy.ndarray): SIFT descriptors.
        mapped_descriptors (numpy.ndarray): SIFT descriptors mapped to centroids.
        centroids (numpy.ndarray): Centroids.

    Returns:
        numpy.ndarray: Vlad descriptors of all images.
    """
    vlad_vectors = np.zeros((len(descriptors), len(centroids), centroids.shape[1]), dtype = np.float32)    
    for i, (descriptor, centroid_indices) in enumerate(zip(descriptors, mapped_descriptors)):
        if descriptor.size == 0:
            continue

        descriptor = np.asarray(descriptor, dtype = np.float32)
        centroid_indices = np.asarray(centroid_indices, dtype = np.int64)
        
        residual = descriptor - centroids[centroid_indices]
        np.add.at(vlad_vectors[i], centroid_indices, residual)
    
    if normalize:
        vlad_vectors = normalize_vlad_vectors(vlad_vectors)

    return vlad_vectors

def intra_normalization(vectors):
    """Apply L2 normalization to vlad vectors row-wise.

    Args:
        vlad_vectors (numpy.ndarray): Vlad vectors.

    Returns:
        numpy.ndarray: Intra-normalized vlad vectors.
    """
    vectors = np.asarray(vectors, dtype = np.float32, order = 'C')

    eps = 1e-12
    
    N, K, D = vectors.shape
    flat = vectors.reshape(-1, D)
    norms = np.linalg.norm(flat, axis = 1)
    mask = norms > eps
    flat[mask] /= norms[mask, None]

    return flat.reshape(N, K, D)

def global_l2_normalization(vectors):
    """Apply global l2 normalization on each vlad vector.

    Args:
        vectors (numpy.ndarray): Vlad vectors.

    Returns:
        numpy.ndarray: Globaly l2 normalized vlad vectors.
    """
    eps = 1e-12

    N, K, D = vectors.shape
    flat = vectors.reshape(N, -1)
    norms = np.linalg.norm(flat, axis = 1)
    mask = norms > eps
    flat[mask] /= norms[mask, None]

    return flat.reshape(N, K, D)

def normalize_vlad_vectors(vlad_vectors):
    intra_norm_vectors = intra_normalization(vlad_vectors)
    global_l2_over_intra_vectors = global_l2_normalization(intra_norm_vectors)

    return global_l2_over_intra_vectors
#endregion

#region PREPARE AND SAVE
def load_training_images():
    """Load the grayscale training dataset (array of shape (N, H, W)) from the first existing path in TRAINING_DATASET_PATHS."""
    for path in TRAINING_DATASET_PATHS:
        if os.path.exists(path):
            return np.load(path, mmap_mode = "r")

    raise FileNotFoundError(f"Training dataset not found. Expected one of: {', '.join(TRAINING_DATASET_PATHS)}")

def train_and_save_centroids():
    """Train centroids on the training dataset and save them."""
    training_images = load_training_images()
    print(f"Training {NUMBER_OF_CLUSTERS} centroids on {len(training_images)} images. This can take a long time.")
    centroids = calculate_centroids(training_images)

    os.makedirs(constants.CODEBOOKS_DIR, exist_ok = True)
    np.save(TRAINED_CENTROIDS_NPY, centroids)

    return centroids
#endregion

def get_terrain_subimages_dir(terrain_name):
    return os.path.join(constants.SUBIMAGES_DIR, terrain_name)

def get_terrain_vlad_vectors_path(terrain_name):
    return os.path.join(constants.CODEBOOKS_DIR, f'vlad_vectors_{terrain_name}.npy')

def prepare_terrain_subimages_and_codebooks(retrain = False):
    """Prepare terrain-specific subimages and VLAD vectors. Centroids are trained first if they are missing or if retrain is True."""
    if retrain or not os.path.exists(TRAINED_CENTROIDS_NPY):
        centroids = train_and_save_centroids()
    else:
        centroids = np.load(TRAINED_CENTROIDS_NPY)

    for terrain_name, image_name in constants.TERRAIN_IMAGE_NAMES.items():
        image_path = os.path.join(constants.ORIGINAL_IMAGES_DIR, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skipping terrain '{terrain_name}', image not found: {image_path}")
            continue

        terrain_subimages_dir = get_terrain_subimages_dir(terrain_name)
        divide_image_on_subimages_and_save(image, constants.SUBIMAGE_GRID_SIZE, terrain_subimages_dir, True)

        testing_set = get_dict_of_dataset_images(terrain_subimages_dir + os.sep)
        _, descriptors = root_sift_visual_features(list(testing_set.values()))
        mapped_descriptors = map_descriptors_to_centroids(descriptors, centroids)
        vlad_vectors = calculate_vlad_vectors(descriptors, mapped_descriptors, centroids)

        np.save(get_terrain_vlad_vectors_path(terrain_name), vlad_vectors)
        print(f"Prepared terrain '{terrain_name}': {len(testing_set)} subimages and VLAD vectors {vlad_vectors.shape}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Prepare subimages and VLAD codebooks. Run from the project's root directory.")
    parser.add_argument("--train", action = "store_true", help = "Retrain the centroids even if codebooks/trained_centroids.npy already exists.")
    prepare_terrain_subimages_and_codebooks(parser.parse_args().train)
