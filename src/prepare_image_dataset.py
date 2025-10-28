"""This module contains code for segmentation of the original image on set of subimages. Original image and subimages are saved in 'data/' directory. Besides,
it uses subimages to calculate BOW codebook, which is saved as JSON file in 'codebooks/' directory.
"""

import cv2
import glob
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.cluster.vq import kmeans, vq

import constants
import utils

#region DATASET PREPARATION
ORIGINAL_IMAGES_DIR = 'data/original_images/'
SUBIMAGES_DIR = 'data/subimages/'
COLORED_SATELITE_IMAGE_NAME = 'colored_satelite_image.png'
BW_SATELITE_IMAGE_NAME = 'bw_satelite_image.png'

TRAINING_DATASET = 'training_data/ims_gray.npy'

PROJECT_ROOT_DIR = 'Swarm-Localization'

def convert_image_to_bw_and_save(original_image_path, original_image_name, path_to_save = None):
    """Convert given image in black and white and save it into desired directory.

    Args:
        original_image_path (string): Path of the original image.
        original_image_name (string): Name of the original image
        path_to_save (string, optional): Path to the desired directory of the converted image. 
        If None, converted image will be saved in the same directory as original one. Defaults to None.
    """
    image = cv2.imread(original_image_path + original_image_name)
    bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if not path_to_save:
        path_to_save = original_image_path
    
    cv2.imwrite(path_to_save + BW_SATELITE_IMAGE_NAME, bw_image)

def divide_image_on_subimages_and_save(image, divide_on, remove_old_files = True):
    """Divide original image on desired number of subimages and save them in subimages directory.

    Args:
        image (numpy.ndarray): Original image as numpy array.
        divide_on (tuple[int, int]): Number of subimages in horizontal and vertical direction.
        remove_old_files(bool, optional): If True, remove previous image files in the directory. Defaults to True.
    """
    if remove_old_files:
        remove_files_from_dir(SUBIMAGES_DIR)

    shape = image.shape
    
    subimage_x_size = math.floor(shape[1] / divide_on[0])
    subimage_y_size = math.floor(shape[0] / divide_on[1])
    
    for y in range(divide_on[1]):
        for x in range(divide_on[0]):
            sub_image = image[y * subimage_y_size : (y + 1) * subimage_y_size, x * subimage_x_size : (x + 1) * subimage_x_size]
            first_index = x + 1 if x + 1 >= 10 else '0' + str(x + 1)
            second_index = y + 1 if y + 1 >= 10 else '0' + str(y + 1)
            cv2.imwrite(SUBIMAGES_DIR + f"{first_index}_{second_index}.png", sub_image)

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
    files = glob.glob(dir + '*')
    for file in files:
        if os.path.basename(file) == '.gitkeep':
            continue
        os.remove(file)

def is_correct_dir():
    """Check if script is called from correct directory. Correct directory is Swarm-Localization/.

    Returns:
        bool: True, if directory is correct, False otherwise.
    """
    current_dir = os.getcwd()
    last_dir = current_dir.split("/")[-1]

    return last_dir == PROJECT_ROOT_DIR
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
    sift_extractor = constants.SIFT_EXCTRACTOR
    keypoints = []
    descriptors = []

    for image in list_of_images:
        img_keypoints, img_descriptors = sift_extractor.detectAndCompute(image, None)
        if img_descriptors is None:
            img_descriptors = np.empty((constants.SIFT_EXCTRACTOR.descriptorSize(), 128), dtype = np.float32)
        root_sift_descriptors = utils.root_sift(img_descriptors)
        keypoints.append(img_keypoints)
        descriptors.append(root_sift_descriptors)
    
    return keypoints, descriptors

def show_sift_keypoints(image, keypoints):
    """Show sift keypoints on image.

    Args:
        image (numpy.ndarray): Image.
        keypoints (list): Keypoints of the given image.
    """
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, 0, (255, 0, 0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(image_with_keypoints, cmap = "gray")
    plt.show()

def kmeans_over_descriptors(stacked_descriptors):
    """Kmeans clustering over all descriptors from dataset.

    Args:
        all_descriptors (numpy.ndaray): Array of all descriptors.

    Returns:
        numpy.ndarray: Array of centroids.
    """
    iters = 20
    centroids, _ = kmeans(stacked_descriptors, NUMBER_OF_CLUSTERS, iters)

    return centroids.astype(np.float32, copy = False)

def calculate_centroids(dataset):
    _, descriptors_in_images = root_sift_visual_features(dataset)
    descriptors_in_images = descriptors_in_images[descriptors_in_images != None]
    centroids = kmeans_over_descriptors(np.vstack(descriptors_in_images))

    return centroids

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
        # For each descriptor return index of centroid the descriptor belongs to.
        visual_words_in_image, _ = vq(descriptor, centroids)
        mapped_descriptors.append(visual_words_in_image)
    
    return mapped_descriptors
#endregion

#region VLAD
VLAD_VECTORS_NPY = 'codebooks/vlad_vectors.npy'
TRAINED_CENTROIDS_NPY = 'codebooks/trained_centroids.npy'

def calculate_vlad_vectors(descriptors, mapped_descriptors, centroids, normalize = True):
    """Calculate vlad descriptors for all images.

    Args:
        descriptors (numpy.ndarray): SIFT descriptors.
        mapped_descriptors (numpy.ndarray): SIFT descriptors mapped to centroids.
        centroids (numpy.ndarray): Centroids.

    Returns:
        numpy.ndarray: Vlad descriptors of all images.
    """
    vlad_vectors = np.zeros((len(descriptors), len(centroids), len(descriptors[0][0])), dtype = np.float32)    
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
def prepare_and_save_vlad_codebook(training_set, testing_set, training = False):
    """Prepare and save VLAD descriptors of database images.

    Args:
        subimages (numpy.ndarray): Subimages.
    """
    if training:
        centroids = calculate_centroids(training_set)
        np.save(TRAINED_CENTROIDS_NPY, centroids)
    else:
        centroids = np.load(TRAINED_CENTROIDS_NPY)

    _, descriptors = root_sift_visual_features(list(testing_set.values()))
    mapped_descriptors = map_descriptors_to_centroids(descriptors, centroids)
    vlad_vectors = calculate_vlad_vectors(descriptors, mapped_descriptors, centroids)

    np.save(VLAD_VECTORS_NPY, vlad_vectors)
#endregion

def create_similarity_json(codebook, testing_set):
    """Create JSON file with filenames of 5 most similar images for each image in the dataset.

    Args:
        codebook (numpy.ndarray): Codebook of visual words.
        testing_set (dict): Dictionary of images in the dataset.
    """
    filenames = list(testing_set.keys())
    similarity_dict = {}
    for i, filename in enumerate(filenames):
        similarity_vector = utils.evaluate_similarity(codebook[i], codebook)
        ten_best_indices = list(np.argsort(similarity_vector)[::-1][1:51])
        ten_best_filenames = [filenames[index] for index in ten_best_indices]
        similarity_dict[filename] = {fn : float(similarity_vector[ten_best_indices[i]]) for i, fn in enumerate(ten_best_filenames)}

    with open('data/similarity_table.json', 'w') as f:
        json.dump(similarity_dict, f, indent = 4)

if __name__ == '__main__':
    if is_correct_dir():
        # divide_image_on_subimages_and_save(cv2.imread(ORIGINAL_IMAGES_DIR + BW_SATELITE_IMAGE_NAME, cv2.IMREAD_GRAYSCALE), (20, 20), True)

        training_set = np.load(TRAINING_DATASET)
        testing_set = get_dict_of_dataset_images(SUBIMAGES_DIR)

        create_similarity_json(np.load(VLAD_VECTORS_NPY), testing_set)

        # prepare_and_save_vlad_codebook(training_set, testing_set, True)

    else:
        print(f"Call script from project's root directory: {PROJECT_ROOT_DIR}")

    
        
    
