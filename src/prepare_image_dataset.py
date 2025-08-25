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
import pickle as pkl
from scipy.cluster.vq import kmeans, vq

#region DATASET PREPARATION
ORIGINAL_IMAGES_DIR = 'data/original_images/'
SUBIMAGES_DIR = 'data/subimages/'
COLORED_SATELITE_IMAGE_NAME = 'colored_satelite_image.png'
BW_SATELITE_IMAGE_NAME = 'bw_satelite_image.png'

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
            cv2.imwrite(SUBIMAGES_DIR + f"{x + 1}_{y + 1}.png", sub_image)

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
            image = cv2.imread(dir + filename)
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
NUMBER_OF_CLUSTERS = 64

def sift_visual_features(list_of_images):
    """Compute keypoints and descriptors with SIFT algorithm.

    Args:
        list_of_images (list): List of images, represented as numpy arrays.

    Returns:
        tuple[list, list]: Keypoints and descriptors.
    """
    sift_extractor = cv2.SIFT_create()
    keypoints = []
    descriptors = []

    for image in list_of_images:
        img_keypoints, img_descriptors = sift_extractor.detectAndCompute(image, None)
        keypoints.append(img_keypoints)
        descriptors.append(img_descriptors)
    
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
    iters = 1
    centroids, _ = kmeans(stacked_descriptors, NUMBER_OF_CLUSTERS, iters)

    return centroids

def map_descriptors_to_centroids(descriptors, centroids):
    """Assign descriptors to the closest centroids and save them as list of visual words.

    Args:
        descriptors (numpy.ndarray): Arrays of descriptors of images.
        centroids (numpy.ndarray): Array of centroids of clusters.

    Returns:
        numpy.ndarray: List of visual words for each image.
    """
    visual_words = []
    for descriptor in descriptors:
        # For each descriptor return index of centroid the descriptor belongs to.
        visual_words_in_image, _ = vq(descriptor, centroids)
        visual_words.append(visual_words_in_image)
    
    return visual_words
#endregion

#region BOW AND VLAD
BOW_CODEBOOK_JSON = 'codebooks/bow_codebook.json'
VISUAL_WORDS_NPY = 'codebooks/visual_words.npy'
VISUAL_WORDS_IN_IMAGES_PKL = 'codebooks/visual_words_in_images.pkl'
VISUAL_WORDS_HISTOGRAMS_NPY = 'codebooks/visual_words_histograms.npy'
VISUAL_WORDS_WEIGHTS_NPY = 'codebooks/visual_words_weights.npy'
VLAD_CENTROIDS_NPY = 'codebooks/vlad_centroids.npy'
VLAD_VECTORS_NPY = 'codebooks/vlad_vectors.npy'

def calculate_visual_words_histograms(visual_words):
    """Calculate how many times each centroid (visual word) appears in each image.

    Args:
        visual_words (list): List of visual words.
    
    Returns:
        numpy.ndarray: Array of frequency vectors for each image.
    """
    frequency_vectors = []
    for image_as_visual_words in visual_words:
        image_frequency_vector = np.zeros(NUMBER_OF_CLUSTERS)
        for visual_word in image_as_visual_words:
            image_frequency_vector[visual_word] += 1
        frequency_vectors.append(image_frequency_vector)
    
    return(np.stack(frequency_vectors))

def idf(visual_words_histograms):
    """Calculate the importance of visual words using TFIDF algorithm.

    Args:
        visual_words_histograms (numpy.ndarray): Histograms of visual words for each image.

    Returns:
        numpy.ndarray: Weights of the centroids.
    """
    number_of_images = len(visual_words_histograms)
    # Number of images that contain visual words.
    tf = np.sum(visual_words_histograms > 0, axis = 0)
    idf = np.log(number_of_images / tf)

    return idf

def calculate_vlad_vectors(descriptors, mapped_descriptors, centroids):
    """Calculate vlad descriptors for all images.

    Args:
        descriptors (numpy.ndarray): SIFT descriptors.
        mapped_descriptors (numpy.ndarray): SIFT descriptors mapped to centroids.
        centroids (numpy.ndarray): Centroids.

    Returns:
        numpy.ndarray: Vlad descriptors of all images.
    """
    vlad_vectors = np.zeros((len(descriptors), len(centroids), len(descriptors[0][0])))
    for image_index, descriptors_in_image in enumerate(descriptors):
        for descriptor_index, descriptor in enumerate(descriptors_in_image):
            centroid_index = mapped_descriptors[image_index][descriptor_index]
            difference = np.round(descriptor - centroids[centroid_index], 4)
            vlad_vectors[image_index][centroid_index] += difference
        print(vlad_vectors[image_index][0])

    return vlad_vectors


#endregion

#region PREPARE AND SAVE
def save_bow_codebook(tf_idfs, subimages_dict):
    """Save BOW tfidf codebook to JSON file.

    Args:
        tf_idfs (numpy.ndarray): Array of tfidf-weighted visual words. 
        subimages_dict (dict): Dictionary with image names and images.
    """
    codebook = {}
    for idx, tf_idf in enumerate(tf_idfs):
        image_name = list(subimages_dict.keys())[idx]
        codebook[image_name] = list(tf_idf)

    with open(BOW_CODEBOOK_JSON, 'w') as file:
        json.dump(codebook, file)

def prepare_and_save_bow_codebook(subimages):
    """Prepare bag-of-words tfidf-weighted codebook and save it as JSON file.

    Args:
        subimages (dict): Dictionary of images and their filenames.
    """
    _, descriptors = sift_visual_features(list(subimages.values()))
    visual_words = kmeans_over_descriptors(np.vstack(descriptors))
    visual_words_in_images = map_descriptors_to_centroids(descriptors, visual_words)

    visual_words_histograms = calculate_visual_words_histograms(visual_words_in_images)
    weights = idf(visual_words_histograms)

    tf_idf = visual_words_histograms * weights

    save_bow_codebook(tf_idf, subimages)

    np.save(VISUAL_WORDS_WEIGHTS_NPY, weights)
    np.save(VISUAL_WORDS_NPY, visual_words)
    np.save(VISUAL_WORDS_HISTOGRAMS_NPY, visual_words_histograms)  
    with open(VISUAL_WORDS_IN_IMAGES_PKL, 'wb') as file:
        pkl.dump(visual_words_in_images, file)

def prepare_and_save_vlad_codebook(subimages):
    """Prepare and save VLAD descriptors of database images.

    Args:
        subimages (numpy.ndarray): Subimages.
    """
    keypoints, descriptors = sift_visual_features(list(subimages.values()))
    centroids = kmeans_over_descriptors(np.vstack(descriptors))
    mapped_descriptors = map_descriptors_to_centroids(descriptors, centroids)
    vlad_vectors = calculate_vlad_vectors(descriptors, mapped_descriptors, centroids)

    np.save(VLAD_CENTROIDS_NPY, centroids)
    np.save(VLAD_VECTORS_NPY, vlad_vectors)
#endregion

if __name__ == '__main__':
    if is_correct_dir():
        # divide_image_on_subimages_and_save(cv2.imread(ORIGINAL_IMAGES_DIR + BW_SATELITE_IMAGE_NAME), (5, 3), True)
        subimages = get_dict_of_dataset_images(SUBIMAGES_DIR)
        
        # prepare_and_save_bow_codebook(subimages)
        prepare_and_save_vlad_codebook(subimages)

    else:
        print(f"Call script from project's root directory: {PROJECT_ROOT_DIR}")
    
