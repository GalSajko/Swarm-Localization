import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.cluster.vq import kmeans, vq

BW_IMAGES_DIRECTORY = "bw_images/"
NUMBER_OF_CLUSTERS = 1000

def convert_to_bw():
    """Convert resized colored images to black and white images and save them in bw_images directory.
    """
    original_images_directory = "resized_images/"
    for file in os.listdir(original_images_directory):
        filename = os.fsdecode(file)
        original_relative_path = original_images_directory + filename
        image = cv2.imread(original_relative_path)
        bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(BW_IMAGES_DIRECTORY + filename, bw_image)
    
def write_images_to_list():
    """Save images from bw_images directory to list.

    Returns:
        list: List of bw images from directory.
    """
    bw_images_list = []
    for file in os.listdir(BW_IMAGES_DIRECTORY):
        filename = os.fsdecode(file)
        image = cv2.imread(BW_IMAGES_DIRECTORY + filename)
        bw_images_list.append(image)
    
    return bw_images_list

def compute_visual_features(bw_images):
    """Compute keypoints and descriptors with SIFT algorithm.

    Args:
        bw_images (list): List of images.

    Returns:
        tuple[list, list]: Keypoints and descriptors.
    """
    sift_extractor = cv2.SIFT_create()
    keypoints = []
    descriptors = []

    for image in bw_images:
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

def kmeans_clustering_over_descriptors(all_descriptors):
    """Kmeans clustering over list of all descriptors from dataset.

    Args:
        all_descriptors (numpy.ndaray): Array of all descriptors.

    Returns:
        numpy.ndarray: Array of centroids.
    """
    iters = 1
    centroids, _ = kmeans(all_descriptors, NUMBER_OF_CLUSTERS, iters)

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
        visual_words_in_image, _ = vq(descriptor, centroids)
        visual_words.append(visual_words_in_image)
    
    return visual_words

def calculate_frequency_vectors(visual_words):
    frequency_vectors = []
    for image_as_visual_words in visual_words:
        image_frequency_vector = np.zeros(NUMBER_OF_CLUSTERS)
        for visual_word in image_as_visual_words:
            image_frequency_vector[visual_word] += 1
        frequency_vectors.append(image_frequency_vector)
    
    return(np.stack(frequency_vectors))

if __name__ == "__main__":
    # bw_images = write_images_to_list()
    # keypoints, descriptors = compute_visual_features(bw_images)

    # # Write all descriptors into single array.
    # stacked_descriptors = np.vstack(descriptors)
    # # Perform kmeans clustering over the array of all descriptors and calculate centroids of the clusters.
    # codebook = kmeans_clustering_over_descriptors(stacked_descriptors)
    # # Each image is now represented with visual words. Each visual word is a list of index values of vectors from the codebook.
    # images_as_visual_words = map_descriptors_to_centroids(descriptors, codebook)
    # # Calculate how many times each word appears in each image. 
    # frequency_vectors = calculate_frequency_vectors(images_as_visual_words)

    # for vector in frequency_vectors:
    #     print(len(np.where(vector == 0)[0]))

    # plt.bar(list(range(NUMBER_OF_CLUSTERS)), frequency_vectors[1])
    # plt.show()

