import numpy as np
import cv2

SIFT_EXTRACTOR = cv2.SIFT_create()

NUMBER_OF_AGENTS = 25
COMMUNICATION_RING_RANGE = 1

ORIGINAL_IMAGES_DIR = "data/original_images"
SUBIMAGES_DIR = "data/subimages"
CODEBOOKS_DIR = "codebooks"
CONTINUOUS_RESULTS_DIR = "results/continuous_est"
TRAINED_CENTROIDS_PATH = f"{CODEBOOKS_DIR}/trained_centroids.npy"
DEFAULT_VLAD_VECTORS_PATH = f"{CODEBOOKS_DIR}/vlad_vectors.npy"
SUBIMAGE_GRID_SIZE = (20, 20)
TERRAIN_IMAGE_NAMES = {
    "city": "city_bw.png",
    "desert": "desert_bw.png",
    "mountains": "mountains_bw.png",
    "rural": "rural_bw.png",
}
TERRAIN_NAMES = tuple(TERRAIN_IMAGE_NAMES.keys())

# Height 100 m, 82 deg diagonal FOV, 4:3 aspect ratio.
AGENT_SIZE = np.array([79, 63])

DS_DECISION_THRESHOLD = 10.0
DS_THRESHOLD_MIN = 0.15
DS_THRESHOLD_MAX = 101.5
