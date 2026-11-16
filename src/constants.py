import numpy as np
import cv2

SIFT_EXTRACTOR = cv2.SIFT_create()

NUMBER_OF_AGENTS = 25

ORIGINAL_IMAGES_DIR = "data/original_images"
CODEBOOKS_DIR = "codebooks"
FCM_CENTERS_PATH = f"{CODEBOOKS_DIR}/fcm_centers.npz"

# Number of fuzzy c-means clusters.
NUMBER_OF_CLUSTERS = 64
# Fuzzifier used to train the fuzzy c-means centers. Changing it requires retraining (prepare_image_dataset.py --train).
TRAINING_FUZZIFIER = 1.1
# Fuzzifier used to calculate memberships of map features and of features detected by agents. Changing it only requires recalculating the
# memberships (prepare_image_dataset.py --membership_m).
MEMBERSHIP_FUZZIFIER = 1.1

# Repeatability filter of map features. The whole map is degraded NUMBER_OF_DEGRADED_COPIES times like an agent's photo, and a map feature is
# kept in the codebook only if it is found again in at least MIN_REPEAT_COUNT of the copies. MIN_REPEAT_COUNT = 0 keeps all features.
NUMBER_OF_DEGRADED_COPIES = 6
MIN_REPEAT_COUNT = 5

TERRAIN_IMAGE_NAMES = {
    "city": "city_bw.png",
    "desert": "desert_bw.png",
    "mountains": "mountains_bw.png",
    "rural": "rural_bw.png",
}
TERRAIN_NAMES = tuple(TERRAIN_IMAGE_NAMES.keys())

# Height 100 m, 82 deg diagonal FOV, 4:3 aspect ratio.
AGENT_SIZE = np.array([79, 63])
