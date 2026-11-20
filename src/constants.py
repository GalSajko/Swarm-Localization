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

# Selection of map features for the codebook. The whole map is degraded NUMBER_OF_DEGRADED_COPIES times like an agent's photo, and every map
# feature gets a repeat count (in how many copies it is found again). A window of the agent's view size then slides over the map with a step of
# 1/WINDOW_STEPS of its size, and in every window the feature with the highest repeat count wins (ties broken by the highest SIFT response). The
# codebook keeps the features that win at least one window.
NUMBER_OF_DEGRADED_COPIES = 6
WINDOW_STEPS = 5

TERRAIN_IMAGE_NAMES = {
    "city": "city_bw.png",
    "desert": "desert_bw.png",
    "mountains": "mountains_bw.png",
    "rural": "rural_bw.png",
}
TERRAIN_NAMES = tuple(TERRAIN_IMAGE_NAMES.keys())

# Height 100 m, 82 deg diagonal FOV, 4:3 aspect ratio.
AGENT_SIZE = np.array([79, 63])
METERS_PER_PIXEL = 2.0

# Online stage: grid search movement of the main branch. The search area is a near-square grid of cells, one per agent, centered on the drop
# point. A cell is half a tile of the TILE_GRID_SIZE tile grid. Each agent sweeps its cell in lanes LANE_SPACING_FACTOR view sizes apart.
TILE_GRID_SIZE = (20, 20)
LANE_SPACING_FACTOR = 0.8
# Pixels per simulation second (17 m/s), seconds between photos, and the time limit of the search.
AGENT_SPEED = 8.5
CAPTURE_PERIOD = 0.5
MAX_SEARCH_TIME = 400.0

# Online stage: stop rule. In every photo the agent takes its strongest feature (highest SIFT response) and stops if
#   'response': the response is above RESPONSE_THRESHOLD, or
#   'lowe':     the Lowe ratio of its descriptor against the map descriptors (nearest / second nearest distance) is below LOWE_RATIO_THRESHOLD.
# RESPONSE_THRESHOLD is the 90th percentile of the strongest response in random agent photos, pooled over the four terrains.
STOP_RULES = ("response", "lowe")
STOP_RULE = "response"
RESPONSE_THRESHOLD = 0.0633
LOWE_RATIO_THRESHOLD = 0.7
# After re-centering, the selected feature is looked for within this many photo pixels of the view center.
RECENTER_RADIUS = 6.0

# Online stage: pattern vote. Map features whose fuzzy overlap with an agent feature exceeds CANDIDATE_OVERLAP_THRESHOLD are its candidates. Votes
# of different agents agree if they lie within MATCHING_RADIUS pixels.
CANDIDATE_OVERLAP_THRESHOLD = 0.5
MATCHING_RADIUS = 5.0

# Evaluation: an agent's position estimate is correct if it lies within this many meters of its true position.
CORRECT_POSITION_TOLERANCE = 10.0
FCM_RESULTS_DIR = "results/fcm_patterns"
