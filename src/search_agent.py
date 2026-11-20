"""One agent of the online stage: grid search in its own cell, feature selection, stop rule and re-centering.

Positions are map pixel coordinates (x, y). An agent never uses them to localize itself: its own position estimate is the map position of the
feature it matched (see SearchAgent.recenter). The simulation uses true positions only to take photos and to emulate flying.
"""

import numpy as np

import constants
import utils
from image_augmentation import NOISE_SIGMA_RANGE, RESOLUTION_FACTOR_RANGE, ZOOM_FACTOR_RANGE, augment_drone_capture

VIEW_WIDTH, VIEW_HEIGHT = (int(value) for value in constants.AGENT_SIZE)

def take_photo(image, center):
    """Take a degraded photo centered at a map position and detect its features.

    The view is kept inside the image. The photo is degraded like a drone capture with random zoom, resolution loss and noise.

    Args:
        image (numpy.ndarray): Grayscale terrain image.
        center (numpy.ndarray): Requested view center (x, y) in map pixels.

    Returns:
        dict: 'view_center' (map position of the view center), 'pixel_xy' (keypoint positions in the photo), 'map_xy' (true map positions of the
            keypoints, known only to the simulation), 'responses' and 'descriptors' (RootSIFT). The keypoint arrays are empty if no feature was
            detected.
    """
    x0 = int(np.clip(round(center[0] - VIEW_WIDTH / 2), 0, image.shape[1] - VIEW_WIDTH))
    y0 = int(np.clip(round(center[1] - VIEW_HEIGHT / 2), 0, image.shape[0] - VIEW_HEIGHT))
    zoom_factor = np.random.uniform(*ZOOM_FACTOR_RANGE)
    photo = augment_drone_capture(
        image[y0 : y0 + VIEW_HEIGHT, x0 : x0 + VIEW_WIDTH],
        zoom_factor = zoom_factor,
        resolution_factor = np.random.uniform(*RESOLUTION_FACTOR_RANGE),
        noise_sigma = np.random.uniform(*NOISE_SIGMA_RANGE),
    )
    keypoints, descriptors = constants.SIFT_EXTRACTOR.detectAndCompute(photo, None)
    view_center = np.array([x0 + VIEW_WIDTH / 2, y0 + VIEW_HEIGHT / 2])

    if descriptors is None or len(keypoints) == 0:
        return {
            "view_center": view_center,
            "pixel_xy": np.zeros((0, 2)),
            "map_xy": np.zeros((0, 2)),
            "responses": np.zeros(0),
            "descriptors": np.zeros((0, 128), dtype = np.float32),
        }

    pixel_xy = np.array([keypoint.pt for keypoint in keypoints], dtype = np.float64)
    # The digital zoom stretches the photo around its center, so a keypoint lies (pixel offset / zoom) from the view center on the map.
    map_xy = view_center + (pixel_xy - np.array([VIEW_WIDTH / 2, VIEW_HEIGHT / 2])) / zoom_factor
    return {
        "view_center": view_center,
        "pixel_xy": pixel_xy,
        "map_xy": map_xy,
        "responses": np.array([keypoint.response for keypoint in keypoints], dtype = np.float64),
        "descriptors": utils.root_sift(descriptors),
    }

def calculate_lawnmower_path(cell_bounds, sweep_axis, path_length):
    """Calculate the positions at which an agent takes photos while it sweeps its cell.

    The path starts at the (min x, min y) corner of the traversable area (the cell shrunk by half the view size, so the view stays inside the cell)
    and runs along lanes LANE_SPACING_FACTOR view sizes apart, including the lane on the far border. After the last lane it sweeps back.

    Args:
        cell_bounds (tuple[float, float, float, float]): Min x, max x, min y and max y of the cell.
        sweep_axis (str): 'horizontal' (lanes along x) or 'vertical' (lanes along y).
        path_length (float): Length of the path in pixels.

    Returns:
        numpy.ndarray: Photo positions (x, y), one every AGENT_SPEED * CAPTURE_PERIOD pixels.
    """
    min_x, max_x, min_y, max_y = cell_bounds
    min_x, max_x = min_x + VIEW_WIDTH / 2, max_x - VIEW_WIDTH / 2
    min_y, max_y = min_y + VIEW_HEIGHT / 2, max_y - VIEW_HEIGHT / 2
    step = constants.AGENT_SPEED * constants.CAPTURE_PERIOD

    if sweep_axis == "horizontal":
        primary_min, primary_max, secondary_min, secondary_max = min_x, max_x, min_y, max_y
        lane_spacing = max(VIEW_HEIGHT * constants.LANE_SPACING_FACTOR, step)
    else:
        primary_min, primary_max, secondary_min, secondary_max = min_y, max_y, min_x, max_x
        lane_spacing = max(VIEW_WIDTH * constants.LANE_SPACING_FACTOR, step)

    lanes = list(np.arange(secondary_min, secondary_max, lane_spacing))
    if not lanes or not np.isclose(lanes[-1], secondary_max):
        lanes.append(secondary_max)
    lane_order = lanes + lanes[-2:0:-1] if len(lanes) > 1 else lanes

    vertices = []
    covered_length = 0.0
    lane_index = 0
    starts_at_min = True
    while covered_length <= path_length + 2 * (primary_max - primary_min):
        lane = lane_order[lane_index % len(lane_order)]
        start, end = (primary_min, primary_max) if starts_at_min else (primary_max, primary_min)
        vertices += [(start, lane), (end, lane)]
        covered_length += (primary_max - primary_min) + lane_spacing
        starts_at_min = not starts_at_min
        lane_index += 1

    vertices = np.array(vertices, dtype = np.float64)
    if sweep_axis == "vertical":
        vertices = vertices[:, ::-1]
    segment_lengths = np.linalg.norm(np.diff(vertices, axis = 0), axis = 1)
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    distances = np.arange(0.0, min(path_length, cumulative_lengths[-1]) + 1e-9, step)
    return np.stack([np.interp(distances, cumulative_lengths, vertices[:, 0]), np.interp(distances, cumulative_lengths, vertices[:, 1])], axis = 1)

class SearchAgent:
    """Agent that sweeps its cell until its strongest feature passes the stop rule, then re-centers on it."""

    def __init__(self, agent_id, cell_bounds, sweep_axis, stop_rule, stop_threshold):
        """Initialize the agent at the start of its lawnmower path.

        Args:
            agent_id (int): Index of the agent.
            cell_bounds (tuple[float, float, float, float]): Min x, max x, min y and max y of its search cell.
            sweep_axis (str): 'horizontal' or 'vertical'.
            stop_rule (str): 'response' or 'lowe' (see constants.STOP_RULE).
            stop_threshold (float): Response threshold or Lowe ratio threshold.
        """
        if stop_rule not in constants.STOP_RULES:
            raise ValueError(f"Stop rule must be one of {constants.STOP_RULES}.")
        self.agent_id = agent_id
        self.stop_rule = stop_rule
        self.stop_threshold = stop_threshold
        self.path = calculate_lawnmower_path(cell_bounds, sweep_axis, constants.AGENT_SPEED * constants.MAX_SEARCH_TIME)
        self.path_index = 0
        self.is_searching = True
        self.is_fallback = False
        self.stop_time = None
        self.selected = None
        self.best = None
        self.true_position = None
        self.own_estimate = None
        self.memberships = None
        self.decision_time = None
        self.centering_residual = None

    @property
    def position(self):
        """Current true map position (simulation only)."""
        return self.path[min(self.path_index, len(self.path) - 1)]

    def capture(self, image):
        """Take a photo at the current position and return its strongest feature, or None if no feature was detected."""
        photo = take_photo(image, self.position)
        if len(photo["responses"]) == 0:
            return None
        strongest = int(np.argmax(photo["responses"]))
        return {
            "descriptor": photo["descriptors"][strongest],
            "response": float(photo["responses"][strongest]),
            "map_xy": photo["map_xy"][strongest],
            "lowe_ratio": None,
            "nearest_index": None,
            "time": self.path_index * constants.CAPTURE_PERIOD,
        }

    def score(self, feature):
        """Rank features for the fallback choice; lower is better."""
        return feature["lowe_ratio"] if self.stop_rule == "lowe" else -feature["response"]

    def passes(self, feature):
        """Check the stop rule for a feature."""
        if self.stop_rule == "lowe":
            return feature["lowe_ratio"] < self.stop_threshold
        return feature["response"] > self.stop_threshold

    def consider(self, feature):
        """Remember the best feature so far and stop if the feature passes the stop rule."""
        if self.best is None or self.score(feature) < self.score(self.best):
            self.best = feature
        if self.passes(feature):
            self.is_searching = False
            self.selected = feature
            self.stop_time = feature["time"]

    def advance(self):
        """Move to the next photo position. An agent at the end of its path or at the time limit stops searching and falls back on its best
        feature."""
        self.path_index += 1
        if self.path_index >= len(self.path) or self.path_index * constants.CAPTURE_PERIOD > constants.MAX_SEARCH_TIME:
            self.give_up()

    def give_up(self):
        """Stop searching and use the best feature seen so far (None if the agent never detected a feature)."""
        if self.is_searching:
            self.is_searching = False
            self.is_fallback = True
            self.selected = self.best
            self.stop_time = constants.MAX_SEARCH_TIME

    def recenter(self, terrain_map):
        """Fly so that the selected feature is under the view center, look again, and take the matched map feature's position as own position.

        The flight is emulated by moving the view center to the feature's true map position; a real agent would steer on its image until the
        feature is centered. In the new photo the feature is re-identified as the keypoint within RECENTER_RADIUS photo pixels of the view center
        whose descriptor is closest to the selected one; its distance from the view center is the centering residual.

        Args:
            terrain_map (map_matching.TerrainMap): Terrain the agent flies over.

        Returns:
            bool: False if the agent has no selected feature (it never detected one).
        """
        if self.selected is None:
            return False

        if self.selected["nearest_index"] is None:
            lowe_ratios, nearest_indices = terrain_map.match_descriptors(self.selected["descriptor"])
            self.selected["lowe_ratio"], self.selected["nearest_index"] = float(lowe_ratios[0]), int(nearest_indices[0])

        flight_distance = np.linalg.norm(self.selected["map_xy"] - self.position)
        photo = take_photo(terrain_map.image, self.selected["map_xy"])
        self.true_position = photo["view_center"]
        self.decision_time = self.stop_time + flight_distance / constants.AGENT_SPEED + constants.CAPTURE_PERIOD

        if len(photo["responses"]):
            distances_to_center = np.linalg.norm(photo["pixel_xy"] - np.array([VIEW_WIDTH / 2, VIEW_HEIGHT / 2]), axis = 1)
            near_center = np.flatnonzero(distances_to_center <= constants.RECENTER_RADIUS)
            if len(near_center):
                closest = near_center[np.argmin(np.linalg.norm(photo["descriptors"][near_center] - self.selected["descriptor"], axis = 1))]
            else:
                closest = int(np.argmin(distances_to_center))
            self.centering_residual = float(np.linalg.norm(photo["map_xy"][closest] - self.true_position) * constants.METERS_PER_PIXEL)

        self.own_estimate = terrain_map.keypoints_xy[self.selected["nearest_index"]]
        self.memberships = terrain_map.calculate_memberships(self.selected["descriptor"])
        return True
