"""Terrain map used by the agents online: matching of descriptors and fuzzy overlap candidates.

Everything here uses only data an agent can carry: the terrain codebook prepared by prepare_image_dataset.py (positions, RootSIFT descriptors and
fuzzy memberships of the selected map features, and the fuzzy c-means centers). The terrain image itself is loaded only to simulate the agents'
photos.
"""

import cv2
import numpy as np
import os

import constants
import fuzzy_clustering
import prepare_image_dataset

# Agent descriptors are compared with the map descriptors in chunks, so the (agents x map features) distance matrix stays small.
MATCHING_CHUNK_SIZE = 32

class TerrainMap:
    """Codebook and image of one terrain."""

    def __init__(self, terrain_name):
        """Load the codebook and the image of a terrain.

        Args:
            terrain_name (str): One of constants.TERRAIN_NAMES.
        """
        codebook_path = prepare_image_dataset.get_terrain_codebook_path(terrain_name)
        if not os.path.exists(codebook_path):
            raise FileNotFoundError(f"Codebook '{codebook_path}' not found. Run src/prepare_image_dataset.py first.")

        with np.load(codebook_path) as codebook:
            if "descriptors" not in codebook.files:
                raise KeyError(f"Codebook '{codebook_path}' has no descriptors. Rerun src/prepare_image_dataset.py (step 4 adds them).")
            self.keypoints_xy = codebook["keypoints_xy"].astype(np.float64)
            self.descriptors = np.ascontiguousarray(codebook["descriptors"], dtype = np.float32)
            self.memberships = codebook["memberships"].astype(np.float32)
            self.centers = codebook["centers"].astype(np.float32)
            self.membership_fuzzifier = float(codebook["membership_fuzzifier"])
            self.map_shape = tuple(int(value) for value in codebook["map_shape"])

        self.terrain_name = terrain_name
        self.squared_descriptor_norms = np.einsum("ij,ij->i", self.descriptors, self.descriptors)
        image_path = os.path.join(constants.ORIGINAL_IMAGES_DIR, constants.TERRAIN_IMAGE_NAMES[terrain_name])
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise FileNotFoundError(f"Terrain image '{image_path}' not found.")

    def match_descriptors(self, descriptors):
        """Find the nearest map feature of each descriptor and its Lowe ratio.

        Args:
            descriptors (numpy.ndarray): RootSIFT descriptors of shape (N, 128).

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Lowe ratios (N,), i.e. distance to the nearest map descriptor divided by the distance to the second
                nearest one, and indices (N,) of the nearest map features.
        """
        descriptors = np.atleast_2d(np.asarray(descriptors, dtype = np.float32))
        lowe_ratios = np.empty(len(descriptors), dtype = np.float64)
        nearest_indices = np.empty(len(descriptors), dtype = np.int64)

        for start in range(0, len(descriptors), MATCHING_CHUNK_SIZE):
            chunk = descriptors[start : start + MATCHING_CHUNK_SIZE]
            squared_distances = np.maximum(
                self.squared_descriptor_norms[None, :] - 2.0 * chunk @ self.descriptors.T + np.einsum("ij,ij->i", chunk, chunk)[:, None],
                0.0,
            )
            two_nearest = np.argpartition(squared_distances, 1, axis = 1)[:, :2]
            two_distances = np.take_along_axis(squared_distances, two_nearest, axis = 1)
            order = np.argsort(two_distances, axis = 1)
            two_nearest = np.take_along_axis(two_nearest, order, axis = 1)
            two_distances = np.sqrt(np.take_along_axis(two_distances, order, axis = 1))
            lowe_ratios[start : start + len(chunk)] = two_distances[:, 0] / np.maximum(two_distances[:, 1], 1e-12)
            nearest_indices[start : start + len(chunk)] = two_nearest[:, 0]

        return lowe_ratios, nearest_indices

    def calculate_memberships(self, descriptors):
        """Calculate fuzzy memberships of agent descriptors with the codebook's centers and fuzzifier."""
        return fuzzy_clustering.calculate_memberships(descriptors, self.centers, self.membership_fuzzifier)

    def find_candidates(self, memberships):
        """Find the map features whose fuzzy overlap with an agent feature exceeds constants.CANDIDATE_OVERLAP_THRESHOLD.

        The fuzzy overlap of membership vectors u and v is sum_k min(u_k, v_k): 1 for identical vectors and 0 for vectors with no cluster in common.

        Args:
            memberships (numpy.ndarray): Memberships of one agent feature, shape (C,).

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Indices of the candidate map features and their overlaps.
        """
        overlaps = np.minimum(self.memberships, np.asarray(memberships, dtype = np.float32)[None, :]).sum(axis = 1)
        candidate_indices = np.flatnonzero(overlaps > constants.CANDIDATE_OVERLAP_THRESHOLD)
        return candidate_indices, overlaps[candidate_indices].astype(np.float64)
