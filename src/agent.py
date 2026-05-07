from collections import Counter
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.cluster.vq import vq
import sys
import threading
import time

import constants
import decision_making as dm
import image_augmentation as ia
import utils

class Agent:
    def __init__(self, original_image, centroids, codebook, start_position, id, dm_method, speed_up_factor, movement_type = constants.RANDOM_MOVEMENT, search_bounds = None):
        self.id = id 
        self.dm_method = dm_method
        self.speed_up_factor = speed_up_factor
        self.movement_type = movement_type
        self.search_bounds = search_bounds

        self.do_stop = False

        self.position_locker = threading.Lock()
        self.position = start_position

        self.centroids = centroids
        self.codebook = codebook
        self.descriptors = []

        self.sift_extractor = constants.SIFT_EXTRACTOR
        self.original_image = original_image

        self.state = constants.EXPLORATION_STATE
        self.state_locker = threading.Lock()
        self.exploration_state_counter = 0

        # Velocity gain and exploration period specifies how many captures the agent makes per meter of traveled path.
        # velocity_gain = 0.085 corresponds to cca 15m/s real-time velocity over given sub-image, exploration_period = 0.5 means that
        # agent captures the surface twice per real-time second.
        # Given those values, agents captures the surface on cca 7.5 meters of traveled path. 
        self.exploration_period = 0.5 / self.speed_up_factor
        self.velocity_gain = 0.085 * self.speed_up_factor
        self.receiving_time = constants.RECEIVING_TIME / self.speed_up_factor
        self.max_dissemination_time = constants.MAX_DISSEMINATION_TIME / self.speed_up_factor
        self.movement_period = 0.01
        self.state_polling_period = 0.1 / self.speed_up_factor

        # Variables for keeping data used in decision making applicable to both methods.
        self.social_information = {} if self.dm_method == constants.ASODM_METHOD else []
        self.opinions = np.zeros(len(self.codebook)) if self.dm_method == constants.ASODM_METHOD else {}

        # aSODM variables
        self.personal_information = np.zeros(len(self.codebook))
        self.personal_information_locker = threading.Lock()
        self.social_factor = 2.0
        self.decision_threshold = np.ones_like(self.opinions) * constants.ASODM_DECISION_THRESHOLD if self.dm_method == constants.ASODM_METHOD else constants.DS_DECISION_THRESHOLD

        if self.dm_method == constants.DEMPSER_SHAFER_METHOD:
            self.theta_set = set(range(len(self.codebook)))

        # For results only
        self.max_opinions = []

        # Opinion is defined as name of the subimage.
        self.decision = None
        self.decision_locker = threading.Lock()

        self.vlad_vectors = np.zeros((len(self.centroids), len(self.centroids[0])))

        self.grid_mode = "move_to_corner"
        self.sweep_axis = np.random.choice(["horizontal", "vertical"])
        self.primary_direction = np.random.choice([-1.0, 1.0])
        self.secondary_direction = np.random.choice([-1.0, 1.0])
        self.current_lane_coord = None
        self.target_lane_coord = None
        self.border_anchor = None
        self.target_corner = self.calculate_closest_cell_corner() if self.movement_type == constants.GRID_SEARCH_MOVEMENT else None

        self.move_thread = self.move_in_thread()
        self.decision_making_thread = self.start_decision_making_thread()

    def get_position(self):
        """Get current position in a thread-safe way.

        Returns:
            numpy.ndarray: [x, y] position.
        """
        with self.position_locker:
            return self.position
    
    def set_position(self, value):
        """Set position in a thread-safe way.

        Args:
            value (numpy.ndarray): [x, y] position.
        """
        with self.position_locker:
            self.position = value
    
    def get_personal_information(self):
        """Get current value of personal information in a thread-safe way.

        Returns:
            list: Personal information.
        """
        with self.personal_information_locker:
            return self.personal_information
    
    def set_personal_information(self, value):
        """Set personal information in a thread-safe way.

        Args:
            value (list): Private information.
        """
        with self.personal_information_locker:
            self.personal_information = value
    
    def calculate_decision(self):
        """Get current decision in a thread-safe way.

        Returns:
            string: Current decision.
        """
        with self.decision_locker:
            return self.decision
    
    def get_decision(self):
        """Get current decision in a thread-safe way.

        Returns:
            string: Current decision.
        """
        with self.decision_locker:
            return self.decision
        
    def set_decision(self, value):
        """Set current decision in a thread-safe way.

        Args:
            value (string): New decision.
        """
        with self.decision_locker:
            self.decision = value
    
    def get_state(self):
        """Get agent's current state in a thread-safe way.

        Returns:
            string: Current state.
        """
        with self.state_locker:
            return self.state
        
    def set_state(self, value):
        """Set agent's current state in a thread-safe way.

        Args:
            value (string): Desired state.
        """
        if value not in (constants.EXPLORATION_STATE, constants.DISSEMINATION_STATE, constants.DECISION_MAKING_STATE, constants.RECEIVING_STATE):
            return
        with self.state_locker:
            self.state = value
    
    def capture_photo(self):
        """Capture a patch centered at the agent's position with size constants.AGENT_SIZE and apply gaussian noise over it.

        Returns:
            numpy.ndarray: Cropped image patch.
        """
        current_position = self.get_position()

        photo = self.original_image[
            int(current_position[1] - constants.AGENT_SIZE[0] / 2) : int(current_position[1] + constants.AGENT_SIZE[0] / 2),
            int(current_position[0] - constants.AGENT_SIZE[1] / 2) : int(current_position[0] + constants.AGENT_SIZE[1] / 2)
            ]
        
        try:
            # photo = ia.pixelate_image(photo, (2/3) * constants.AGENT_SIZE)
            photo = ia.augment_drone_capture(photo, randomize=True)
        except Exception:
            pass
        
        return photo
    
    def calculate_random_velocity(self):
        """Calculate random velocity for the agent.

        Returns:
            numpy.ndarray: Velocity vector in x and y directions.
        """
        x = np.random.uniform(low = -1.0, high = 1.0)
        y = np.random.choice([-1, 1]) * np.sqrt(1.0 - x**2)

        return np.array([x, y], dtype = np.float32) * self.velocity_gain

    def get_cell_traversable_bounds(self):
        """Return center-position bounds that keep the agent inside its assigned cell.

        Returns:
            tuple[float, float, float, float]: Min x, max x, min y and max y.
        """
        if self.search_bounds is None:
            return (
                constants.AGENT_SIZE[0] / 2,
                self.original_image.shape[1] - constants.AGENT_SIZE[0] / 2,
                constants.AGENT_SIZE[1] / 2,
                self.original_image.shape[0] - constants.AGENT_SIZE[1] / 2,
            )

        cell_min_x, cell_max_x, cell_min_y, cell_max_y = self.search_bounds
        min_x = cell_min_x + constants.AGENT_SIZE[0] / 2
        max_x = cell_max_x - constants.AGENT_SIZE[0] / 2
        min_y = cell_min_y + constants.AGENT_SIZE[1] / 2
        max_y = cell_max_y - constants.AGENT_SIZE[1] / 2

        if min_x > max_x:
            center_x = (cell_min_x + cell_max_x) / 2
            min_x = center_x
            max_x = center_x
        if min_y > max_y:
            center_y = (cell_min_y + cell_max_y) / 2
            min_y = center_y
            max_y = center_y

        return min_x, max_x, min_y, max_y

    def calculate_closest_cell_corner(self):
        """Calculate the closest traversable corner of the assigned search cell.

        Returns:
            numpy.ndarray: Closest corner position.
        """
        min_x, max_x, min_y, max_y = self.get_cell_traversable_bounds()
        corners = [
            np.array([min_x, min_y], dtype = np.float32),
            np.array([max_x, min_y], dtype = np.float32),
            np.array([min_x, max_y], dtype = np.float32),
            np.array([max_x, max_y], dtype = np.float32),
        ]

        return min(corners, key = lambda corner: np.linalg.norm(self.get_position() - corner))

    def initialize_grid_sweep(self):
        """Initialize lawnmower movement after reaching the assigned cell corner."""
        min_x, max_x, min_y, max_y = self.get_cell_traversable_bounds()
        cell_center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2], dtype = np.float32)
        current_position = self.get_position()

        if self.sweep_axis == "horizontal":
            self.current_lane_coord = current_position[1]
            self.primary_direction = 1.0 if current_position[0] <= cell_center[0] else -1.0
            self.secondary_direction = 1.0 if current_position[1] <= cell_center[1] else -1.0
        else:
            self.current_lane_coord = current_position[0]
            self.primary_direction = 1.0 if current_position[1] <= cell_center[1] else -1.0
            self.secondary_direction = 1.0 if current_position[0] <= cell_center[0] else -1.0

        self.grid_mode = "sweep"

    def calculate_next_grid_lane(self, secondary_min, secondary_max, lane_step):
        """Calculate the next lawnmower lane, including border lanes.

        Args:
            secondary_min (float): Minimum coordinate on the secondary axis.
            secondary_max (float): Maximum coordinate on the secondary axis.
            lane_step (float): Distance between neighboring sweep lanes.

        Returns:
            float: Coordinate of the next secondary-axis lane.
        """
        candidate_lane_coord = self.current_lane_coord + self.secondary_direction * lane_step

        if candidate_lane_coord > secondary_max:
            if not np.isclose(self.current_lane_coord, secondary_max):
                return secondary_max
            self.secondary_direction = -1.0
            return max(secondary_min, self.current_lane_coord + self.secondary_direction * lane_step)

        if candidate_lane_coord < secondary_min:
            if not np.isclose(self.current_lane_coord, secondary_min):
                return secondary_min
            self.secondary_direction = 1.0
            return min(secondary_max, self.current_lane_coord + self.secondary_direction * lane_step)

        return candidate_lane_coord

    def wait_for_scaled_duration(self, duration):
        """Wait for a scaled state duration while allowing the agent to stop.

        Args:
            duration (float): State duration in real seconds after speed-up scaling.
        """
        start_time = time.time()
        while not self.do_stop:
            loop_start_time = time.time()
            elapsed_time = loop_start_time - start_time
            remaining_time = duration - elapsed_time
            if remaining_time <= 0:
                return

            utils.sleep_with_adjustable_period(
                min(self.state_polling_period, remaining_time),
                loop_start_time,
            )

    def update_grid_search_position(self):
        """Update position using cell-entry and lawnmower grid-search movement."""
        current_position = self.get_position()
        step_distance = self.velocity_gain

        if self.grid_mode == "move_to_corner":
            direction = self.target_corner - current_position
            distance = np.linalg.norm(direction)
            if distance <= step_distance:
                self.set_position(self.target_corner.copy())
                self.initialize_grid_sweep()
                return

            self.set_position(current_position + direction / max(distance, 1e-12) * step_distance)
            return

        min_x, max_x, min_y, max_y = self.get_cell_traversable_bounds()
        if self.sweep_axis == "horizontal":
            primary_index = 0
            secondary_index = 1
            primary_min, primary_max = min_x, max_x
            secondary_min, secondary_max = min_y, max_y
            lane_step = max(constants.AGENT_SIZE[1] * 0.8, step_distance)
        else:
            primary_index = 1
            secondary_index = 0
            primary_min, primary_max = min_y, max_y
            secondary_min, secondary_max = min_x, max_x
            lane_step = max(constants.AGENT_SIZE[0] * 0.8, step_distance)

        if self.grid_mode == "shift_lane":
            target_lane_coord = self.current_lane_coord if self.target_lane_coord is None else self.target_lane_coord
            lane_delta = target_lane_coord - current_position[secondary_index]

            if abs(lane_delta) <= step_distance:
                self.current_lane_coord = target_lane_coord
                self.target_lane_coord = None
                self.border_anchor = None
                self.grid_mode = "sweep"
                next_secondary = self.current_lane_coord
            else:
                next_secondary = current_position[secondary_index] + np.sign(lane_delta) * step_distance

            next_primary = current_position[primary_index] if self.border_anchor is None else self.border_anchor
            next_position = current_position.copy()
            next_position[primary_index] = next_primary
            next_position[secondary_index] = next_secondary
            self.set_position(next_position.astype(np.float32))
            return

        next_primary = current_position[primary_index] + self.primary_direction * step_distance
        hit_border = next_primary >= primary_max or next_primary <= primary_min

        if hit_border:
            next_primary = primary_max if self.primary_direction > 0 else primary_min
            next_lane_coord = self.calculate_next_grid_lane(secondary_min, secondary_max, lane_step)
            self.primary_direction *= -1
            self.border_anchor = next_primary

            if not np.isclose(next_lane_coord, self.current_lane_coord):
                self.target_lane_coord = next_lane_coord
                self.grid_mode = "shift_lane"
            else:
                self.current_lane_coord = next_lane_coord

        next_position = current_position.copy()
        next_position[primary_index] = next_primary
        next_position[secondary_index] = self.current_lane_coord
        self.set_position(next_position.astype(np.float32))

    def move_in_thread(self):
        """Agent movement in separate thread.
        """
        def move():
            velocity = self.calculate_random_velocity()
            start_time, final_time = utils.set_timer(5.0 / self.speed_up_factor, 20.0 / self.speed_up_factor)

            while True:
                loop_start_time = time.time()
                current_position = self.get_position()

                if self.movement_type == constants.GRID_SEARCH_MOVEMENT:
                    self.update_grid_search_position()
                else:
                    if self.position[0] + constants.AGENT_SIZE[0] / 2 >= self.original_image.shape[1] or self.position[0] - constants.AGENT_SIZE[0] <= 0:
                        velocity[0] *= -1
                    if self.position[1] + constants.AGENT_SIZE[1] / 2 >= self.original_image.shape[0] or self.position[1] - constants.AGENT_SIZE[1] <= 1:
                        velocity[1] *= -1

                    elapsed_time = time.time() - start_time
                    if elapsed_time > final_time:
                        start_time, final_time = utils.set_timer(5.0 / self.speed_up_factor, 20.0 / self.speed_up_factor)
                        velocity = self.calculate_random_velocity()
                    
                    self.set_position(current_position + velocity)

                if self.do_stop:
                    return 
                
                utils.sleep_with_adjustable_period(self.movement_period, loop_start_time)

        move_thread = threading.Thread(target = move, name = 'move_thread_' + self.id, daemon = True)
        move_thread.start()

        return move_thread

    def start_exploration_thread(self):
        """Start thread for exploring the environment.

        Returns:
            Thread: Thread object.
        """
        def exploration():
            """Take photos and calculate sift descriptors.
            """
            descriptor_buffer_length = 1
            while not self.do_stop:
                loop_start_time = time.time()
                photo = self.capture_photo()
                self.update_descriptors(photo)
                
                if len(self.descriptors) >= descriptor_buffer_length:
                    self.calculate_vlad_vector()
                    cosine_similarity = utils.evaluate_similarity(self.vlad_vectors, self.codebook)

                    if self.dm_method == constants.ASODM_METHOD:
                        personal_information = dm.calculate_personal_information(cosine_similarity)
                    elif self.dm_method == constants.DEMPSER_SHAFER_METHOD:
                        personal_information = utils.softmax_over_vector(cosine_similarity)
                    
                    self.set_personal_information(personal_information)
                    self.descriptors = []
                
                utils.sleep_with_adjustable_period(self.exploration_period, loop_start_time)

        exploration_thread = threading.Thread(target = exploration, name = 'photo_thread_' + self.id, daemon = True)
        exploration_thread.start()
        
        return exploration_thread
    
    # TODO: Move to another file.
    def update_descriptors(self, photo):
        """Update list of local descriptors.

        Args:
            photo (numpy.ndarray): PHoto taken by the agent.
        """
        try:
            _, sift_descriptors = self.sift_extractor.detectAndCompute(photo.astype(np.uint8), None)
            root_sift_descriptors = utils.root_sift(sift_descriptors)
        except cv2.error:
            return
        try:
            for descriptor in root_sift_descriptors:
                self.descriptors.append(descriptor)
        except TypeError:
            return
        
    def calculate_vlad_vector(self):
        """Calculate VLAD vector.
        """
        if not self.descriptors:
            return
        mapped_descriptors, _ = vq(self.descriptors, self.centroids)

        for descriptor_index, descriptor in enumerate(self.descriptors):
            centroid_index = mapped_descriptors[descriptor_index]
            residual = np.array(descriptor - self.centroids[centroid_index], dtype = np.float32)
            self.vlad_vectors[centroid_index] += residual

        # Intra-normalization
        eps = 1e-12
        norms = np.linalg.norm(self.vlad_vectors, axis = 1, keepdims = True)
        self.vlad_vectors /= np.maximum(norms, eps)
        # Global L2 normalization
        norms = np.linalg.norm(self.vlad_vectors, axis = 1)
        mask = norms > eps
        self.vlad_vectors[mask] /= norms[mask, None]
        
    def start_decision_making_thread(self):
        """Thread for updating agent's opinion, based on private and social information.
        """
        def decision_making_in_loop():
            exploration_thread = self.start_exploration_thread()
            while not self.do_stop:
                # Receive social information. Method for collecting social information is called from the simulation module, because it needs access to other Agents objects. 
                # Here we only set state and reset social information buffer.
                self.social_information = {} if self.dm_method == constants.ASODM_METHOD else []
                self.set_state(constants.RECEIVING_STATE)
                self.wait_for_scaled_duration(self.receiving_time)

                # Decision making, based on personal and social information
                self.set_state(constants.DECISION_MAKING_STATE)
                
                if self.dm_method == constants.ASODM_METHOD:
                    received_opinions = np.array(list(self.social_information.values()))
                    new_opinion = dm.asodm(self.get_personal_information(), self.social_factor, received_opinions, len(self.codebook), self.decision_threshold)
                    self.opinions += new_opinion
                elif self.dm_method == constants.DEMPSER_SHAFER_METHOD:
                    self.opinions = dm.dempster_shafer(self.get_personal_information(), self.opinions, self.social_information, self.theta_set)

                self.set_decision(self.calculate_decision())

                # Disseminate 
                self.set_state(constants.DISSEMINATION_STATE)
                dissemination_time = self.get_dissemination_time()
                self.wait_for_scaled_duration(dissemination_time)

                # For results only
                self.max_opinions.append(np.argmax(self.opinions))

            exploration_thread.join()

        decision_making_thread = threading.Thread(target = decision_making_in_loop, name = 'decision_making_thread_' + self.id, daemon = True)
        decision_making_thread.start()

        return decision_making_thread
                
    def collect_opinions(self, agents):
        """Receive opinions from the neighbours that are currently in dissemination state.

        Args:
            agents (list): List of all Agent objects.
        """
        for agent in agents:
            if agent.id == self.id:
                continue
            if np.linalg.norm(self.get_position() - agent.get_position()) <= constants.COMMUNICATION_RANGE:
                if agent.get_state() == constants.DISSEMINATION_STATE:
                    agent_information = agent.opinions
                    if len(agent_information):
                        if self.dm_method == constants.ASODM_METHOD:
                            self.social_information[agent.id] = agent_information 
                        elif self.dm_method == constants.DEMPSER_SHAFER_METHOD:
                            self.social_information.append(agent_information)
    
    def calculate_decision(self):
        """Calculate current decision based on opinions.

        Returns:
            int: Id of the decided subimage or None if no decision has been made yet.
        """
        if self.dm_method == constants.ASODM_METHOD:
            if np.max(self.opinions) >= self.decision_threshold[np.argmax(self.opinions)]:
                return np.argmax(self.opinions)
            return None
        elif self.dm_method == constants.DEMPSER_SHAFER_METHOD:
            max_belief = -1
            best_singleton = None
            for hyp, belief in self.opinions.items():
                if len(hyp) == 1 and belief > max_belief:
                    max_belief = belief
                    best_singleton = hyp
            if best_singleton is not None:
                if max_belief > self.decision_threshold:
                    return list(best_singleton)[0]
            return None

        return None
    
    def get_dissemination_time(self):
        """Calculate dissemination time, based on current opinion.

        Returns:
            float: Dissemination time in seconds.
        """
        if self.dm_method == constants.ASODM_METHOD:
            decision_threshold = self.decision_threshold[np.argmax(self.opinions)]
            dissemination_time = self.max_dissemination_time * np.max(self.opinions) / decision_threshold
            if dissemination_time > self.max_dissemination_time:
                dissemination_time = self.max_dissemination_time
        elif self.dm_method == constants.DEMPSER_SHAFER_METHOD:
            if not self.opinions:
                return self.max_dissemination_time
            max_belief = max(self.opinions.values())
            dissemination_time = self.max_dissemination_time * max_belief

        return dissemination_time
    
    def stop_threads(self):
        """Stop all running threads.
        """
        self.do_stop = True
        self.move_thread.join()
        self.decision_making_thread.join()
