import cv2
import numpy as np
from scipy.cluster.vq import vq
import threading
import time

import constants
import decision_making as dm
import image_augmentation as ia
import utils

class Agent:
    def __init__(
        self,
        original_image,
        centroids,
        codebook,
        start_position,
        id,
        speed_up_factor,
        search_bounds = None,
        use_fixed_threshold = False,
        use_yager_social_fusion = False,
        use_laplacian_consensus = False,
        fixed_neighbor_ids = None,
        threshold_value = None,
        trial_start_time = None,
    ):
        self.id = id 
        self.speed_up_factor = speed_up_factor
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
        self.min_dissemination_time = constants.MIN_DISSEMINATION_TIME / self.speed_up_factor
        self.movement_period = 0.01
        self.state_polling_period = 0.1 / self.speed_up_factor
        self.laplacian_cycle_period = constants.LAPLACIAN_COMMUNICATION_PERIOD / self.speed_up_factor
        self.trial_start_time = time.time() if trial_start_time is None else trial_start_time
        self.startup_release_time = time.time()

        self.social_information = []
        self.social_best_hypotheses = []
        self.social_thresholds = []
        self.social_decision_flags = []
        self.social_consensus_positions = []
        self.social_consensus_confidences = []
        self.social_consensus_frozen_flags = []
        self.received_neighbor_ids = set()
        self.opinions = {}
        self.ds_scores = np.zeros(len(self.codebook), dtype = np.float32)
        self.personal_information = np.zeros(len(self.codebook))
        self.personal_information_locker = threading.Lock()
        self.decision_threshold = constants.DS_DECISION_THRESHOLD if threshold_value is None else float(threshold_value)
        self.use_fixed_threshold = use_fixed_threshold
        self.use_yager_social_fusion = use_yager_social_fusion
        self.use_laplacian_consensus = use_laplacian_consensus
        self.fixed_neighbor_ids = set() if fixed_neighbor_ids is None else {str(neighbor_id) for neighbor_id in fixed_neighbor_ids}
        self.dynamic_threshold = self.decision_threshold
        self.threshold_locker = threading.Lock()
        self.theta_set = set(range(len(self.codebook)))
        self.consensus_position = None
        self.previous_consensus_position = None
        self.consensus_initialized = False
        self.consensus_stable_counter = 0
        self.laplacian_frozen = False
        self.laplacian_done = False
        self.laplacian_done_counter = 0
        self.frozen_consensus_confidence = None
        self.consensus_locker = threading.Lock()
        self.grid_shape = dm._infer_grid_shape(len(self.codebook))

        # For results only
        self.max_opinions = []
        self.decision_cycle_index = 0
        self.first_decision_time = None
        self.first_decision_cycle = None
        self.decision_dynamics = []
        self.best_decision_history = []

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
        self.target_corner = self.calculate_closest_cell_corner()

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

    def wait_for_startup_release(self):
        """Block all agent activity until the initial spawn pause has elapsed."""
        while not self.do_stop:
            remaining_time = self.startup_release_time - time.time()
            if remaining_time <= 0:
                return
            loop_start_time = time.time()
            utils.sleep_with_adjustable_period(
                min(self.state_polling_period, remaining_time),
                loop_start_time,
            )

    def get_elapsed_simulation_time(self):
        """Return elapsed trial time rescaled to simulation seconds."""
        return (time.time() - self.trial_start_time) * self.speed_up_factor

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
            self.wait_for_startup_release()
            while True:
                loop_start_time = time.time()
                if not (self.use_laplacian_consensus and self.is_laplacian_frozen()):
                    self.update_grid_search_position()

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
            self.wait_for_startup_release()
            descriptor_buffer_length = 1
            while not self.do_stop:
                loop_start_time = time.time()
                if self.use_laplacian_consensus and self.is_laplacian_frozen():
                    utils.sleep_with_adjustable_period(self.exploration_period, loop_start_time)
                    continue
                photo = self.capture_photo()
                self.update_descriptors(photo)
                
                if len(self.descriptors) >= descriptor_buffer_length:
                    self.calculate_vlad_vector()
                    cosine_similarity = utils.evaluate_similarity(self.vlad_vectors, self.codebook)
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
            self.wait_for_startup_release()
            exploration_thread = self.start_exploration_thread()
            if self.use_laplacian_consensus:
                self.set_state(constants.DECISION_MAKING_STATE)
                while not self.do_stop:
                    loop_start_time = time.time()
                    utils.sleep_with_adjustable_period(self.state_polling_period, loop_start_time)
                exploration_thread.join()
                return

            while not self.do_stop:
                # Receive social information. Method for collecting social information is called from the simulation module, because it needs access to other Agents objects. 
                # Here we only set state and reset social information buffer.
                self.reset_social_buffers()
                self.set_state(constants.RECEIVING_STATE)
                self.wait_for_scaled_duration(self.receiving_time)

                # Decision making, based on personal and social information
                self.set_state(constants.DECISION_MAKING_STATE)

                self.opinions, self.ds_scores = dm.dempster_shafer(
                    self.get_personal_information(),
                    self.opinions,
                    self.social_information,
                    self.theta_set,
                    self.id,
                    self.use_yager_social_fusion,
                )
                if not self.use_fixed_threshold:
                    self.update_dynamic_threshold()
                if self.use_laplacian_consensus:
                    self.update_laplacian_consensus()

                current_decision = self.calculate_decision()
                self.set_decision(current_decision)
                self.record_decision_cycle(current_decision)
                self.append_best_decision_history()

                # Disseminate 
                self.set_state(constants.DISSEMINATION_STATE)
                dissemination_time = self.get_dissemination_time()
                self.wait_for_scaled_duration(dissemination_time)

                # For results only
                self.max_opinions.append(int(np.argmax(self.ds_scores)))

            exploration_thread.join()

        decision_making_thread = threading.Thread(target = decision_making_in_loop, name = 'decision_making_thread_' + self.id, daemon = True)
        decision_making_thread.start()

        return decision_making_thread

    def run_laplacian_decision_cycle(self):
        """Run one evidence-plus-consensus cycle in Laplacian mode."""
        if not self.is_laplacian_frozen():
            social_information = self.get_filtered_social_information()
            self.opinions, self.ds_scores = dm.dempster_shafer(
                self.get_personal_information(),
                self.opinions,
                social_information,
                self.theta_set,
                self.id,
                self.use_yager_social_fusion,
            )
            if not self.use_fixed_threshold:
                self.update_dynamic_threshold()
            self.update_laplacian_consensus()
            current_decision = self.calculate_decision()
            self.set_decision(current_decision)
            self.append_best_decision_history()
            self.max_opinions.append(int(np.argmax(self.ds_scores)))
        else:
            self.update_frozen_average_consensus()
            current_decision = self.get_decision()

        self.record_decision_cycle(current_decision)

    def reset_social_buffers(self):
        """Clear one communication-cycle snapshot of received neighbor data."""
        self.social_information = []
        self.social_best_hypotheses = []
        self.social_thresholds = []
        self.social_decision_flags = []
        self.social_consensus_positions = []
        self.social_consensus_confidences = []
        self.social_consensus_frozen_flags = []
        self.received_neighbor_ids = set()
                
    def collect_opinions(self, agents):
        """Receive opinions from the neighbours that are currently in dissemination state.

        Args:
            agents (list): List of all Agent objects.
        """
        for agent in agents:
            if agent.id == self.id:
                continue
            if agent.id in self.received_neighbor_ids:
                continue
            if self.use_laplacian_consensus:
                if agent.id not in self.fixed_neighbor_ids:
                    continue
            else:
                if np.linalg.norm(self.get_position() - agent.get_position()) > constants.COMMUNICATION_RANGE:
                    continue

            if not self.use_laplacian_consensus and agent.get_state() != constants.DISSEMINATION_STATE:
                continue

            agent_information = agent.opinions
            if len(agent_information):
                self.social_information.append(agent_information)
                neighbor_best_hypothesis, _ = agent.get_best_decision_score()
                self.social_best_hypotheses.append(neighbor_best_hypothesis)
            if not self.use_fixed_threshold:
                self.social_thresholds.append(agent.get_dynamic_threshold())
                self.social_decision_flags.append(agent.get_decision() is not None)
            if self.use_laplacian_consensus:
                neighbor_consensus_position = agent.get_consensus_position()
                neighbor_consensus_confidence = agent.get_consensus_confidence()
                if neighbor_consensus_position is not None and neighbor_consensus_confidence > 0:
                    self.social_consensus_positions.append(neighbor_consensus_position)
                    self.social_consensus_confidences.append(float(neighbor_consensus_confidence))
                    self.social_consensus_frozen_flags.append(agent.is_laplacian_frozen())
            self.received_neighbor_ids.add(agent.id)

    def get_filtered_social_information(self):
        """Return Laplacian-mode social evidence filtered by adjacency of neighbors' best hypotheses."""
        if not self.use_laplacian_consensus:
            return self.social_information
        if len(self.social_information) <= 1:
            return self.social_information

        valid_entries = [
            (index, best_hypothesis)
            for index, best_hypothesis in enumerate(self.social_best_hypotheses)
            if best_hypothesis is not None
        ]
        if len(valid_entries) <= 1:
            return self.social_information

        kept_indices = set()
        for left_offset, (left_index, left_hypothesis) in enumerate(valid_entries):
            for right_index, right_hypothesis in valid_entries[left_offset + 1:]:
                if self.are_adjacent_hypotheses(left_hypothesis, right_hypothesis):
                    kept_indices.add(left_index)
                    kept_indices.add(right_index)

        if not kept_indices:
            return self.social_information

        return [
            self.social_information[index]
            for index in sorted(kept_indices)
            if index < len(self.social_information)
        ]

    def are_adjacent_hypotheses(self, first_hypothesis, second_hypothesis):
        """Return whether two hypothesis tiles are adjacent, including diagonal adjacency."""
        rows, _ = self.grid_shape
        first_x = int(first_hypothesis) // rows
        first_y = int(first_hypothesis) % rows
        second_x = int(second_hypothesis) // rows
        second_y = int(second_hypothesis) % rows
        return max(abs(first_x - second_x), abs(first_y - second_y)) == 1
    
    def calculate_decision(self):
        """Calculate current decision based on opinions.

        Returns:
            int: Id of the decided subimage or None if no decision has been made yet.
        """
        best_hypothesis, best_decision_score = self.get_best_decision_score()
        if best_hypothesis is None:
            return None
        threshold = self.decision_threshold if self.use_fixed_threshold else self.get_dynamic_threshold()
        if best_decision_score >= threshold:
            return best_hypothesis

        return None

    def get_current_threshold(self):
        """Return the active threshold used for the current decision rule."""
        return self.decision_threshold if self.use_fixed_threshold else self.get_dynamic_threshold()

    def get_best_decision_score(self):
        """Return the highest-scoring tile according to the current decision scores."""
        if self.ds_scores is None or not len(self.ds_scores):
            return None, -1.0
        best_hypothesis = int(np.argmax(self.ds_scores))
        return best_hypothesis, float(self.ds_scores[best_hypothesis])

    def get_best_focal_set(self):
        """Return the highest-mass non-Theta focal set from the current DS mass."""
        best_focal_set = frozenset()
        best_mass = -1.0
        theta_size = len(self.theta_set)

        for focal_set, mass in self.opinions.items():
            if len(focal_set) == 0 or len(focal_set) == theta_size:
                continue
            if mass > best_mass:
                best_focal_set = focal_set
                best_mass = float(mass)

        return best_focal_set

    def get_best_focal_set_and_mass(self):
        """Return the strongest non-Theta focal set and its mass."""
        best_focal_set = frozenset()
        best_mass = -1.0
        theta_size = len(self.theta_set)

        for focal_set, mass in self.opinions.items():
            if len(focal_set) == 0 or len(focal_set) == theta_size:
                continue
            if mass > best_mass:
                best_focal_set = focal_set
                best_mass = float(mass)

        return best_focal_set, best_mass

    def record_decision_cycle(self, current_decision):
        """Store one per-cycle snapshot for later analysis export."""
        self.decision_cycle_index += 1
        best_hypothesis, best_decision_score = self.get_best_decision_score()
        elapsed_time = self.get_elapsed_simulation_time()
        threshold = self.get_current_threshold()
        unique_received_opinion_count = len(self.social_information)
        consensus_position = self.get_consensus_position()
        consensus_position_change = self.get_consensus_position_change()

        if current_decision is not None and self.first_decision_time is None:
            self.first_decision_time = elapsed_time
            self.first_decision_cycle = self.decision_cycle_index

        self.decision_dynamics.append(
            {
                "cycle_index": self.decision_cycle_index,
                "elapsed_time": elapsed_time,
                "best_hypothesis": best_hypothesis,
                "best_score": best_decision_score,
                "threshold": threshold,
                "unique_received_opinion_count": unique_received_opinion_count,
                "consensus_position_x": None if consensus_position is None else float(consensus_position[0]),
                "consensus_position_y": None if consensus_position is None else float(consensus_position[1]),
                "consensus_position_change": consensus_position_change,
                "consensus_confidence": self.get_consensus_confidence(),
                "consensus_initialized": self.is_consensus_initialized(),
                "laplacian_frozen": self.is_laplacian_frozen(),
                "laplacian_done": self.is_laplacian_done(),
                "decision": current_decision,
            }
        )

    def get_consensus_position(self):
        """Return the current Laplacian-consensus estimate if initialized."""
        with self.consensus_locker:
            if self.consensus_position is None:
                return None
            return np.array(self.consensus_position, dtype = np.float32)

    def set_consensus_position(self, value):
        """Store the current Laplacian-consensus estimate."""
        with self.consensus_locker:
            self.consensus_position = np.array(value, dtype = np.float32)

    def is_consensus_initialized(self):
        """Return whether the consensus state has been initialized."""
        with self.consensus_locker:
            return bool(self.consensus_initialized)

    def get_consensus_position_change(self):
        """Return the latest per-cycle consensus-position change magnitude."""
        with self.consensus_locker:
            if self.consensus_position is None or self.previous_consensus_position is None:
                return None
            return float(np.linalg.norm(self.consensus_position - self.previous_consensus_position))

    def get_consensus_confidence(self):
        """Return the local confidence used in Laplacian consensus."""
        with self.consensus_locker:
            if self.frozen_consensus_confidence is not None:
                return float(self.frozen_consensus_confidence)
        _, best_decision_score = self.get_best_decision_score()
        return max(0.0, float(best_decision_score))

    def get_consensus_stable_counter(self):
        """Return the number of consecutive stable consensus cycles."""
        with self.consensus_locker:
            return int(self.consensus_stable_counter)

    def is_laplacian_locally_stable(self):
        """Return whether the consensus position has stabilized locally."""
        with self.consensus_locker:
            return bool(
                self.consensus_initialized
                and self.consensus_stable_counter >= int(constants.LAPLACIAN_STABLE_CYCLES)
            )

    def is_laplacian_frozen(self):
        """Return whether the agent has frozen its evidence state in Laplacian mode."""
        with self.consensus_locker:
            return bool(self.laplacian_frozen)

    def is_laplacian_done(self):
        """Return whether the agent has completed the post-freeze neighbor-agreement phase."""
        with self.consensus_locker:
            return bool(self.laplacian_done)

    def freeze_laplacian_state(self):
        """Freeze movement, sensing, and DS updates while keeping communication and consensus active."""
        frozen_confidence = self.get_consensus_confidence()
        with self.consensus_locker:
            if self.laplacian_frozen:
                return
            self.laplacian_frozen = True
            self.laplacian_done = False
            self.laplacian_done_counter = 0
            self.frozen_consensus_confidence = frozen_confidence

    def update_laplacian_done_status(self):
        """Update the post-freeze completion criterion from neighbor consensus agreement."""
        with self.consensus_locker:
            if not self.laplacian_frozen or self.consensus_position is None:
                self.laplacian_done = False
                self.laplacian_done_counter = 0
                return

            frozen_neighbor_positions = [
                np.asarray(neighbor_position, dtype = np.float32)
                for neighbor_position, is_frozen in zip(
                    self.social_consensus_positions,
                    self.social_consensus_frozen_flags,
                )
                if is_frozen
            ]

            if not frozen_neighbor_positions:
                self.laplacian_done = False
                self.laplacian_done_counter = 0
                return

            current_position = np.array(self.consensus_position, dtype = np.float32)
            max_neighbor_gap = max(
                float(np.linalg.norm(neighbor_position - current_position))
                for neighbor_position in frozen_neighbor_positions
            )
            if max_neighbor_gap < float(constants.LAPLACIAN_NEIGHBOR_DELTA):
                self.laplacian_done_counter += 1
            else:
                self.laplacian_done_counter = 0

            self.laplacian_done = self.laplacian_done_counter >= int(constants.LAPLACIAN_DONE_CYCLES)

    def convert_decision_to_tile_center(self, decision):
        """Convert a tile index to the corresponding full-image tile center."""
        rows, cols = self.grid_shape
        subimage_width = self.original_image.shape[1] / cols
        subimage_height = self.original_image.shape[0] / rows
        x_index = int(decision) // rows
        y_index = int(decision) % rows
        return np.array(
            [
                (x_index + 0.5) * subimage_width,
                (y_index + 0.5) * subimage_height,
            ],
            dtype = np.float32,
        )

    def get_observation_position(self):
        """Return the local observation-derived tile center of the best hypothesis."""
        best_hypothesis, _ = self.get_best_decision_score()
        if best_hypothesis is None:
            return None
        return self.convert_decision_to_tile_center(best_hypothesis)

    def update_laplacian_consensus(self):
        """Update the decentralized confidence-weighted Laplacian consensus estimate."""
        observation_position = self.get_observation_position()
        if observation_position is None:
            return

        local_confidence = self.get_consensus_confidence()
        if local_confidence <= 0:
            return

        with self.consensus_locker:
            if not self.consensus_initialized:
                self.previous_consensus_position = None
                self.consensus_position = np.array(observation_position, dtype = np.float32)
                self.consensus_initialized = True
                self.consensus_stable_counter = 0
                return

            current_position = np.array(self.consensus_position, dtype = np.float32)
            neighbor_shift = np.zeros(2, dtype = np.float32)
            for neighbor_position, neighbor_confidence in zip(
                self.social_consensus_positions,
                self.social_consensus_confidences,
            ):
                denominator = local_confidence + float(neighbor_confidence)
                if denominator <= 0:
                    continue
                edge_weight = float(neighbor_confidence) / denominator
                neighbor_shift += edge_weight * (np.asarray(neighbor_position, dtype = np.float32) - current_position)

            updated_position = (
                current_position
                + constants.LAPLACIAN_CONSENSUS_GAIN * neighbor_shift
                + constants.LAPLACIAN_OBSERVATION_GAIN * (np.asarray(observation_position, dtype = np.float32) - current_position)
            )
            position_change = float(np.linalg.norm(updated_position - current_position))
            self.previous_consensus_position = current_position
            self.consensus_position = np.array(updated_position, dtype = np.float32)
            if not self.laplacian_frozen:
                if position_change < float(constants.LAPLACIAN_POSITION_DELTA):
                    self.consensus_stable_counter += 1
                else:
                    self.consensus_stable_counter = 0
            else:
                self.consensus_stable_counter = int(constants.LAPLACIAN_STABLE_CYCLES)

        if not self.is_laplacian_frozen() and self.is_laplacian_locally_stable():
            self.freeze_laplacian_state()
        if self.is_laplacian_frozen():
            self.update_laplacian_done_status()

    def update_frozen_average_consensus(self):
        """Run Stage-2 weighted average consensus using only frozen neighbors."""
        if not self.is_laplacian_frozen():
            return

        current_confidence = max(float(self.get_consensus_confidence()), 1e-9)
        with self.consensus_locker:
            if self.consensus_position is None:
                self.laplacian_done = False
                self.laplacian_done_counter = 0
                return

            current_position = np.array(self.consensus_position, dtype = np.float32)
            frozen_neighbor_positions = [
                np.asarray(neighbor_position, dtype = np.float32)
                for neighbor_position, is_frozen in zip(
                    self.social_consensus_positions,
                    self.social_consensus_frozen_flags,
                )
                if is_frozen
            ]

            self.previous_consensus_position = current_position.copy()
            if frozen_neighbor_positions:
                delta = np.zeros(2, dtype = np.float32)
                for neighbor_position in frozen_neighbor_positions:
                    delta += neighbor_position - current_position
                updated_position = current_position + (
                    (float(constants.LAPLACIAN_AVERAGE_GAIN) / current_confidence) * delta
                )
                self.consensus_position = np.array(updated_position, dtype = np.float32)

        self.update_laplacian_done_status()

    def get_dynamic_threshold(self):
        """Get the current decentralized decision threshold."""
        with self.threshold_locker:
            return float(self.dynamic_threshold)

    def set_dynamic_threshold(self, value):
        """Set the current decentralized decision threshold."""
        with self.threshold_locker:
            self.dynamic_threshold = float(value)

    def calculate_confidence_threshold_proposal(self):
        """Propose a threshold from the latest local decision score."""
        _, best_decision_score = self.get_best_decision_score()
        if best_decision_score <= 0:
            return constants.DS_THRESHOLD_MIN

        proposed_threshold = best_decision_score - constants.DS_THRESHOLD_MARGIN
        return float(np.clip(proposed_threshold, constants.DS_THRESHOLD_MIN, constants.DS_THRESHOLD_MAX))

    def calculate_decided_neighbor_majority_factor(self):
        """Return a majority factor based on the fraction of decided neighbors."""
        if not self.social_decision_flags:
            return 0.0

        decided_ratio = float(np.mean(self.social_decision_flags, dtype = np.float32))
        return max(0.0, (decided_ratio - 0.5) / 0.5)

    def calculate_stability_factor(self):
        """Return a score-stability factor from a sliding window of best decision scores."""
        best_hypothesis, best_decision_score = self.get_best_decision_score()
        if best_hypothesis is None or best_decision_score <= 0:
            return 0.0

        window_size = int(constants.DS_THRESHOLD_SCORE_WINDOW)
        if window_size < 2:
            return 0.0

        prior_history = self.best_decision_history[-(window_size - 1):]
        score_window = prior_history + [(best_hypothesis, best_decision_score)]
        if len(score_window) < window_size:
            return 0.0

        hypotheses = [hypothesis for hypothesis, _ in score_window]
        if len(set(hypotheses)) != 1:
            return 0.0

        score_differences = [
            abs(score_window[index][1] - score_window[index - 1][1])
            for index in range(1, len(score_window))
        ]
        average_difference = float(np.mean(score_differences, dtype = np.float32))
        epsilon = float(constants.DS_THRESHOLD_STABILITY_EPSILON)
        if epsilon <= 0:
            return 0.0

        return max(0.0, 1.0 - average_difference / epsilon)

    def append_best_decision_history(self):
        """Append the current best hypothesis and score to the short stability history."""
        best_hypothesis, best_decision_score = self.get_best_decision_score()
        if best_hypothesis is None or best_decision_score < 0:
            return

        self.best_decision_history.append((best_hypothesis, best_decision_score))
        max_history_length = max(0, int(constants.DS_THRESHOLD_SCORE_WINDOW) - 1)
        if max_history_length == 0:
            self.best_decision_history = []
            return
        if len(self.best_decision_history) > max_history_length:
            self.best_decision_history = self.best_decision_history[-max_history_length:]

    def update_dynamic_threshold(self):
        """Update the threshold from confidence, decided-neighbor majority, stability, and neighbor consensus."""
        current_threshold = self.get_dynamic_threshold()
        confidence_threshold = self.calculate_confidence_threshold_proposal()
        majority_factor = self.calculate_decided_neighbor_majority_factor()
        stability_factor = self.calculate_stability_factor()
        relaxation_factor = (
            constants.DS_THRESHOLD_DECIDED_NEIGHBOR_GAIN * majority_factor
            + constants.DS_THRESHOLD_STABILITY_GAIN * stability_factor
        )
        relaxation_factor = float(np.clip(relaxation_factor, 0.0, 1.0))
        local_threshold = confidence_threshold - relaxation_factor * (
            confidence_threshold - constants.DS_THRESHOLD_MIN
        )
        local_threshold = float(np.clip(local_threshold, constants.DS_THRESHOLD_MIN, constants.DS_THRESHOLD_MAX))
        anchored_threshold = (
            (1.0 - constants.DS_THRESHOLD_LOCAL_GAIN) * current_threshold
            + constants.DS_THRESHOLD_LOCAL_GAIN * local_threshold
        )

        if self.social_thresholds:
            consensus_shift = np.mean(
                [neighbor_threshold - anchored_threshold for neighbor_threshold in self.social_thresholds],
                dtype = np.float32,
            )
            updated_threshold = anchored_threshold + constants.DS_THRESHOLD_CONSENSUS_GAIN * float(consensus_shift)
        else:
            updated_threshold = anchored_threshold

        updated_threshold = float(np.clip(updated_threshold, constants.DS_THRESHOLD_MIN, constants.DS_THRESHOLD_MAX))
        self.set_dynamic_threshold(updated_threshold)
    
    def get_dissemination_time(self):
        """Calculate dissemination time, based on current opinion.

        Returns:
            float: Dissemination time in seconds.
        """
        _, best_decision_score = self.get_best_decision_score()
        if best_decision_score <= 0:
            return self.min_dissemination_time
        dissemination_time = self.max_dissemination_time * best_decision_score
        if dissemination_time > self.max_dissemination_time:
            dissemination_time = self.max_dissemination_time
        if dissemination_time < self.min_dissemination_time:
            dissemination_time = self.min_dissemination_time

        return dissemination_time
    
    def stop_threads(self):
        """Stop all running threads.
        """
        self.do_stop = True
        self.move_thread.join()
        self.decision_making_thread.join()
