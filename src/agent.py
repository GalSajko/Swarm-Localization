import cv2
import numpy as np
from scipy.cluster.vq import vq
import threading
import time

import constants
import decision_making as dm
import image_augmentation as ia
import utils

EXPLORATION_STATE = "e"
DECISION_MAKING_STATE = "dm"

EXPLORATION_PERIOD = 10.0
CAPTURE_PERIOD = 0.5
DESCRIPTOR_BUFFER_LENGTH = 1000

DS_THRESHOLD_MARGIN = 0.05
DS_THRESHOLD_LOCAL_GAIN = 0.5
DS_THRESHOLD_CONSENSUS_GAIN = 0.5
DS_THRESHOLD_SCORE_WINDOW = 4
DS_THRESHOLD_STABILITY_EPSILON = 0.05
DS_THRESHOLD_DECIDED_NEIGHBOR_GAIN = 0.5
DS_THRESHOLD_STABILITY_GAIN = 0.5

CONTINUOUS_STAGE2_AVERAGE_GAIN = 0.025
CONTINUOUS_NEIGHBOR_DELTA = 10.0
CONTINUOUS_DONE_CYCLES = 3
FROZEN_NEIGHBOR_FREEZE_RATIO = 0.75
CONTINUOUS_CONFIDENCE_FLOOR = 0.1
CONTINUOUS_MAX_STAGE2_STEP = 50.0
FROZEN_STAGE2_COMMUNICATION_PERIOD = 0.1

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
        social_fusion_rule = "comp_boost",
        personal_fusion_rule = "dempster",
        fixed_neighbor_ids = None,
        grid_matrix = None,
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
        self.descriptor_locker = threading.Lock()

        self.sift_extractor = constants.SIFT_EXTRACTOR
        self.original_image = original_image

        self.state = EXPLORATION_STATE
        self.state_locker = threading.Lock()
        self.exploration_state_counter = 0

        # Velocity gain and capture period specify how many captures the agent makes per meter of traveled path.
        # velocity_gain = 0.085 corresponds to cca 15m/s real-time velocity over given sub-image, capture_period = 0.5 means that
        # agent captures the surface twice per real-time second.
        # Given those values, agents captures the surface on cca 7.5 meters of traveled path. 
        self.max_exploration_period = EXPLORATION_PERIOD / self.speed_up_factor
        self.capture_period = CAPTURE_PERIOD / self.speed_up_factor
        self.velocity_gain = 0.085 * self.speed_up_factor
        self.movement_period = 0.01
        self.state_polling_period = 0.1 / self.speed_up_factor
        self.trial_start_time = time.time() if trial_start_time is None else trial_start_time
        self.startup_release_time = time.time()

        self.social_information = []
        self.social_information_neighbor_ids = []
        self.social_best_hypotheses = []
        self.social_thresholds = []
        self.social_decision_flags = []
        self.social_frozen_flags = []
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
        self.social_fusion_rule = str(social_fusion_rule)
        self.personal_fusion_rule = str(personal_fusion_rule)
        self.fixed_neighbor_ids = set() if fixed_neighbor_ids is None else {str(neighbor_id) for neighbor_id in fixed_neighbor_ids}
        self.swarm_agents = None
        self.dynamic_threshold = self.decision_threshold
        self.threshold_locker = threading.Lock()
        self.previous_best_hypothesis = None
        self.best_hypothesis_stability = 0.0
        self.theta_set = set(range(len(self.codebook)))
        self.consensus_position = None
        self.previous_consensus_position = None
        self.consensus_initialized = False
        self.consensus_stable_counter = 0
        self.continuous_frozen = False
        self.continuous_done = False
        self.continuous_done_counter = 0
        self.frozen_consensus_confidence = None
        self.consensus_locker = threading.Lock()
        self.grid_shape = dm._infer_grid_shape(len(self.codebook))
        self.grid_matrix = None if grid_matrix is None else np.asarray(grid_matrix, dtype = np.int32)

        # For results only
        self.max_opinions = []
        self.decision_cycle_index = 0
        self.first_decision_time = None
        self.first_decision_cycle = None
        self.decision_dynamics = []
        self.best_decision_history = []
        self.personal_fusion_yager_count = 0
        self.personal_fusion_dempster_count = 0

        # Opinion is defined as name of the subimage.
        self.decision = None
        self.decision_locker = threading.Lock()

        self.vlad_vectors = np.zeros((len(self.centroids), len(self.centroids[0])), dtype = np.float32)

        self.grid_mode = "move_to_corner"
        self.sweep_axis = np.random.choice(["horizontal", "vertical"])
        self.primary_direction = np.random.choice([-1.0, 1.0])
        self.secondary_direction = np.random.choice([-1.0, 1.0])
        self.current_lane_coord = None
        self.target_lane_coord = None
        self.border_anchor = None
        self.target_corner = self.calculate_closest_cell_corner()

        self.move_thread = self.move_in_thread()
        self.control_thread = None

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
        if value not in (EXPLORATION_STATE, DECISION_MAKING_STATE):
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

    def attach_swarm(self, agents):
        """Attach the full swarm reference and start the control loop once neighbors are available."""
        self.swarm_agents = agents
        if self.control_thread is None:
            self.control_thread = self.start_control_thread()

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
                if not self.is_continuous_frozen():
                    self.update_grid_search_position()

                if self.do_stop:
                    return 
                
                utils.sleep_with_adjustable_period(self.movement_period, loop_start_time)

        move_thread = threading.Thread(target = move, name = 'move_thread_' + self.id, daemon = True)
        move_thread.start()

        return move_thread

    def run_exploration_cycle(self):
        """Explore until enough descriptors are collected or the exploration time limit is reached.

        Frozen agents skip exploration and simply wait through the exploration window.
        """
        exploration_start_time = time.time()
        next_capture_time = exploration_start_time
        next_frozen_communication_time = exploration_start_time

        while not self.do_stop:
            current_time = time.time()
            elapsed_time = current_time - exploration_start_time
            if elapsed_time >= self.max_exploration_period:
                return

            if self.is_continuous_frozen():
                if current_time >= next_frozen_communication_time:
                    self.run_frozen_stage2_cycle()
                    next_frozen_communication_time += FROZEN_STAGE2_COMMUNICATION_PERIOD
                    continue
                remaining_time = self.max_exploration_period - elapsed_time
                sleep_duration = min(
                    self.state_polling_period,
                    remaining_time,
                    max(0.0, next_frozen_communication_time - current_time),
                )
                utils.sleep_with_adjustable_period(sleep_duration, current_time)
                continue

            if current_time >= next_capture_time:
                photo = self.capture_photo()
                self.update_descriptors(photo)
                if self.get_descriptor_count() > int(DESCRIPTOR_BUFFER_LENGTH):
                    return
                next_capture_time += self.capture_period
                continue

            sleep_duration = min(next_capture_time - current_time, self.max_exploration_period - elapsed_time)
            utils.sleep_with_adjustable_period(sleep_duration, current_time)
    
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
            with self.descriptor_locker:
                for descriptor in root_sift_descriptors:
                    self.descriptors.append(descriptor)
        except TypeError:
            return

    def get_descriptor_count(self):
        """Return the number of descriptors collected in the current exploration buffer."""
        with self.descriptor_locker:
            return len(self.descriptors)
        
    def calculate_vlad_vector(self, descriptors):
        """Calculate VLAD vector.
        """
        if not descriptors:
            return
        descriptors_array = np.asarray(descriptors, dtype = np.float32)
        mapped_descriptors, _ = vq(descriptors_array, self.centroids)
        centroid_indices = np.asarray(mapped_descriptors, dtype = np.int64)
        residual = descriptors_array - self.centroids[centroid_indices]
        np.add.at(self.vlad_vectors, centroid_indices, residual)

        # Intra-normalization
        eps = 1e-12
        norms = np.linalg.norm(self.vlad_vectors, axis = 1, keepdims = True)
        self.vlad_vectors /= np.maximum(norms, eps)
        # Global L2 normalization
        flat_vlad = self.vlad_vectors.reshape(1, -1)
        global_norm = np.linalg.norm(flat_vlad, axis = 1, keepdims = True)
        flat_vlad /= np.maximum(global_norm, eps)

    def drain_visual_evidence_buffer(self):
        """Atomically detach one full exploration-cycle descriptor batch for decision making."""
        with self.descriptor_locker:
            drained_descriptors = list(self.descriptors)
            self.descriptors = []
        self.vlad_vectors.fill(0.0)
        return drained_descriptors

    def update_personal_information_from_visual_buffer(self):
        """Build one per-cycle personal-evidence vector from the descriptors collected since the last decision step."""
        descriptors = self.drain_visual_evidence_buffer()
        if not descriptors:
            uniform_information = np.full(len(self.codebook), 1.0 / len(self.codebook), dtype = np.float32)
            self.set_personal_information(uniform_information)
            return

        self.calculate_vlad_vector(descriptors)
        cosine_similarity = utils.evaluate_similarity(self.vlad_vectors, self.codebook)
        personal_information = utils.softmax_over_vector(cosine_similarity)
        self.set_personal_information(personal_information)

    def resolve_personal_fusion_rule(self):
        """Return the effective personal fusion rule for this agent."""
        if self.personal_fusion_rule != "hybrid":
            return self.personal_fusion_rule
        return "yager" if int(self.id) % 2 == 0 else "dempster"

    def run_continuous_decision_cycle(self):
        """Run one discrete-plus-continuous collective-decision cycle."""
        self.set_state(DECISION_MAKING_STATE)
        if not self.is_continuous_frozen():
            previous_ranked_hypotheses = self.get_ranked_hypotheses_from_current_scores(max_rank = 4)
            self.update_personal_information_from_visual_buffer()
            social_information = self.social_information
            social_neighbor_ids = self.social_information_neighbor_ids
            effective_personal_fusion_rule = self.resolve_personal_fusion_rule()
            self.opinions, self.ds_scores, selected_personal_fusion_rule = dm.dempster_shafer(
                self.get_personal_information(),
                self.opinions,
                social_information,
                self.theta_set,
                agent_id = self.id,
                social_fusion_rule = self.social_fusion_rule,
                personal_fusion_rule = effective_personal_fusion_rule,
                social_neighbor_ids = social_neighbor_ids,
                swarm_grid_matrix = self.grid_matrix,
            )
            if self.personal_fusion_rule == "adaptive":
                if selected_personal_fusion_rule == "yager":
                    self.personal_fusion_yager_count += 1
                elif selected_personal_fusion_rule == "dempster":
                    self.personal_fusion_dempster_count += 1
            if not self.use_fixed_threshold:
                self.update_dynamic_threshold(previous_ranked_hypotheses)
            current_decision = self.calculate_decision()
            if current_decision is not None:
                self.set_decision(current_decision)
                self.freeze_from_discrete_stage(current_decision)
            elif self.try_freeze_from_frozen_neighbors():
                current_decision = self.get_decision()
            else:
                self.set_decision(current_decision)
            self.append_best_decision_history()
            self.max_opinions.append(int(np.argmax(self.ds_scores)))
        else:
            self.update_continuous_consensus()
            current_decision = self.get_decision()

        self.record_decision_cycle(current_decision)

    def run_frozen_stage2_cycle(self):
        """Run one frozen-only Stage-2 consensus exchange with frozen neighbors."""
        self.set_state(DECISION_MAKING_STATE)
        self.reset_social_buffers()
        if self.swarm_agents is not None:
            self.collect_frozen_neighbor_consensus(self.swarm_agents)
        self.update_continuous_consensus()
        self.record_decision_cycle(self.get_decision())

    def start_control_thread(self):
        """Run the per-agent exploration then decision loop in one dedicated control thread."""
        def control_loop():
            self.wait_for_startup_release()
            while not self.do_stop:
                self.set_state(EXPLORATION_STATE)
                self.run_exploration_cycle()
                if self.do_stop:
                    return

                if self.is_continuous_frozen():
                    continue

                self.reset_social_buffers()
                if self.swarm_agents is not None:
                    self.collect_opinions(self.swarm_agents)
                self.run_continuous_decision_cycle()

        control_thread = threading.Thread(target = control_loop, name = 'control_thread_' + self.id, daemon = True)
        control_thread.start()
        return control_thread

    def reset_social_buffers(self):
        """Clear one communication-cycle snapshot of received neighbor data."""
        self.social_information = []
        self.social_information_neighbor_ids = []
        self.social_best_hypotheses = []
        self.social_thresholds = []
        self.social_decision_flags = []
        self.social_frozen_flags = []
        self.social_consensus_positions = []
        self.social_consensus_confidences = []
        self.social_consensus_frozen_flags = []
        self.received_neighbor_ids = set()
                
    def collect_opinions(self, agents):
        """Request current evidence and consensus state from fixed grid neighbors.

        Args:
            agents (list): List of all Agent objects.
        """
        for agent in agents:
            if agent.id == self.id:
                continue
            if agent.id in self.received_neighbor_ids:
                continue
            if agent.id not in self.fixed_neighbor_ids:
                continue

            agent_information = agent.opinions
            if len(agent_information):
                self.social_information.append(agent_information)
                self.social_information_neighbor_ids.append(str(agent.id))
                neighbor_best_hypothesis, _ = agent.get_best_decision_score()
                self.social_best_hypotheses.append(neighbor_best_hypothesis)
            if not self.use_fixed_threshold:
                self.social_thresholds.append(agent.get_dynamic_threshold())
                self.social_decision_flags.append(agent.get_decision() is not None)
            self.social_frozen_flags.append(agent.is_continuous_frozen())
            neighbor_consensus_position = agent.get_consensus_position()
            neighbor_consensus_confidence = agent.get_consensus_confidence()
            if neighbor_consensus_position is not None and neighbor_consensus_confidence > 0:
                self.social_consensus_positions.append(neighbor_consensus_position)
                self.social_consensus_confidences.append(float(neighbor_consensus_confidence))
                self.social_consensus_frozen_flags.append(agent.is_continuous_frozen())
            self.received_neighbor_ids.add(agent.id)

    def collect_frozen_neighbor_consensus(self, agents):
        """Request only frozen-neighbor consensus state for Stage-2 updates."""
        for agent in agents:
            if agent.id == self.id:
                continue
            if agent.id in self.received_neighbor_ids:
                continue
            if agent.id not in self.fixed_neighbor_ids:
                continue

            is_neighbor_frozen = agent.is_continuous_frozen()
            self.social_frozen_flags.append(is_neighbor_frozen)
            if not is_neighbor_frozen:
                self.received_neighbor_ids.add(agent.id)
                continue

            neighbor_consensus_position = agent.get_consensus_position()
            neighbor_consensus_confidence = agent.get_consensus_confidence()
            if neighbor_consensus_position is not None and neighbor_consensus_confidence > 0:
                self.social_consensus_positions.append(neighbor_consensus_position)
                self.social_consensus_confidences.append(float(neighbor_consensus_confidence))
                self.social_consensus_frozen_flags.append(True)
            self.received_neighbor_ids.add(agent.id)

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

    def get_ranked_hypotheses_from_current_scores(self, max_rank = None):
        """Return hypotheses sorted by current decision score, excluding zero-support entries."""
        if self.ds_scores is None or not len(self.ds_scores):
            return []

        ranked_indices = np.argsort(self.ds_scores)[::-1]
        ranked_hypotheses = [
            int(index)
            for index in ranked_indices
            if float(self.ds_scores[index]) > 0.0
        ]
        if max_rank is not None:
            return ranked_hypotheses[:max_rank]
        return ranked_hypotheses

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
                "continuous_frozen": self.is_continuous_frozen(),
                "continuous_done": self.is_continuous_done(),
                "decision": current_decision,
            }
        )

    def get_consensus_position(self):
        """Return the current Laplacian-consensus estimate if initialized."""
        with self.consensus_locker:
            if self.consensus_position is None:
                return None
            if not np.all(np.isfinite(self.consensus_position)):
                return None
            return np.array(self.consensus_position, dtype = np.float32)

    def set_consensus_position(self, value):
        """Store the current Laplacian-consensus estimate."""
        with self.consensus_locker:
            candidate_position = np.array(value, dtype = np.float32)
            if not np.all(np.isfinite(candidate_position)):
                return
            self.consensus_position = candidate_position

    def is_consensus_initialized(self):
        """Return whether the consensus state has been initialized."""
        with self.consensus_locker:
            return bool(self.consensus_initialized)

    def get_consensus_position_change(self):
        """Return the latest per-cycle consensus-position change magnitude."""
        with self.consensus_locker:
            if self.consensus_position is None or self.previous_consensus_position is None:
                return None
            if not np.all(np.isfinite(self.consensus_position)) or not np.all(np.isfinite(self.previous_consensus_position)):
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

    def is_continuous_frozen(self):
        """Return whether the agent has frozen its stage-1 discrete evidence state."""
        with self.consensus_locker:
            return bool(self.continuous_frozen)

    def is_continuous_done(self):
        """Return whether the agent has completed the stage-2 neighbor-agreement phase."""
        with self.consensus_locker:
            return bool(self.continuous_done)

    def freeze_continuous_state(self, initial_consensus_position = None):
        """Freeze movement, sensing, and DS updates while keeping communication and consensus active."""
        frozen_confidence = self.get_consensus_confidence()
        with self.consensus_locker:
            if self.continuous_frozen:
                return
            if initial_consensus_position is not None:
                initial_position = np.asarray(initial_consensus_position, dtype = np.float32)
                self.consensus_position = initial_position.copy()
                self.previous_consensus_position = initial_position.copy()
                self.consensus_initialized = True
            self.continuous_frozen = True
            self.continuous_done = False
            self.continuous_done_counter = 0
            self.frozen_consensus_confidence = frozen_confidence
            self.consensus_stable_counter = 0

    def freeze_from_discrete_stage(self, decision):
        """Freeze directly from a discrete best hypothesis selected by the dynamic threshold logic."""
        if decision is None:
            return
        self.freeze_continuous_state(self.convert_decision_to_tile_center(decision))

    def convert_position_to_decision_index(self, position):
        """Convert a continuous image position to the index of the containing tile."""
        rows, cols = self.grid_shape
        subimage_width = self.original_image.shape[1] / cols
        subimage_height = self.original_image.shape[0] / rows
        x_index = min(max(int(position[0] // subimage_width), 0), cols - 1)
        y_index = min(max(int(position[1] // subimage_height), 0), rows - 1)
        return int(x_index * rows + y_index)

    def try_freeze_from_frozen_neighbors(self):
        """Force-freeze an active agent from a strong frozen-neighbor majority."""
        if self.is_continuous_frozen():
            return False
        if not self.social_frozen_flags:
            return False

        frozen_neighbor_ratio = sum(bool(is_frozen) for is_frozen in self.social_frozen_flags) / float(len(self.social_frozen_flags))
        if frozen_neighbor_ratio < float(FROZEN_NEIGHBOR_FREEZE_RATIO):
            return False

        weighted_positions = [
            (np.asarray(position, dtype = np.float32), float(confidence))
            for position, confidence, is_frozen in zip(
                self.social_consensus_positions,
                self.social_consensus_confidences,
                self.social_consensus_frozen_flags,
            )
            if is_frozen and confidence > 0
        ]
        if not weighted_positions:
            return False

        total_weight = sum(weight for _, weight in weighted_positions)
        if total_weight <= 0:
            return False

        weighted_average_position = sum(
            weight * position for position, weight in weighted_positions
        ) / total_weight
        adopted_decision = self.convert_position_to_decision_index(weighted_average_position)
        self.set_decision(adopted_decision)
        self.freeze_continuous_state(weighted_average_position)
        return True
    
    def update_continuous_done_status(self):
        """Update the post-freeze completion criterion from neighbor consensus agreement."""
        with self.consensus_locker:
            if not self.continuous_frozen or self.consensus_position is None:
                self.continuous_done = False
                self.continuous_done_counter = 0
                return
            if not np.all(np.isfinite(self.consensus_position)):
                self.continuous_done = False
                self.continuous_done_counter = 0
                return

            frozen_neighbor_positions = [
                np.asarray(neighbor_position, dtype = np.float32)
                for neighbor_position, is_frozen in zip(
                    self.social_consensus_positions,
                    self.social_consensus_frozen_flags,
                )
                if is_frozen and np.all(np.isfinite(neighbor_position))
            ]

            if not frozen_neighbor_positions:
                self.continuous_done = False
                self.continuous_done_counter = 0
                return

            current_position = np.array(self.consensus_position, dtype = np.float32)
            max_neighbor_gap = max(
                float(np.linalg.norm(neighbor_position - current_position))
                for neighbor_position in frozen_neighbor_positions
            )
            if max_neighbor_gap < float(CONTINUOUS_NEIGHBOR_DELTA):
                self.continuous_done_counter += 1
            else:
                self.continuous_done_counter = 0

            self.continuous_done = self.continuous_done_counter >= int(CONTINUOUS_DONE_CYCLES)

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

    def update_continuous_consensus(self):
        """Run Stage-2 weighted average consensus using only frozen neighbors."""
        if not self.is_continuous_frozen():
            return

        current_confidence = max(
            float(self.get_consensus_confidence()),
            float(CONTINUOUS_CONFIDENCE_FLOOR),
        )
        with self.consensus_locker:
            if self.consensus_position is None:
                self.continuous_done = False
                self.continuous_done_counter = 0
                return

            current_position = np.array(self.consensus_position, dtype = np.float32)
            if not np.all(np.isfinite(current_position)):
                self.continuous_done = False
                self.continuous_done_counter = 0
                return
            frozen_neighbor_positions = [
                np.asarray(neighbor_position, dtype = np.float32)
                for neighbor_position, is_frozen in zip(
                    self.social_consensus_positions,
                    self.social_consensus_frozen_flags,
                )
                if is_frozen and np.all(np.isfinite(neighbor_position))
            ]

            self.previous_consensus_position = current_position.copy()
            if frozen_neighbor_positions:
                delta = np.zeros(2, dtype = np.float32)
                for neighbor_position in frozen_neighbor_positions:
                    delta += neighbor_position - current_position
                raw_step = (float(CONTINUOUS_STAGE2_AVERAGE_GAIN) / current_confidence) * delta
                step_norm = float(np.linalg.norm(raw_step))
                if np.isfinite(step_norm) and step_norm > float(CONTINUOUS_MAX_STAGE2_STEP):
                    raw_step = raw_step / max(step_norm, 1e-12) * float(CONTINUOUS_MAX_STAGE2_STEP)
                updated_position = current_position + raw_step
                if not np.all(np.isfinite(updated_position)):
                    updated_position = current_position.copy()
                self.consensus_position = np.array(updated_position, dtype = np.float32)

        self.update_continuous_done_status()

    def get_dynamic_threshold(self):
        """Get the current decentralized decision threshold."""
        with self.threshold_locker:
            return float(self.dynamic_threshold)

    def set_dynamic_threshold(self, value):
        """Set the current decentralized decision threshold."""
        with self.threshold_locker:
            self.dynamic_threshold = float(value)

    def calculate_hypothesis_change_retention(self, previous_ranked_hypotheses, best_hypothesis):
        """Return how much stability to keep after a best-hypothesis change."""
        if best_hypothesis not in previous_ranked_hypotheses:
            return 0.0

        previous_rank = previous_ranked_hypotheses.index(best_hypothesis) + 1
        if previous_rank == 2:
            return 0.75
        if previous_rank == 3:
            return 0.50
        if previous_rank == 4:
            return 0.25
        return 0.0

    def calculate_hypothesis_stability_factor(self, previous_ranked_hypotheses):
        """Return a stability factor based only on persistence of the best hypothesis."""
        best_hypothesis, _ = self.get_best_decision_score()
        saturation_cycles = max(1, int(DS_THRESHOLD_SCORE_WINDOW))
        stability_step = 1.0 / float(saturation_cycles)

        if best_hypothesis is None:
            self.previous_best_hypothesis = None
            self.best_hypothesis_stability = 0.0
            return 0.0

        if self.previous_best_hypothesis is None:
            self.previous_best_hypothesis = best_hypothesis
            self.best_hypothesis_stability = 0.0
            return self.best_hypothesis_stability

        if best_hypothesis == self.previous_best_hypothesis:
            self.best_hypothesis_stability = min(1.0, self.best_hypothesis_stability + stability_step)
            return self.best_hypothesis_stability

        retention_factor = self.calculate_hypothesis_change_retention(previous_ranked_hypotheses, best_hypothesis)
        self.best_hypothesis_stability *= retention_factor
        self.previous_best_hypothesis = best_hypothesis
        return self.best_hypothesis_stability

    def append_best_decision_history(self):
        """Append the current best hypothesis and score to the short stability history."""
        best_hypothesis, best_decision_score = self.get_best_decision_score()
        if best_hypothesis is None or best_decision_score < 0:
            return

        self.best_decision_history.append((best_hypothesis, best_decision_score))
        max_history_length = max(0, int(DS_THRESHOLD_SCORE_WINDOW) - 1)
        if max_history_length == 0:
            self.best_decision_history = []
            return
        if len(self.best_decision_history) > max_history_length:
            self.best_decision_history = self.best_decision_history[-max_history_length:]

    def update_dynamic_threshold(self, previous_ranked_hypotheses):
        """Update the threshold from best-hypothesis persistence with softened resets on changes."""
        stability_factor = self.calculate_hypothesis_stability_factor(previous_ranked_hypotheses)
        threshold_max = float(self.decision_threshold)
        threshold_min = float(constants.DS_THRESHOLD_MIN)
        updated_threshold = threshold_max - stability_factor * (threshold_max - threshold_min)
        updated_threshold = float(np.clip(updated_threshold, threshold_min, threshold_max))
        self.set_dynamic_threshold(updated_threshold)
    
    def stop_threads(self):
        """Stop all running threads.
        """
        self.do_stop = True
        self.move_thread.join()
        if self.control_thread is not None:
            self.control_thread.join()
