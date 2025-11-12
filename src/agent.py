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
    def __init__(self, original_image, centroids, codebook, start_position, id):
        self.id = id 

        self.do_stop = False

        self.position_locker = threading.Lock()
        self.position = start_position

        self.centroids = centroids
        self.codebook = codebook
        self.descriptors = []

        self.sift_extractor = constants.SIFT_EXCTRACTOR
        self.original_image = original_image

        self.state = constants.EXPLORATION_STATE
        self.state_locker = threading.Lock()
        self.exploration_state_counter = 0

        self.personal_information = np.zeros(len(self.codebook))
        self.personal_information_locker = threading.Lock()
        self.social_information_dict = {}
        self.opinions = np.zeros(len(self.codebook))

        # For results only
        self.max_opinions = []

        self.social_factor = 2.0
        self.decision_threshold = np.ones_like(self.opinions) * constants.INITIAL_DECISION_THRESHOLD

        # Opinion is defined as name of the subimage.
        self.decision = None
        self.decision_locker = threading.Lock()

        self.vlad_vectors = np.zeros((len(self.centroids), len(self.centroids[0])))

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
            photo = ia.add_gaussian_noise(photo)
        except Exception:
            pass
        
        return photo
    
    def calculate_random_velocity(self):
        """Calculate random velocity for the agent.

        Returns:
            numpy.ndarray: Velocity vector in x and y directions.
        """
        velocity_gain = 3.0

        x = np.random.uniform(low = -1.0, high = 1.0)
        y = np.random.choice([-1, 1]) * np.sqrt(1.0 - x**2)

        return np.array([x, y], dtype = np.float32) * velocity_gain

    def move_in_thread(self):
        """Agent movement in separate thread.
        """
        def move():
            velocity = self.calculate_random_velocity()
            start_time, final_time = utils.set_timer(0.5, 7.0)

            while True:
                current_position = self.get_position()

                if self.position[0] + constants.AGENT_SIZE[0] / 2 >= self.original_image.shape[1] or self.position[0] - constants.AGENT_SIZE[0] <= 0:
                    velocity[0] *= -1
                if self.position[1] + constants.AGENT_SIZE[1] / 2 >= self.original_image.shape[0] or self.position[1] - constants.AGENT_SIZE[1] <= 1:
                    velocity[1] *= -1

                elapsed_time = time.time() - start_time
                if elapsed_time > final_time:
                    start_time, final_time = utils.set_timer(0.5, 7.0)
                    velocity = self.calculate_random_velocity()
                
                self.set_position(current_position + velocity)

                if self.do_stop:
                    return 
                
                time.sleep(0.01)

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
            descriptor_buffer_length = 50
            while not self.do_stop:
                photo = self.capture_photo()
                self.update_descriptors(photo)
                
                if len(self.descriptors) >= descriptor_buffer_length:
                    self.calculate_vlad_vector()
                    cosine_similarity = utils.evaluate_similarity(self.vlad_vectors, self.codebook)
                    personal_information = dm.calculate_personal_information(cosine_similarity)
                    self.set_personal_information(personal_information)
                    self.descriptors = []
                
                time.sleep(0.15)

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
                self.social_information_dict = {}
                self.set_state(constants.RECEIVING_STATE)
                start_time = time.time()
                elapsed_time = 0
                while elapsed_time < constants.RECEIVING_TIME:
                    if self.do_stop:
                        break
                    elapsed_time = time.time() - start_time
                    time.sleep(0.1)

                # Decision making, based on personal and social information
                self.set_state(constants.DECISION_MAKING_STATE)
                
                received_opinions = np.array(list(self.social_information_dict.values()))
                new_opinion = dm.asodm(self.get_personal_information(), self.social_factor, received_opinions, len(self.codebook), self.decision_threshold)
                self.opinions += new_opinion

                # self.update_decision_threshold(received_opinions)

                self.set_decision(self.calculate_decision())

                # Disseminate 
                self.set_state(constants.DISSEMINATION_STATE)
                dissemination_time = self.get_dissemination_time()
                start_time = time.time()
                elapsed_time = 0
                while elapsed_time < dissemination_time:
                    if self.do_stop:
                        break
                    elapsed_time = time.time() - start_time
                    time.sleep(0.1)

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
                        self.social_information_dict[agent.id] = agent_information 
    
    def calculate_decision(self):
        """Calculate current decision based on opinions.

        Returns:
            int: Id of the decided subimage or None if no decision has been made yet.
        """
        if np.max(self.opinions) >= self.decision_threshold[np.argmax(self.opinions)]:
            return np.argmax(self.opinions)
        return None
    
    def update_decision_threshold(self, received_opinions):
        """Update decision threshold based on received opinions.

        Args:
            received_opinions (list): List of received opinions.
        """
        if not received_opinions.size:
            return
        
        max_counts = np.zeros(len(self.codebook))
        for opinions in received_opinions:
            max_counts[np.argmax(opinions)] += 1

        self.decision_threshold = self.decision_threshold - np.power(max_counts, 1.2) + 2
        np.clip(self.decision_threshold, 5.0, constants.INITIAL_DECISION_THRESHOLD, out = self.decision_threshold)
    
    def get_dissemination_time(self):
        """Calculate dissemination time, based on current opinion.

        Returns:
            float: Dissemination time in seconds.
        """
        decision_threshold = self.decision_threshold[np.argmax(self.opinions)]
        dissemination_time = constants.MAX_DISSEMINATION_TIME * np.max(self.opinions) / decision_threshold
        if dissemination_time > constants.MAX_DISSEMINATION_TIME:
            dissemination_time = constants.MAX_DISSEMINATION_TIME

        return dissemination_time
    
    def stop_threads(self):
        """Stop all running threads.
        """
        self.do_stop = True
        self.move_thread.join()
        self.decision_making_thread.join()
