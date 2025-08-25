import cv2
import numpy as np
import random
from scipy.cluster.vq import vq
import threading
import time

import constants
import image_augmentation as ia
import utils

class Agent:
    def __init__(self, original_image, algorithm, centroids, codebook, start_position, id, is_forgetting, weights = None):
        self.id = id 

        self.do_stop = False

        self.algorithm = algorithm

        self.position_locker = threading.Lock()
        self.position = start_position

        self.centroids = centroids
        self.weights = weights
        self.histogram = np.zeros(len(self.centroids))
        self.codebook = codebook

        self.sift_extractor = cv2.SIFT_create()
        self.original_image = original_image

        self.state = constants.EXPLORATION_STATE
        self.state_locker = threading.Lock()
        self.exploration_state_counter = 0

        # private information is defined as vector of cosine similarities to all base images.
        self.private_information = []
        self.private_information_locker = threading.Lock()
        self.social_information_dict = {}

        # Opinion is defined as name of the subimage.
        self.opinion = None
        self.opinion_locker = threading.Lock()
        self.aggregated_decision_vectors = []
        self.decision_vector = np.zeros(len(codebook))
        self.forgetting_factors = np.zeros_like(self.decision_vector)
        self.is_forgetting = is_forgetting

        self.vlad_vectors = np.zeros((len(self.centroids), len(self.centroids[0])))

        self.color = None
        self.color_locker = threading.Lock()

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
    
    def get_private_information(self):
        """Get current value of private information in a thread-safe way.

        Returns:
            list: private information.
        """
        with self.private_information_locker:
            return self.private_information
    
    def set_private_information(self, value):
        """Set private information in a thread-safe way.

        Args:
            value (list): private information.
        """
        with self.private_information_locker:
            self.private_information = value
    
    def get_opinion(self):
        """Get current opinion in a thread-safe way.

        Returns:
            string: Current opinion.
        """
        with self.opinion_locker:
            return self.opinion
        
    def set_opinion(self, value):
        """Set current opinion in a thread-safe way.

        Args:
            value (string): New opinion.
        """
        with self.opinion_locker:
            self.opinion = value

    def get_color(self):
        """Get agent's current color in a thread-safe way.

        Returns:
            tuple: Tuple representing RGB color.
        """
        with self.color_locker:
            return self.color

    def set_color(self, value):
        """Set agent's color in a thread-safe way.

        Args:
            value (tuple): Desired RGB color tuple.
        """
        with self.color_locker:
            self.color = value
    
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

    def start_exploration_thread(self, exploration_time):
        """Start thread for exploring the environment.

        exploration_period (float): Desired exploration time.

        Returns:
            Thread: Thread object.
        """
        def exploration(exploration_time):
            """Take photos and calculate sift descriptors.
            """
            self.set_state(constants.EXPLORATION_STATE)
            loop_period = 1.0
            exploration_start_time = time.time()
            exploration_elapsed_time = 0

            while exploration_elapsed_time < exploration_time:
                loop_start_time = time.time()

                current_position = self.get_position()

                photo = self.original_image[
                    int(current_position[1] - constants.AGENT_SIZE[0] / 2) : int(current_position[1] + constants.AGENT_SIZE[0] / 2),
                    int(current_position[0] - constants.AGENT_SIZE[1] / 2) : int(current_position[0] + constants.AGENT_SIZE[1] / 2)
                    ]
                
                try:
                    photo = ia.pixelate_image(photo, (15, 15))
                    photo = ia.add_gaussian_noise(photo)
                except Exception:
                    pass
                
                if self.algorithm == constants.BOW_ALGORITHM:
                    weighted_histogram = self.calculate_weighted_histogram(photo)
                    if weighted_histogram.any():
                        distance_vector = self.evaluate_similarity(weighted_histogram)
                        self.set_private_information(distance_vector)
                elif self.algorithm == constants.VLAD_ALGORITHM:
                    self.update_vlad_vector(photo)
                
                exploration_elapsed_time = time.time() - exploration_start_time
                utils.sleep_with_adjustable_period(loop_period, loop_start_time)

                if self.do_stop:
                    return
        
        exploration_thread = threading.Thread(target = exploration, name = 'photo_thread_' + self.id, args = (exploration_time, ), daemon = True)
        exploration_thread.start()
        
        return exploration_thread
    
    def calculate_weighted_histogram(self, photo):
        """Calculate weighted histogram based on BOW algorithm.

        Args:
            photo (numpy.ndarray): Photo taken by the agent.

        Returns:
            numpy.ndarray: Weighted histogram.
        """
        _, descriptors = self.sift_extractor.detectAndCompute(photo.astype(np.uint8), None)

        if descriptors is None:
            return np.zeros(len(self.centroids))

        centroids_indexes_of_descriptors, _ = vq(descriptors, self.centroids)
        for centroid_index in centroids_indexes_of_descriptors:
            self.histogram[centroid_index] += 1
 
        if self.histogram.any():
            return self.histogram * self.weights
        
        return np.zeros(len(self.centroids))

    def update_vlad_vector(self, photo):
        _, descriptors = self.sift_extractor.detectAndCompute(photo.astype(np.uint8), None)
        if descriptors is None:
            return np.zeros(len(self.centroids))
        mapped_descriptors, _ = vq(descriptors, self.centroids)

        for descriptor_index, descriptor in enumerate(descriptors):
            centroid_index = mapped_descriptors[descriptor_index]
            difference = np.round(descriptor - self.centroids[centroid_index], 4)
            self.vlad_vectors[centroid_index] += difference

    
    def evaluate_similarity(self, weighted_histogram):
        """Calculate similarity between weighted histogram vector and codebook vector for each image.

        Args:
            weighted_histogram (numpy.ndarray): Weighted histogram vector.

        Returns:
            numpy.ndarray: Vector of similarities between weighted histogram and each image.
        """
        similarity_vector = []
        for codebook_vector in self.codebook.values():
            similarity_vector.append(np.round(utils.calculate_cosine_similarity(weighted_histogram, codebook_vector), 3))
        return np.array(similarity_vector)
            
    def start_decision_making_thread(self):
        """Thread for updating agent's opinion, based on private and social information.
        """
        def decision_making_in_loop():
            while True:
                # Explore the environment and calculate personal information
                _, exploration_time = utils.set_timer(7.0, 15.0 - self.exploration_state_counter)
                exploration_thread = self.start_exploration_thread(exploration_time)
                exploration_thread.join()
                self.exploration_state_counter = self.exploration_state_counter + 1 if self.exploration_state_counter < 7 else self.exploration_state_counter
                if self.do_stop:
                    return

                # Receive social information. Method for collecting social information is called from the simulation module, because it needs access to others Agents objects. 
                # Here we only set state and reset social information buffer.
                self.social_information_dict = {}
                self.set_state(constants.RECEIVING_STATE)
                start_time = time.time()
                elapsed_time = 0
                while elapsed_time < constants.RECEIVING_TIME:
                    if self.do_stop:
                        return
                    elapsed_time = time.time() - start_time
                    time.sleep(0.1)

                # Decision making, based on personal and social information
                self.set_state(constants.DECISION_MAKING_STATE)
                self.update_opinion()

                # Disseminate 
                self.set_state(constants.DISSEMINATION_STATE)
                dissemination_time = self.get_dissemination_time()
                start_time = time.time()
                elapsed_time = 0
                while elapsed_time < dissemination_time:
                    if self.do_stop:
                        return
                    elapsed_time = time.time() - start_time
                    time.sleep(0.1)
        
        decision_making_thread = threading.Thread(target = decision_making_in_loop, name = 'decision_making_thread_' + self.id, daemon = True)
        decision_making_thread.start()

        return decision_making_thread
                
    def collect_social_information(self, agents):
        """Receive social information from the neighbours that are currently in dissemination state.

        Args:
            agents (list): List of all Agent objects.
        """
        for agent in agents:
            if agent.id == self.id:
                continue
            if np.linalg.norm(self.get_position() - agent.get_position()) <= constants.COMMUNICATION_RANGE:
                if agent.get_state() == constants.DISSEMINATION_STATE:
                    agent_information = agent.get_private_information()
                    if len(agent_information):
                        self.social_information_dict[agent.id] = agent_information

    def update_opinion(self):
        """Update opinion based on private and social parts of information.
        """
        private_information = np.array(self.get_private_information())
        social_information = np.array(list(self.social_information_dict.values()))

        if not len(social_information) and not len(private_information):
            return
        
        if not social_information.any() and not private_information.any():
            return
        
        if len(social_information):
            if len(private_information):
                total_information = np.vstack((private_information, social_information))
            else:
                total_information = np.copy(social_information)

            information_histogram = np.zeros_like(total_information[0])
            selection_weights = np.zeros(len(total_information))

            for i, information in enumerate(total_information):
                max_value, difference, max_index = utils.get_biggest_element_and_difference_to_second(information)
                selection_weights[i] = max_value
                information_histogram[max_index] += max_value

            norm_selection_weights = (selection_weights / np.linalg.norm(selection_weights)) ** 2
            # new_private_information = random.choices(population = total_information, weights = norm_selection_weights)[0]
            new_private_information = np.average(total_information, axis = 0, weights = norm_selection_weights)

            self.set_private_information(new_private_information)

        else:
            if len(private_information):
                information_histogram = np.zeros_like(private_information)
                max_index = np.where(private_information == max(private_information))[0][0]
                information_histogram[max_index] += max(private_information)

        if self.is_forgetting:
            information_histogram_max_index = np.where(information_histogram == max(information_histogram))[0][0] 
            mask = np.ones(self.decision_vector.shape, dtype = bool)
            mask[information_histogram_max_index] = False
            self.forgetting_factors[mask] += 1
            self.forgetting_factors[information_histogram_max_index] = 0
            self.decision_vector[mask] -= np.power(0.5, 7 - self.forgetting_factors[mask])

        self.decision_vector += information_histogram
        self.decision_vector[np.where(self.decision_vector < 0)] = 0

        self.aggregated_decision_vectors.append(np.copy(self.decision_vector))

        try:
            opinion = list(self.codebook.keys())[np.argmax(self.decision_vector)]
        except ValueError:
            opinion = None 
        self.set_opinion(opinion)

        # print(f"Agent: {self.id}, opinion: {opinion}")
        
        # if opinion is not None:
        #     self.set_color(constants.AGENT_COLOR_OPINION_DICT[opinion])
    
    def get_dissemination_time(self):
        """Calculate dissemination time, based on current opinion.

        Returns:
            float: Dissemination time in seconds.
        """
        dissemination_time = 10 * np.arctan(10 * max(self.decision_vector) + 0.0001) 
        if dissemination_time > constants.MAX_DISSEMINATION_TIME:
            dissemination_time = constants.MAX_DISSEMINATION_TIME

        return dissemination_time
    
    def stop_threads(self):
        self.do_stop = True
        self.move_thread.join()
        self.decision_making_thread.join()
        