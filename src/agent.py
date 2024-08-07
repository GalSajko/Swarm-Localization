import cv2
import numpy as np
from scipy.cluster.vq import vq
import threading
import time

import constants
import image_augmentation as ia
import utils

class Agent:
    def __init__(self, original_image, centroids, weights, codebook, start_position, id):
        self.id = id 

        self.position_locker = threading.Lock()
        self.position = start_position

        self.centroids = centroids
        self.weights = weights
        self.histogram = np.zeros(len(self.centroids))
        self.weighted_histogram = np.zeros_like(self.histogram)
        self.codebook = codebook

        self.sift_extractor = cv2.SIFT_create()
        self.original_image = original_image

        self.state = constants.EXPLORATION_STATE
        self.state_locker = threading.Lock()

        # private information is defined as vector of cosine similarities to all base images.
        self.private_information = []
        self.private_information_locker = threading.Lock()
        self.social_information_dict = {}
        # Opinion is defined as name of the subimage.
        self.opinion = None
        self.opinion_locker = threading.Lock()

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
                
                photo = ia.pixelate_image(photo, (20, 20))
                photo = ia.add_gaussian_noise(photo)

                self.get_cosine_similarities_to_base_images(photo)
                
                exploration_elapsed_time = time.time() - exploration_start_time
                utils.sleep_with_adjustable_period(loop_period, loop_start_time)
        
        exploration_thread = threading.Thread(target = exploration, name = 'photo_thread_' + self.id, args = (exploration_time, ), daemon = True)
        exploration_thread.start()
        
        return exploration_thread

    def get_cosine_similarities_to_base_images(self, photo):
        """Calculate cosine similarities between taken photo and all of the baseline photos.

        Args:
            photo (numpy.ndarray): Photo taken by the agent.
        """
        _, descriptors = self.sift_extractor.detectAndCompute(photo.astype(np.uint8), None)
        if descriptors is not None:
            centroids_indexes_of_descriptors, _ = vq(descriptors, self.centroids)
            for centroid_index in centroids_indexes_of_descriptors:
                self.histogram[centroid_index] += 1

            if self.histogram.any():
                self.weighted_histogram = self.histogram * self.weights

                cosine_similarities = []
                for codebook_vector in self.codebook.values():
                    cosine_similarities.append(np.round(utils.calculate_cosine_similarity(self.weighted_histogram, codebook_vector), 4))

                self.set_private_information(cosine_similarities)
    
    def start_decision_making_thread(self):
        """Thread for updating agent's opinion, based on private and social information.
        """
        def decision_making_in_loop():
            while True:
                # Explore the environment and calculate personal information
                _, exploration_time = utils.set_timer(7.0, 15.0)
                exploration_thread = self.start_exploration_thread(exploration_time)
                exploration_thread.join()

                # Receive social information. Method for collecting social information is called from the simulation module, because it needs access to others Agents objects. 
                # Here we only set state and reset social information buffer.
                self.social_information_dict = {}
                self.set_state(constants.RECEIVING_STATE)
                time.sleep(constants.RECEIVING_TIME)

                # Decision making, based on personal and social information
                self.set_state(constants.DECISION_MAKING_STATE)
                self.update_opinion()

                # Disseminate 
                self.set_state(constants.DISSEMINATION_STATE)
                dissemination_time = self.get_dissemination_time()
                time.sleep(dissemination_time)
        
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
        private_information = self.get_private_information()
        social_information = list(self.social_information_dict.values())

        if not len(social_information) and len(private_information):
            total_information = private_information
        elif not len(private_information) and len(social_information):
            total_information = np.mean(social_information, axis = 0)
        elif not len(social_information) and not len(private_information):
            total_information = []
        else:
            # total_information = np.mean(np.vstack([social_information, private_information]), axis = 0)
            average_social_information = np.mean(social_information, axis = 0)
            total_information = private_information + average_social_information
        
        try:
            opinion = list(self.codebook.keys())[np.argmax(total_information)]
        except ValueError:
            opinion = None

        self.set_private_information(total_information)
        self.set_opinion(opinion)

        print(f"Agent: {self.id}, opinion: {opinion}")
        
        if opinion is not None:
            self.set_color(constants.AGENT_COLOR_OPINION_DICT[opinion])
    
    def get_dissemination_time(self):
        """Calculate dissemination time, based on current values of private information.

        Returns:
            float: Dissemination time in seconds.
        """
        private_information = self.get_private_information()

        if len(private_information):
            return constants.MAX_DISSEMINATION_TIME * max(private_information) * 2
        return 0
        