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

        self.cosine_similarities = None
        self.cosine_similarities_locker = threading.Lock()

        self.color = None
        self.color_locker = threading.Lock()

        self.move_thread = self.move_in_thread()
        self.photo_thread = self.take_photo_in_thread()
    
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
    
    def get_cosine_similarities(self):
        """Get current value of cosine similarities in a thread-safe way.

        Returns:
            list: Cosine similarities.
        """
        with self.cosine_similarities_locker:
            return self.cosine_similarities
    
    def set_cosine_similarities(self, value):
        """Set cosine similarities in a thread-safe way.

        Args:
            value (list): Cosine similarities.
        """
        with self.cosine_similarities_locker:
            self.cosine_similarities = value
    
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

    def take_photo_in_thread(self):
        """Start taking photos in separate thread.
        """
        def take_photo_in_loop():
            """Take photos and calculate sift descriptors.
            """
            period = 1.0
            while True:
                start_time = time.time()

                current_position = self.get_position()

                photo = self.original_image[
                    int(current_position[1] - constants.AGENT_SIZE[0] / 2) : int(current_position[1] + constants.AGENT_SIZE[0] / 2),
                    int(current_position[0] - constants.AGENT_SIZE[1] / 2) : int(current_position[0] + constants.AGENT_SIZE[1] / 2)
                    ]
                
                photo = ia.pixelate_image(photo, (20, 20))
                photo = ia.add_gaussian_noise(photo)

                self.get_cosine_similarities_to_base_images(photo)

                time.sleep(period - (time.time() - start_time))
        
        photo_thread = threading.Thread(target = take_photo_in_loop, name = 'photo_thread_' + self.id, daemon = True)
        photo_thread.start()
        
        return photo_thread

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
            
            self.weighted_histogram = self.histogram * self.weights

            cosine_similarities = []
            for codebook_vector in self.codebook.values():
                cosine_similarities.append(np.round(utils.calculate_cosine_similarity(self.weighted_histogram, codebook_vector), 4))
            closest_image = list(self.codebook.keys())[np.argmax(cosine_similarities)]

            self.set_color(utils.AGENT_COLOR_OPINION_DICT[closest_image])

            # print(f"Agent: {self.id}, closest image: {closest_image}, similarities: {cosine_similarities}")

            self.set_cosine_similarities(cosine_similarities)