import constants
import numpy as np


class Agent:
    def __init__(self, image, start_position):
        self.position = start_position
        self.velocity = self.calculate_initial_velocity()

        self.size = (50, 50)

        self.image = image
    
    def calculate_initial_velocity(self):
        """Calculate random initial velocity for the agent.

        Returns:
            numpy.ndarray: Velocity vector in x and y directions.
        """
        x = np.random.uniform(low = -1.0, high = 1.0)
        y = np.random.choice([-1, 1]) * np.sqrt(1.0 - x**2)

        return np.array([x, y], dtype = np.float32)

    def move(self):
        """Update agents position. Prevent agent going outside of the image.
        """
        if self.position[0] + constants.AGENT_SIZE[0] / 2 >= self.image.shape[1] or self.position[0] - constants.AGENT_SIZE[0] <= 0:
            self.velocity[0] *= -1
        if self.position[1] + constants.AGENT_SIZE[1] / 2 >= self.image.shape[0] or self.position[1] - constants.AGENT_SIZE[1] <= 1:
            self.velocity[1] *= -1
        
        self.position += self.velocity
