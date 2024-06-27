import constants
import cv2
import numpy as np

from agent import Agent

IMAGE_DIR = "data/subimages/1_1.png"

def calculate_initial_poisitions(image):
    number_in_x = 4
    number_in_y = 5

    initial_padding = 350

    x_positions = np.linspace(initial_padding, image.shape[1] - initial_padding, number_in_x)
    y_positions = np.linspace(initial_padding, image.shape[0] - initial_padding, number_in_y)

    positions = []

    for x in x_positions:
        for y in y_positions:
            positions.append(np.array([x, y]))
    
    return np.asarray(positions)



if __name__ == "__main__":
    original_image = cv2.imread(IMAGE_DIR)
    positions = calculate_initial_poisitions(original_image)

    agents = [Agent(original_image, positions[i]) for i in range(20)]

    while True:
        image_copy = np.copy(original_image)
        for agent in agents:
            agent.move()

            start_point = np.array(agent.position - constants.AGENT_SIZE / 2).astype(np.int32)
            end_point = np.array(agent.position + constants.AGENT_SIZE / 2).astype(np.int32)
            cv2.rectangle(image_copy, start_point, end_point, (0, 0, 255), 5)
        
        cv2.imshow("img", image_copy)
        cv2.waitKey(5)    