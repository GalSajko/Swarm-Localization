import cv2
import numpy as np

def pixelate_image(original_image, new_size):
    height, width = original_image.shape[:2]
    new_width, new_height = new_size
    
    temp_resized_image = cv2.resize(original_image, (new_width, new_height), interpolation = cv2.INTER_LINEAR)
    resized_image = cv2.resize(temp_resized_image, (width, height), interpolation = cv2.INTER_NEAREST)

    return resized_image

def add_gaussian_noise(original_image):
    height, width, channels = original_image.shape
    mean = 0
    var = 0.2
    sigma = var ** 0.5
    gaussian_noise = np.random.normal(mean, sigma, (height, width, channels))
    gaussian_noise = gaussian_noise.reshape(height, width, channels)
    noisy_image = (original_image + gaussian_noise).astype(np.uint8)

    return noisy_image
