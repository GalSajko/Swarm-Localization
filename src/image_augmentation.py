import cv2
import numpy as np

def pixelate_image(original_image, new_size):
    """Reduce image resolution and upscale it back to the original size.

    Args:
        original_image (numpy.ndarray): Input image.
        new_size (tuple[int, int]): Temporary downscaled width and height.

    Returns:
        numpy.ndarray: Pixelated image.
    """
    height, width = original_image.shape[:2]
    new_width, new_height = new_size
    new_width = max(1, int(round(new_width)))
    new_height = max(1, int(round(new_height)))

    # A mild pre-blur better approximates optical/sensor resolution loss than
    # nearest-neighbor blockiness alone.
    blurred_image = cv2.GaussianBlur(original_image, (3, 3), 0.8)
    temp_resized_image = cv2.resize(blurred_image, (new_width, new_height), interpolation = cv2.INTER_AREA)
    resized_image = cv2.resize(temp_resized_image, (width, height), interpolation = cv2.INTER_LINEAR)

    return resized_image

def add_gaussian_noise(original_image):
    """Add Gaussian image noise with a fixed default sigma.

    Args:
        original_image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Noisy image.
    """
    original_image = original_image.astype(np.float32)
    mean = 0.0
    sigma = 10.0
    gaussian_noise = np.random.normal(mean, sigma, original_image.shape).astype(np.float32)
    noisy_image = np.clip(original_image + gaussian_noise, 0, 255).astype(np.uint8)

    return noisy_image

def zoom_in_image(original_image, zoom_factor = 1.1):
    """Crop around the center and resize back to simulate digital zoom.

    Args:
        original_image (numpy.ndarray): Input image.
        zoom_factor (float, optional): Zoom multiplier. Defaults to 1.1.

    Returns:
        numpy.ndarray: Zoomed image.
    """
    height, width = original_image.shape[:2]

    if zoom_factor <= 1.0:
        return original_image.copy()

    cropped_width = max(1, int(round(width / zoom_factor)))
    cropped_height = max(1, int(round(height / zoom_factor)))

    start_x = (width - cropped_width) // 2
    start_y = (height - cropped_height) // 2
    end_x = start_x + cropped_width
    end_y = start_y + cropped_height

    cropped_image = original_image[start_y:end_y, start_x:end_x]
    zoomed_image = cv2.resize(cropped_image, (width, height), interpolation = cv2.INTER_LINEAR)

    return zoomed_image

def augment_drone_capture(
    original_image,
    randomize = False,
    zoom_factor = 1.05,
    resolution_factor = 0.8,
    noise_sigma = 8.0,
):
    """Apply lightweight drone-camera degradation to a captured patch.

    Args:
        original_image (numpy.ndarray): Input image.
        randomize (bool, optional): Randomize augmentation strengths. Defaults to False.
        zoom_factor (float, optional): Digital zoom multiplier. Defaults to 1.05.
        resolution_factor (float, optional): Downscale factor before upsampling. Defaults to 0.8.
        noise_sigma (float, optional): Gaussian noise sigma. Defaults to 8.0.

    Returns:
        numpy.ndarray: Augmented image.
    """
    if randomize:
        zoom_factor = np.random.uniform(1.01, 1.08)
        resolution_factor = np.random.uniform(0.70, 0.85)
        noise_sigma = np.random.uniform(4.0, 10.0)

    height, width = original_image.shape[:2]
    degraded_width = max(1, int(round(width * resolution_factor)))
    degraded_height = max(1, int(round(height * resolution_factor)))

    augmented_image = zoom_in_image(original_image, zoom_factor = zoom_factor)
    augmented_image = pixelate_image(augmented_image, (degraded_width, degraded_height))

    original_image_float = augmented_image.astype(np.float32)
    gaussian_noise = np.random.normal(0.0, noise_sigma, augmented_image.shape).astype(np.float32)
    augmented_image = np.clip(original_image_float + gaussian_noise, 0, 255).astype(np.uint8)

    return augmented_image
