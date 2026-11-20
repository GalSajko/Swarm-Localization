import cv2
import numpy as np

# Ranges of the randomized camera degradation. The repeatability of map features (feature_repeatability.py) covers the same ranges.
ZOOM_FACTOR_RANGE = (1.01, 1.08)
RESOLUTION_FACTOR_RANGE = (0.70, 0.85)
NOISE_SIGMA_RANGE = (4.0, 10.0)

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
    zoom_factor = 1.05,
    resolution_factor = 0.8,
    noise_sigma = 8.0,
):
    """Apply lightweight drone-camera degradation to a captured patch.

    Args:
        original_image (numpy.ndarray): Input image.
        zoom_factor (float, optional): Digital zoom multiplier. Defaults to 1.05.
        resolution_factor (float, optional): Downscale factor before upsampling. Defaults to 0.8.
        noise_sigma (float, optional): Gaussian noise sigma. Defaults to 8.0.

    Returns:
        numpy.ndarray: Augmented image.
    """
    height, width = original_image.shape[:2]
    degraded_width = max(1, int(round(width * resolution_factor)))
    degraded_height = max(1, int(round(height * resolution_factor)))

    augmented_image = zoom_in_image(original_image, zoom_factor = zoom_factor)
    augmented_image = pixelate_image(augmented_image, (degraded_width, degraded_height))

    original_image_float = augmented_image.astype(np.float32)
    gaussian_noise = np.random.normal(0.0, noise_sigma, augmented_image.shape).astype(np.float32)
    augmented_image = np.clip(original_image_float + gaussian_noise, 0, 255).astype(np.uint8)

    return augmented_image

def degrade_whole_image(original_image, zoom_factor, resolution_factor, noise_sigma, random_generator):
    """Apply the degradation of 'augment_drone_capture' to a whole image without cropping it.

    A drone capture is zoomed by cropping its center and stretching it back to the original size. Here the whole image is enlarged by the zoom
    factor instead, which changes every feature in the same way but keeps the image borders. The result is therefore larger than the input.

    Args:
        original_image (numpy.ndarray): Input grayscale image.
        zoom_factor (float): Zoom multiplier.
        resolution_factor (float): Downscale factor before upsampling.
        noise_sigma (float): Gaussian noise sigma.
        random_generator (numpy.random.Generator): Random generator for the noise.

    Returns:
        numpy.ndarray: Degraded image, enlarged by the zoom factor.
    """
    height, width = original_image.shape[:2]
    zoomed_width = int(round(width * zoom_factor))
    zoomed_height = int(round(height * zoom_factor))
    zoomed_image = cv2.resize(original_image, (zoomed_width, zoomed_height), interpolation = cv2.INTER_LINEAR)

    degraded_image = pixelate_image(zoomed_image, (zoomed_width * resolution_factor, zoomed_height * resolution_factor)).astype(np.float32)
    degraded_image += random_generator.normal(0.0, noise_sigma, degraded_image.shape).astype(np.float32)

    return np.clip(degraded_image, 0, 255).astype(np.uint8)
