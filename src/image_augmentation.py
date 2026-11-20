import cv2
import numpy as np

# Ranges of the mild degradation used for the repeat counts of map features (feature_repeatability.py, degrade_whole_image).
ZOOM_FACTOR_RANGE = (1.01, 1.08)
RESOLUTION_FACTOR_RANGE = (0.70, 0.85)
NOISE_SIGMA_RANGE = (4.0, 10.0)

# Ranges of the cross-sensor degradation of an agent's photo (degrade_drone_photo), i.e. the differences between the satellite map and the drone
# camera. Geometry is one homography from the photo to the map.
PHOTO_SCALE_RANGE = (0.85, 1.20)       # altitude error and different ground resolution (> 1 shows a smaller map area)
PHOTO_HEADING_ERROR_DEGREES = 5.0      # +- rotation
PHOTO_TILT_CORNER_SHIFT = 0.03         # perspective: every photo corner moves by up to this share of the view size
PHOTO_CENTER_SHIFT_PIXELS = 1.0        # +- shift of the view center
PHOTO_GAMMA_RANGE = (0.7, 1.4)
PHOTO_CONTRAST_RANGE = (0.75, 1.3)
PHOTO_BRIGHTNESS_SHIFT = 20.0          # +- gray levels
PHOTO_ILLUMINATION_GRADIENT = 0.15     # +- relative brightness change across the view (sun angle, vignetting, cloud shadow)
PHOTO_BLUR_SIGMA_RANGE = (0.4, 1.2)    # defocus and motion
PHOTO_RESOLUTION_FACTOR_RANGE = (0.5, 0.8)
PHOTO_NOISE_SIGMA_RANGE = (5.0, 14.0)
PHOTO_JPEG_QUALITY_RANGE = (40, 80)
# Extra map border around the view, so that the warped photo has no empty corners.
PHOTO_WARP_MARGIN = 40

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

def degrade_drone_photo(map_image, view_x0, view_y0, view_size):
    """Take an agent's photo of a map view, with the differences between the satellite map and the drone camera.

    Geometry: the view is scaled, rotated by a heading error, tilted (perspective) and shifted around its center. Photometry: gamma, contrast,
    brightness and an illumination gradient. Optics and sensor: blur, resolution loss, noise and JPEG compression. The ranges are the PHOTO_*
    constants; the random values come from numpy's global generator.

    Args:
        map_image (numpy.ndarray): Grayscale map image.
        view_x0 (int): Map x of the view's top-left pixel.
        view_y0 (int): Map y of the view's top-left pixel.
        view_size (tuple[int, int]): View width and height in pixels; the photo has the same size.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: The photo (uint8) and the 3 x 3 homography from photo coordinates to map coordinates (both in the
            convention where pixel centers are at integer + 0.5).
    """
    width, height = view_size
    map_height, map_width = map_image.shape[:2]
    margin_x0, margin_y0 = max(0, view_x0 - PHOTO_WARP_MARGIN), max(0, view_y0 - PHOTO_WARP_MARGIN)
    margin_x1 = min(map_width, view_x0 + width + PHOTO_WARP_MARGIN)
    margin_y1 = min(map_height, view_y0 + height + PHOTO_WARP_MARGIN)
    surroundings = map_image[margin_y0:margin_y1, margin_x0:margin_x1]

    # Map positions of the photo corners.
    scale = np.random.uniform(*PHOTO_SCALE_RANGE)
    heading = np.deg2rad(np.random.uniform(-PHOTO_HEADING_ERROR_DEGREES, PHOTO_HEADING_ERROR_DEGREES))
    center = np.array([view_x0 + width / 2, view_y0 + height / 2]) + np.random.uniform(-PHOTO_CENTER_SHIFT_PIXELS, PHOTO_CENTER_SHIFT_PIXELS, 2)
    corners = np.array([[-width / 2, -height / 2], [width / 2, -height / 2], [width / 2, height / 2], [-width / 2, height / 2]])
    corners += np.random.uniform(-PHOTO_TILT_CORNER_SHIFT, PHOTO_TILT_CORNER_SHIFT, (4, 2)) * [width, height]
    rotation = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
    map_corners = center + (corners / scale) @ rotation.T

    photo_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype = np.float32)
    photo_to_map = cv2.getPerspectiveTransform(photo_corners, map_corners.astype(np.float32))
    photo_to_surroundings = cv2.getPerspectiveTransform(photo_corners, (map_corners - [margin_x0, margin_y0]).astype(np.float32))
    photo = cv2.warpPerspective(
        surroundings, photo_to_surroundings, (width, height), flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode = cv2.BORDER_REFLECT,
    ).astype(np.float32)

    # Photometric differences between the two sensors and acquisition times.
    photo = 255.0 * (photo / 255.0) ** np.random.uniform(*PHOTO_GAMMA_RANGE)
    mean = photo.mean()
    photo = (photo - mean) * np.random.uniform(*PHOTO_CONTRAST_RANGE) + mean + np.random.uniform(-PHOTO_BRIGHTNESS_SHIFT, PHOTO_BRIGHTNESS_SHIFT)
    gradient_x, gradient_y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    direction = np.random.uniform(0, 2 * np.pi)
    gradient = np.random.uniform(-PHOTO_ILLUMINATION_GRADIENT, PHOTO_ILLUMINATION_GRADIENT)
    photo *= 1 + gradient * (np.cos(direction) * gradient_x + np.sin(direction) * gradient_y)
    photo = np.clip(photo, 0, 255).astype(np.uint8)

    # Optics and sensor.
    photo = cv2.GaussianBlur(photo, (0, 0), np.random.uniform(*PHOTO_BLUR_SIGMA_RANGE))
    resolution_factor = np.random.uniform(*PHOTO_RESOLUTION_FACTOR_RANGE)
    low_resolution_size = (max(1, round(width * resolution_factor)), max(1, round(height * resolution_factor)))
    photo = cv2.resize(cv2.resize(photo, low_resolution_size, interpolation = cv2.INTER_AREA), (width, height), interpolation = cv2.INTER_LINEAR)
    photo = photo.astype(np.float32) + np.random.normal(0.0, np.random.uniform(*PHOTO_NOISE_SIGMA_RANGE), photo.shape)
    photo = np.clip(photo, 0, 255).astype(np.uint8)
    _, jpeg = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, int(np.random.uniform(*PHOTO_JPEG_QUALITY_RANGE))])

    return cv2.imdecode(jpeg, cv2.IMREAD_GRAYSCALE), photo_to_map

def degrade_whole_image(original_image, zoom_factor, resolution_factor, noise_sigma, random_generator):
    """Apply a mild degradation (zoom, resolution loss, noise) to a whole image without cropping it, for the repeat counts of map features.

    The whole image is enlarged by the zoom factor, which changes every feature in the same way but keeps the image borders. The result is
    therefore larger than the input.

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
