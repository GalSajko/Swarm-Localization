import cv2
import math
import numpy as np

ORIGINAL_IMAGES_DIR = 'data/original_images/'
SUBIMAGES_DIR = 'data/subimages/'
COLORED_SATELITE_IMAGE_NAME = 'colored_satelite_image.png'
BW_SATELITE_IMAGE_NAME = 'bw_satelite_image.png'

def convert_image_to_bw_and_save(original_image_path, original_image_name, path_to_save = None):
    """Convert given image in black and white and save it into desired directory.

    Args:
        original_image_path (string): Path of the original image.
        original_image_name (string): Name of the original image
        path_to_save (string, optional): Path to the desired directory of the converted image. 
        If None, converted image will be saved in the same directory as original one. Defaults to None.
    """
    image = cv2.imread(original_image_path + original_image_name)
    bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if not path_to_save:
        path_to_save = original_image_path
    
    cv2.imwrite(path_to_save + BW_SATELITE_IMAGE_NAME, bw_image)

def divide_image_on_subimages_and_save(image, divide_on):
    """Divide original image on desired number of subimages.

    Args:
        image (numpy.ndarray): Original image as numpy array.
        divide_on (tuple[int, int]): Number of subimages in horizontal and vertical direction.
    """
    shape = image.shape
    
    subimage_x_size = math.floor(shape[1] / divide_on[0])
    subimage_y_size = math.floor(shape[0] / divide_on[1])
    
    for y in range(divide_on[1]):
        for x in range(divide_on[0]):
            sub_image = image[y * subimage_y_size : (y + 1) * subimage_y_size, x * subimage_x_size : (x + 1) * subimage_x_size]
            cv2.imwrite(SUBIMAGES_DIR + f"{x + 1}_{y + 1}.png", sub_image)

if __name__ == '__main__':
    bw_satelite_image = cv2.imread(ORIGINAL_IMAGES_DIR + BW_SATELITE_IMAGE_NAME)[:, :, 0]
    divide_image_on_subimages_and_save(bw_satelite_image, (5, 2))