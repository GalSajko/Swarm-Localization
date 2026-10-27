"""Download the SpaceNet training images and save them as a grayscale array for training the VLAD centroids. Run from the project's root directory.

The Kaggle dataset 'sabermalek/spacenetsi' stores 1260 RGBA images (1024x1024) as a joblib-pickled array in 'ims.np'. The grayscale conversion
(cv2.COLOR_RGB2GRAY on the first three channels) is written to 'data/training_data/ims_gray.npy' with shape (1260, 1024, 1024) and dtype uint8, which is
the file that 'prepare_image_dataset.py' trains the centroids on. Requires the optional packages 'kagglehub' and 'joblib'.
"""

import argparse
import os

import cv2
import joblib
import kagglehub
import numpy as np

KAGGLE_DATASET = "sabermalek/spacenetsi"
KAGGLE_IMAGES_FILE = "ims.np"
OUTPUT_PATH = os.path.join("data", "training_data", "ims_gray.npy")
CHUNK_SIZE = 32

def convert_to_grayscale_and_save(images_path, output_path):
    """Convert RGBA images to grayscale chunk by chunk, so the dataset never has to fit in memory.

    Args:
        images_path (str): Path of the joblib-pickled array of shape (N, H, W, 4).
        output_path (str): Path of the .npy file to write, with shape (N, H, W).
    """
    images = joblib.load(images_path, mmap_mode = "r")
    os.makedirs(os.path.dirname(output_path), exist_ok = True)
    gray_images = np.lib.format.open_memmap(output_path, mode = "w+", dtype = np.uint8, shape = images.shape[:3])

    for start in range(0, len(images), CHUNK_SIZE):
        for index in range(start, min(start + CHUNK_SIZE, len(images))):
            gray_images[index] = cv2.cvtColor(np.ascontiguousarray(images[index][..., :3]), cv2.COLOR_RGB2GRAY)
        print(f"Converted {min(start + CHUNK_SIZE, len(images))}/{len(images)} images.", flush = True)

    gray_images.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Download the SpaceNet training images and save them as grayscale in data/training_data/.")
    parser.add_argument("--force", action = "store_true", help = f"Overwrite {OUTPUT_PATH} if it already exists.")
    force = parser.parse_args().force

    if os.path.exists(OUTPUT_PATH) and not force:
        print(f"{OUTPUT_PATH} already exists. Use --force to overwrite it.")
    else:
        # Only the images are needed, not the masks. kagglehub keeps the download in its own cache directory.
        downloaded_images_path = kagglehub.dataset_download(KAGGLE_DATASET, path = KAGGLE_IMAGES_FILE)
        convert_to_grayscale_and_save(downloaded_images_path, OUTPUT_PATH)
        print(f"Saved grayscale training images to {OUTPUT_PATH}.")
