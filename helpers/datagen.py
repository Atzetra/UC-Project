import os
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
import tifffile as tiff
from skimage.transform import resize

IMAGE_DIR = "/home/enzo/tensorflow_datasets/downloads/manual/sentinel_water"


def load_data(path):
    """Load data from path."""
    images = sorted(glob(os.path.join(path, "blocks", "*.tif")))
    masks = sorted(glob(os.path.join(path, "masks", "*.tif")))
    return images, masks


def read_image(image_path):
    """Read image from path."""
    image = tiff.imread(image_path)
    image = resize(image, (128, 128, 4))

    # normalize image with values between uint16
    image = image / 65535.0
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def read_mask(image_path):
    """Read image from path."""
    image = tiff.imread(image_path)
    image = resize(image, (128, 128, 1))

    # normalize image with values between uint16
    image = image / 255

    # Change the values such that 0 and one are segmentation classes
    image[image > 0.5] = 1
    image[image <= 0.5] = 0

    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def preprocessing(image_path, mask_path):
    """Preprocessing function for tf.data.Dataset."""

    def parse_with_opencv(image_path, mask_path):
        image = read_image(image_path.decode("utf-8"))
        mask = read_mask(mask_path.decode("utf-8"))
        return image, mask

    image, mask = tf.numpy_function(
        parse_with_opencv,
        [image_path, mask_path],
        Tout=[tf.float32, tf.float32],
    )

    image.set_shape([128, 128, 4])
    mask.set_shape([128, 128, 1])

    return image, mask


def tf_dataset(x, y):
    """Create tf.data.Dataset."""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=1000)

    # map the preprocessing function
    dataset = dataset.map(
        preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def get_dataset():
    """Get dataset."""
    images, masks = load_data(IMAGE_DIR)
    dataset = tf_dataset(images, masks)
    return dataset


if __name__ == "__main__":
    images, masks = load_data(IMAGE_DIR)
    print(f"Images: {len(images)}, Masks: {len(masks)}")

    dataset = get_dataset()

    print(dataset)

    print(images[0], masks[0])
