# ## Prepare dataloader

import os

import tensorflow as tf

import autokeras as ak
import helpers.datagen as datagen

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

dataset = datagen.get_dataset()

imagesegmenter = ak.ImageSegmenter(
    num_classes=2, loss="binary_crossentropy", metrics=["accuracy"]
)

imagesegmenter.fit(
    dataset,
    epochs=10,
)
