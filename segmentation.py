# ## Prepare dataloader

import os

import tensorflow as tf

import autokeras as ak
import helpers.datagen as datagen

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

dataset = datagen.get_dataset()

# imagesegmenter = ak.ImageSegmenter(
#     num_classes=2, loss={'classification_head_1: binary_crossentropy'} metrics=["accuracy"]
# )

input_node = ak.ImageInput()
output_node = ak.SegmentationBlock()(input_node)
output_node = ak.SegmentationHead(num_classes=2)(output_node)

imagesegmenter = ak.AutoModel(
    inputs=input_node, outputs=output_node, overwrite=True, max_trials=1
)


imagesegmenter.fit(
    dataset,
    epochs=10,
)
