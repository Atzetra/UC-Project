import tensorflow as tf
from tensorflow.keras import layers
import autokeras as ak

num_classes = 30

def remote_sensing_segmentation_block(input_tensor):
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x

input_node = ak.ImageInput()
x = remote_sensing_segmentation_block(input_node)
x = remote_sensing_segmentation_block(x)
x = remote_sensing_segmentation_block(x)
output_node = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x) 

l
auto_model = ak.AutoModel(
    inputs=input_node,
    outputs=output_node,
    overwrite=True,
    max_trials=10  
)


auto_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

auto_model.fit(training_data, epochs=10, validation_data=validation_data)
