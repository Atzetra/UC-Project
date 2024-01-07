import tensorflow_datasets as tfds

import autokeras as ak

# Prepare dataloader
classifier = ak.ImageClassifier(
    num_classes=10, loss="categorical_crossentropy", metrics=["accuracy"]
)

dataset = tfds.load("mnist", split="train", as_supervised=True)

classifier.fit(
    dataset,
    epochs=10,
)
