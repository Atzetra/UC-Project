"""sentinel_water dataset."""

from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
import tifffile as tiff


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for sentinel_water dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Download the file from https://drive.google.com/file/d/1N1NahlAvH3W00dDMt4CJP7c4TCyhFUL_/view.
    Extract the file and place the directory in the `manual_dir/` (~/tensorflow_datasets/downloads/manual/)
    Run ../dataset_processing.py on the image and mask files to get the synthetic data."""

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(sentinel_water): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "image": tfds.features.Tensor(
                        shape=(256, 256, 4), dtype=tf.float32
                    ),
                    "mask": tfds.features.Tensor(
                        shape=(256, 256, 1), dtype=tf.float32
                    ),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("image", "mask"),  # Set to `None` to disable
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        dl_manager.manual_dir
        path = dl_manager.manual_dir / "sentinel_water"

        return {
            "train": self._generate_examples(
                image_path=path / "blocks", mask_path=path / "masks"
            ),
        }

    def _generate_examples(self, image_path: Path, mask_path: Path):
        """Yields examples."""
        # TODO(sentinel_water): Yields (key, example) tuples from the dataset
        for _, _, files in tf.io.gfile.walk(image_path):
            for file in files:
                # Select only tif files.
                if file.endswith(".tif"):
                    yield file, {
                        "image": tiff.imread(Path(image_path) / file),
                        "mask": tiff.imread(Path(mask_path) / file),
                    }
