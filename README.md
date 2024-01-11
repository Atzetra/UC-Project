<p align="center">
  <img width="500" alt="logo" src="https://autokeras.com/img/row_red.svg"/>
</p>

# Implementing Image Segmentation into AutoKeras

---

# Installation
To install the dependencies, run
```shell
conda create --file autokeras.yml
pip install -e .
```

This repo has been developed in a WSL/Ubuntu environment using [this](https://hub.docker.com/layers/tensorflow/tensorflow/2.15.0-gpu/images/sha256-66b44c162783bb92ab6f44c1b38bcdfef70af20937089deb7bc20a4f3d7e5491?context=explore) dockerimage for cuda compute compatabilities.

## Structure
All the files needed for dataset generation of the water dataset are located in `./helpers/datagen.py`. This file assumes that you have the Sentinel 2 dataset downloaded and preprocessed.

Use the `./datasets/dataset_processing.ipynb` notebook for this. Here we detailed how to download and generate the files needed.

To run the sementation, run `./segmentation.py`. I (@Atzetra) got to the point we had to define output shapes and had to choose our own segmentation blocks/architecture to incorporate as detailed in our report. A stacktrace when running `/segmentation.py` should show you what we mean.

Running `./segmentation.py` should be enough to run if all the paths are correct. This script generates the tfds pipeline and preprocesses all data if the steps above for generating the sentinel-2 dataset are followed.

## Acknowledgements

I would like to thank the Urban Computing team for offering a very interesting course to be able to follow. As someone of the master's Bio-Pharmaceutical Sciences, it was fun, albeit very challanging, to learn.

- Enzo