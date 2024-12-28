# Action Recognition NN

This project is a workaround for a custom action recognition Neural Network, working with Torch and aims to classify given still images for human actions.

## Dataset

The dataset used for this example is the [Stanford 40 Actions](http://vision.stanford.edu/Datasets/40actions.html), containing more than 9.500 images, capturing human actions. To download the dataset check [here](http://vision.stanford.edu/Datasets/Stanford40.zip)

## Installation

The default approach shown in the readme is based on `uv`, however one can install all dependencies using other tools as well.

### Set Dataset Paths

First, we start by exporting the xml and image directories so our script is able to fetch the two different dirs:

```bash
export ROOT_PATH="path-to-downloaded-stanford40-dataset"
export IMAGE_FILES_PATH="relative-path-stanford40-jpeg-images-dir"
export XML_FILES_PATH="relative-path-stanford40-xml-annotations-dir"
```

### Install Dependencies

You can create a new virtual environment using `uv` and then sync all the depenedencies:

```bash
uv venv --python 3.10
uv sync
```

## Usage

In order to run the Neural Network example you can simply run:

```bash
uv run main.py
```

### Available Configurations:

The script can be configured through the usage of environment variables. The env vars used are:

#### Data Ratio

| Name               | Description                                      | Type    | Default |
| ------------------ | ------------------------------------------------ | ------- | ------- |
| `VALIDATION_RATIO` | The percentage of train data used for validation | `float` | 0.20    |
| `TEST_RATIO`       | The percentage of full data used for test        | `float` | 0.15    |

#### Directory Paths

| Name | Description | Type | Default |
| `ROOT_PATH` | The root path of the dataset | `string` | "Stanford40" |
| `IMAGE_FILES_PATH` | The path to the jpeg images dir | `string` | "JPEGImages" |
| `XML_FILES_PATH` | The path to the xml annotations dir | `string` | "XMLAnnotations" |

#### Neural Network HyperParameters

| Name | Description | Type | Default |
| `NN_IMAGE_READ_MODE` | Mode of image read (GRAY or RGB) | `str` | "RGB" |
| `NN_LEARNING_RATE` | The percentage of learning rate | `float` | 0.001 |
| `NN_TRANSFORM_RESIZE` | The size of the image transformation | `int` | 64 |
| `NN_TRAIN_BATCH_SIZE` | The batch size used for training | `int` | 256 |
| `NN_TEST_BATCH_SIZE` | The batch size used for testing | `int` | 50 |
| `NN_NUM_EPOCHS` | The number of epochs during training | `int` | 5 |
