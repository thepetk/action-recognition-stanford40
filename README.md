# Action Recognition NN

This project is a workaround for a custom action recognition Neural Network, working with Torch and aims to classify given still images for human actions.

## Dataset

The dataset used for this example is the [Stanford 40 Actions](http://vision.stanford.edu/Datasets/40actions.html), containing more than 9.500 images, capturing human actions. To download the dataset check [here](http://vision.stanford.edu/Datasets/Stanford40.zip)

## Installation

The default approach shown in the readme is based on `uv`, however one can install all dependencies using other tools as well.

### Set Dataset Paths

First, we start by exporting the xml and image directories so our script is able to fetch the two different dirs:

```bash
export IMAGE_FILES_PATH="absolute-path-stanford40-jpeg-images-dir"
export XML_FILES_PATH="absolute-path-stanford40-xml-annotations-dir"
```

## Usage

In order to run the Neural Network example you can simply run:

```bash
uv run main.py
```

### Available Configurations

The script can be configured through the usage of environment variables. An example usage with custom configuration is:

```bash
# Increased number of epochs
NN_NUM_EPOCHS=500 uv run main.py

# Increased number of epochs and only pretrained model selected
NN_NUM_EPOCHS=500 MODEL="pretrained" uv run main.py
```

The env vars used are:

#### Data Ratio

| Name               | Description                                      | Type    | Default |
| ------------------ | ------------------------------------------------ | ------- | ------- |
| `VALIDATION_RATIO` | The percentage of train data used for validation | `float` | 0.05    |
| `TEST_RATIO`       | The percentage of full data used for test        | `float` | 0.15    |

#### Directory Paths

| Name               | Description                         | Type     | Default          |
| ------------------ | ----------------------------------- | -------- | ---------------- |
| `IMAGE_FILES_PATH` | The path to the jpeg images dir     | `string` | "JPEGImages"     |
| `XML_FILES_PATH`   | The path to the xml annotations dir | `string` | "XMLAnnotations" |

#### Neural Network HyperParameters

| Name                  | Description                          | Type    | Default |
| --------------------- | ------------------------------------ | ------- | ------- |
| `NN_IMAGE_READ_MODE`  | Mode of image read (GRAY or RGB)     | `str`   | "RGB"   |
| `NN_LEARNING_RATE`    | The percentage of learning rate      | `float` | 1e-4    |
| `NN_TRANSFORM_RESIZE` | The size of the image transformation | `int`   | 224     |
| `NN_TRAIN_BATCH_SIZE` | The batch size used for training     | `int`   | 128     |
| `NN_TEST_BATCH_SIZE`  | The batch size used for testing      | `int`   | 50      |
| `NN_VAL_BATCH_SIZE`   | The batch size used for validation   | `int`   | 15      |
| `NN_NUM_EPOCHS`       | The number of epochs during training | `int`   | 25      |

#### Model and Plot

| Name        | Description                                                                                                                                         | Type   | Default |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------- |
| `MODEL`     | Specify which model you want to use ["pretrained" or "custom"]. If missing the script will iterate over both models (first custom, then pretrained) | `str`  | "both"  |
| `SAVE_PLOT` | Skips showing of the plot and saves it in a file instead (only supported for cpu case)                                                              | `bool` | False   |

## Test Resources

The test resources used are images fetched directly from the Stanford40 public dataset.
