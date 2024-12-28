import os
from torchvision.io import ImageReadMode

# File paths
ROOT_PATH = os.getenv("ROOT_PATH", "Stanford40")
IMAGE_FILES_PATH = f"{ROOT_PATH}/{os.getenv('IMAGE_FILES_PATH', 'JPEGImages')}"
XML_FILES_PATH = f"{ROOT_PATH}/{os.getenv('XML_FILES_PATH', 'XMLAnnotations')}"

# NN config
RAW_IMAGE_READ_MODE = os.getenv("NN_IMAGE_READ_MODE", "GRAY")
NN_LEARNING_RATE = float(os.getenv("NN_LEARNING_RATE", 0.001))
NN_TRANSFORM_RESIZE = int(os.getenv("NN_TRANSFORM_RESIZE", 64))
NN_TRAIN_BATCH_SIZE = int(os.getenv("NN_TRAIN_BATCH_SIZE", 256))
NN_TEST_BATCH_SIZE = int(os.getenv("NN_TEST_BATCH_SIZE", 50))
NN_NUM_EPOCHS = int(os.getenv("NN_NUM_EPOCHS", 5))

# Data Ratio
TEST_RATIO = float(os.getenv("TEST_RATIO", 0.20))
VALIDATION_RATIO = float(os.getenv("VALIDATION_RATIO", 0.15))


def get_image_read_mode(
    img_read_mode: "str" = RAW_IMAGE_READ_MODE,
) -> "ImageReadMode":
    if img_read_mode == "RGB":
        return ImageReadMode.RGB
    else:
        return ImageReadMode.GRAY


NN_IMAGE_READ_MODE = get_image_read_mode()


def get_nn_in_channels(
    img_read_mode: "ImageReadMode" = NN_IMAGE_READ_MODE,
) -> "int":
    if img_read_mode == ImageReadMode.RGB:
        return 3
    else:
        return 1


NN_IN_CHANNELS = get_nn_in_channels()
