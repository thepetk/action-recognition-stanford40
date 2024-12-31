import os
from torchvision.io import ImageReadMode

# File paths
IMAGE_FILES_PATH = os.getenv("IMAGE_FILES_PATH", "Stanford40/JPEGImages")
XML_FILES_PATH = os.getenv("XML_FILES_PATH", "Stanford40/XMLAnnotations")

# NN config
RAW_IMAGE_READ_MODE = os.getenv("NN_IMAGE_READ_MODE", "RGB")
NN_IN_CHANNELS = 3
NN_LEARNING_RATE = float(os.getenv("NN_LEARNING_RATE", 1e-4))
NN_TRANSFORM_RESIZE = int(os.getenv("NN_TRANSFORM_RESIZE", 224))
NN_TRAIN_BATCH_SIZE = int(os.getenv("NN_TRAIN_BATCH_SIZE", 128))
NN_TEST_BATCH_SIZE = int(os.getenv("NN_TEST_BATCH_SIZE", 50))
NN_VAL_BATCH_SIZE = int(os.getenv("NN_VAL_BATCH_SIZE", 15))
NN_NUM_EPOCHS = int(os.getenv("NN_NUM_EPOCHS", 25))

# Data Ratio
TEST_RATIO = float(os.getenv("TEST_RATIO", 0.20))
VALIDATION_RATIO = float(os.getenv("VALIDATION_RATIO", 0.05))


def get_image_read_mode(
    img_read_mode: "str" = RAW_IMAGE_READ_MODE,
) -> "ImageReadMode":
    if img_read_mode == "RGB":
        return ImageReadMode.RGB
    else:
        return ImageReadMode.GRAY


NN_IMAGE_READ_MODE = get_image_read_mode()
