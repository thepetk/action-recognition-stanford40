from dataclasses import dataclass
from xml.etree import ElementTree
from torch.utils.data import DataLoader
from torchvision.io import ImageReadMode
from sfd40.defaults import (
    NN_LEARNING_RATE,
    NN_TRANSFORM_RESIZE,
    NN_TRAIN_BATCH_SIZE,
    NN_TEST_BATCH_SIZE,
    NN_NUM_EPOCHS,
    NN_IMAGE_READ_MODE,
    NN_IN_CHANNELS,
)


@dataclass
class Stanford40DataItem:
    image: "str"
    xml: "str"


DATA_ITEMS = list[Stanford40DataItem]


@dataclass
class Stanford40DataLoader:
    num_classes: "int"
    train: "DataLoader"
    test: "DataLoader"
    validation: "DataLoader"


@dataclass
class Stanford40HyperParameters:
    in_channels: "int"
    learning_rate: "float"
    resize: "int"
    train_batch_size: "int"
    test_batch_size: "int"
    num_epochs: "int"
    image_read_mode: "ImageReadMode"


def get_hyperparameters() -> "Stanford40HyperParameters":
    print("Config:: initializing hyperparameters")
    print(f"Config:: in_channels: {NN_IN_CHANNELS}")
    print(f"Config:: learning_rate: {NN_LEARNING_RATE}")
    print(f"Config:: resize: {NN_TRANSFORM_RESIZE}")
    print(f"Config:: train_batch_size: {NN_TRAIN_BATCH_SIZE}")
    print(f"Config:: test_batch_size: {NN_TEST_BATCH_SIZE}")
    print(f"Config:: num_epochs: {NN_NUM_EPOCHS}")
    return Stanford40HyperParameters(
        in_channels=NN_IN_CHANNELS,
        learning_rate=NN_LEARNING_RATE,
        resize=NN_TRANSFORM_RESIZE,
        train_batch_size=NN_TRAIN_BATCH_SIZE,
        test_batch_size=NN_TEST_BATCH_SIZE,
        num_epochs=NN_NUM_EPOCHS,
        image_read_mode=NN_IMAGE_READ_MODE,
    )


def get_action(xml_file: "str") -> "str":
    root = ElementTree.parse(xml_file).getroot()
    return root.find("object/action").text
