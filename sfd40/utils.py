from dataclasses import dataclass
from xml.etree import ElementTree
from torchvision.io import ImageReadMode


@dataclass
class Stanford40DataItem:
    image: "str"
    xml: "str"


@dataclass
class Stanford40DataItemCollection:
    train: "list[Stanford40DataItem]"
    validation: "list[Stanford40DataItem]"
    test: "list[Stanford40DataItem]"


@dataclass
class Stanford40HyperParameters:
    in_channels: "int"
    learning_rate: "float"
    resize: "int"
    train_batch_size: "int"
    test_batch_size: "int"
    val_batch_size: "int"
    num_epochs: "int"
    image_read_mode: "ImageReadMode"


def get_action(xml_file: "str") -> "str":
    root = ElementTree.parse(xml_file).getroot()
    return root.find("object/action").text  # type: ignore
