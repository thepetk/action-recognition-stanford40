import random
import os
from sfd40.utils import Stanford40DataItem, DATA_ITEMS, get_action
from sfd40.defaults import (
    IMAGE_FILES_PATH,
    XML_FILES_PATH,
    TEST_RATIO,
    VALIDATION_RATIO,
)
from sfd40.errors import DataSeparationError


class Stanford40DataSplitter:
    def __init__(
        self,
        image_files_path: "str" = IMAGE_FILES_PATH,
        xml_files_path: "str" = XML_FILES_PATH,
        test_ratio: "float" = TEST_RATIO,
        validation_ratio: "float" = VALIDATION_RATIO,
    ) -> "None":
        self.image_files = self._get_image_files(image_files_path)
        self.xml_files = self._get_xml_files(xml_files_path)
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        print(
            "Splitter:: separating for given ratios: test {test}, val {val}".format(
                test=self.test_ratio, val=self.validation_ratio
            )
        )

    def _get_image_files(self, image_files_path: "str") -> "list[str]":
        """
        generates shuffled list of image files inside a given path
        """
        _files = [
            os.path.join(image_files_path, f)
            for f in os.listdir(image_files_path)
            if os.path.isfile(os.path.join(image_files_path, f))
        ]
        random.shuffle(_files)
        return _files

    def _get_xml_files(self, xml_files_path: "str") -> "list[str]":
        return [
            os.path.join(xml_files_path, f)
            for f in os.listdir(xml_files_path)
            if os.path.isfile(os.path.join(xml_files_path, f))
        ]

    @property
    def full_size(self) -> "int":
        return len(self.image_files)

    @property
    def test_size(self) -> "int":
        return int(self.full_size * self.test_ratio)

    def _get_xml_file(self, name: "str") -> "str | None":
        xml_name = name.split("/")[-1].split(".")[0]
        for xml_path in self.xml_files:
            if xml_name in xml_path:
                return xml_path
        return None

    @property
    def _all_items(self) -> "DATA_ITEMS":
        return [
            Stanford40DataItem(image=img, xml=self._get_xml_file(img))
            for img in self.image_files
        ]

    @property
    def test_items(self) -> "DATA_ITEMS":
        return [
            Stanford40DataItem(image=img, xml=self._get_xml_file(img))
            for img in self.image_files[: self.test_size]
        ]

    @property
    def validation_size(self) -> "int":
        return int(self.full_size * self.validation_ratio)

    @property
    def validation_items(self) -> "DATA_ITEMS":
        return [
            Stanford40DataItem(image=img, xml=self._get_xml_file(img))
            for img in self.image_files[(self.test_size + self.train_size) :]
        ]

    @property
    def train_size(self) -> "int":
        return self.full_size - self.test_size - self.validation_size

    @property
    def train_items(self) -> "DATA_ITEMS":
        return [
            Stanford40DataItem(image=img, xml=self._get_xml_file(img))
            for img in self.image_files[
                self.test_size : (self.test_size + self.train_size)
            ]
        ]

    @property
    def data_separation_is_valid(self) -> "bool":
        return len(self.image_files) == len(self.train_items) + len(
            self.validation_items
        ) + len(self.test_items)

    def generate_labels(self) -> "dict[str, int]":
        labels: "dict[str, int]" = {}
        idx = 0
        for img_item in self._all_items:
            action = get_action(img_item.xml)
            if labels.get(action) is not None:
                continue
            labels[action] = idx
            idx += 1
        return labels

    def seperate(self) -> "tuple[DATA_ITEMS, DATA_ITEMS, DATA_ITEMS]":
        if not self.data_separation_is_valid:
            raise DataSeparationError("Sets not equal to full size of images")
        print("Splitter:: separated dataset into 3 datasets")
        print(f"Splitter:: train: {len(self.train_items)} items")
        print(f"Splitter:: test: {len(self.test_items)} items")
        print(f"Splitter:: validation: {len(self.validation_items)} items")
        return (self.train_items, self.test_items, self.validation_items)
