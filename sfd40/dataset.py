import torch
from xml.etree import ElementTree
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from sfd40.utils import DATA_ITEMS


class Stanford40Dataset(Dataset):
    def __init__(
        self,
        image_items: "DATA_ITEMS",
        read_mode: "ImageReadMode",
        transform: "transforms.Compose | None" = None,
    ) -> "None":
        self.image_items = image_items
        self.transform = transform
        self.read_mode = read_mode
        self.labels = self._generate_labels()
        print(
            "Dataset:: Generating dataset for {n} given items and {l} labels".format(
                n=len(self.image_items), l=len(self.labels.keys())
            )
        )

    def _get_action(self, xml_file: "str") -> "str":
        root = ElementTree.parse(xml_file).getroot()
        return root.find("object/action").text

    def _generate_labels(self) -> "dict[str, int]":
        labels: "dict[str, int]" = {}
        idx = 0
        for img_item in self.image_items:
            action = self._get_action(img_item.xml)
            if labels.get(action):
                continue
            labels[action] = idx
            idx += 1
        return labels

    def __len__(self) -> "int":
        return len(self.image_items)

    def __getitem__(self, idx) -> "tuple[torch.Tensor, str]":
        image = read_image(self.image_items[idx].image, mode=self.read_mode)
        action = self._get_action(self.image_items[idx].xml)

        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, self.labels[action]
