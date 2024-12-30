from typing import Any
from torchvision import transforms


class Stanford40Transform:
    def __init__(self, resize: "int") -> "None":
        self.resize = resize

    @property
    def train(self) -> "transforms.Compose":
        return transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.RandomResizedCrop(self.resize),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    @property
    def test_val(self) -> "transforms.Compose":
        return transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(self.resize + 2),
                transforms.CenterCrop(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
