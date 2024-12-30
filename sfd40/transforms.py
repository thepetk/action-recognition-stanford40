from typing import Any
from torchvision import transforms


class Stanford40Transform:
    def __init__(self, resize: "int") -> "None":
        self.resize = resize

    @property
    def _grayscale(self) -> "list[Any]":
        return [transforms.Grayscale()]

    @property
    def _augmentations(self) -> "list[Any]":
        return [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ]

    @property
    def _base(self) -> "list[Any]":
        return [
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]

    @property
    def train(self) -> "transforms.Compose":
        return transforms.Compose(self._grayscale + self._augmentations + self._base)

    @property
    def test_val(self) -> "transforms.Compose":
        return transforms.Compose(self._grayscale + self._base)
