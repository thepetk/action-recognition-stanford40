import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from sfd40.utils import DATA_ITEMS, get_action


class Stanford40Dataset(Dataset):
    def __init__(
        self,
        image_items: "DATA_ITEMS",
        read_mode: "ImageReadMode",
        labels: "dict[str, int]",
        transform: "transforms.Compose | None" = None,
    ) -> "None":
        self.image_items = image_items
        self.transform = transform
        self.read_mode = read_mode
        self.labels = labels
        print(
            "Dataset:: Generating dataset for {n} given items and {l} labels".format(
                n=len(self.image_items), l=len(self.labels.keys())
            )
        )

    def __len__(self) -> "int":
        return len(self.image_items)

    def _encode_label(self, label: "int") -> "torch.Tensor":
        _t = torch.zeros(len(self.labels.keys()))
        _t[label] = 1.0
        return _t

    def __getitem__(self, idx) -> "tuple[torch.Tensor, torch.Tensor]":
        image = read_image(self.image_items[idx].image, mode=self.read_mode)
        action = get_action(self.image_items[idx].xml)

        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, self._encode_label(self.labels[action])
