import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from sfd40.utils import Stanford40DataItem, get_action


class Stanford40Dataset(Dataset):
    """
    wraps the torch.utils.data and adds a few attributes
    like labels, transform and read_mode in order to align
    the dataset with the Stanford40 classification issue.
    """

    def __init__(
        self,
        image_items: "list[Stanford40DataItem]",
        read_mode: "ImageReadMode",
        labels: "dict[str, int]",
        transform: "transforms.Compose",
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

    def __getitem__(self, idx) -> "tuple[torch.Tensor, int]":
        image = read_image(self.image_items[idx].image, mode=self.read_mode)
        action = get_action(self.image_items[idx].xml)

        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, self.labels[action]  # type: ignore
