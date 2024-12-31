from torchvision import transforms


class Stanford40Transforms:
    """
    provides two ways of transformation:
    1. Train (with augmentations)
    2. Test or Validation (without augmentations)
    """

    def __init__(self, resize: "int") -> "None":
        self.resize = resize

    @property
    def train(self) -> "transforms.Compose":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(self.resize),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    @property
    def test_val(self) -> "transforms.Compose":
        return transforms.Compose(
            [
                transforms.Resize(self.resize + 2),
                transforms.CenterCrop(self.resize),
                transforms.RandomGrayscale(0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
