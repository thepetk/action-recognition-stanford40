from torchvision import transforms
from torch.utils.data import DataLoader
from sfd40.utils import DATA_ITEMS, Stanford40HyperParameters, Stanford40DataLoader
from sfd40.dataset import Stanford40Dataset


def load_data(
    train_items: "DATA_ITEMS",
    test_items: "DATA_ITEMS",
    validation_items: "DATA_ITEMS",
    transform: "transforms.Compose",
    hparams: "Stanford40HyperParameters",
) -> "Stanford40DataLoader":
    train_dataset = Stanford40Dataset(
        train_items, transform=transform, read_mode=hparams.image_read_mode
    )
    test_dataset = Stanford40Dataset(
        test_items, transform=transform, read_mode=hparams.image_read_mode
    )
    val_dataset = Stanford40Dataset(
        validation_items, transform=transform, read_mode=hparams.image_read_mode
    )
    print("Loader:: Loaded 3 datasets")
    return Stanford40DataLoader(
        num_classes=len(train_dataset.labels.keys()) + 1,
        train=DataLoader(
            train_dataset, batch_size=hparams.train_batch_size, shuffle=True
        ),
        test=DataLoader(
            test_dataset, batch_size=hparams.test_batch_size, shuffle=False
        ),
        validation=DataLoader(
            val_dataset, batch_size=hparams.validation_batch_size, shuffle=False
        ),
    )
