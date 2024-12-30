from torch.utils.data import DataLoader
from sfd40.utils import (
    Stanford40HyperParameters,
    Stanford40DataLoader,
    Stanford40DataItemCollection,
)
from sfd40.transforms import Stanford40Transform
from sfd40.dataset import Stanford40Dataset


def load_data(
    labels: "dict[str, int]",
    items_collection: "Stanford40DataItemCollection",
    transform: "Stanford40Transform",
    hparams: "Stanford40HyperParameters",
) -> "Stanford40DataLoader":
    """
    returns a Stanford40 DataLoader object containing
    3 loaders (train, test and validation)
    """
    train_dataset = Stanford40Dataset(
        items_collection.train,
        labels=labels,
        transform=transform.train,
        read_mode=hparams.image_read_mode,
    )
    test_dataset = Stanford40Dataset(
        items_collection.test,
        labels=labels,
        transform=transform.test_val,
        read_mode=hparams.image_read_mode,
    )
    val_dataset = Stanford40Dataset(
        items_collection.validation,
        labels=labels,
        transform=transform.test_val,
        read_mode=hparams.image_read_mode,
    )
    print("Loader:: Loaded 3 datasets")
    return Stanford40DataLoader(
        num_classes=len(train_dataset.labels.keys()),
        train=DataLoader(
            train_dataset, batch_size=hparams.train_batch_size, shuffle=True
        ),
        test=DataLoader(test_dataset, batch_size=hparams.test_batch_size, shuffle=True),
        validation=DataLoader(
            val_dataset, batch_size=hparams.val_batch_size, shuffle=True
        ),
    )
