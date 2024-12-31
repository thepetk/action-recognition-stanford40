from torch.utils.data import DataLoader
from torchvision import transforms
from sfd40.defaults import (
    IMAGE_FILES_PATH,
    NN_IMAGE_READ_MODE,
    NN_IN_CHANNELS,
    NN_LEARNING_RATE,
    NN_NUM_EPOCHS,
    NN_TEST_BATCH_SIZE,
    NN_TRAIN_BATCH_SIZE,
    NN_TRANSFORM_RESIZE,
    NN_VAL_BATCH_SIZE,
    TEST_RATIO,
    VALIDATION_RATIO,
    XML_FILES_PATH,
)
from sfd40.splitter import Stanford40DataSplitter
from sfd40.utils import (
    Stanford40DataItem,
    Stanford40HyperParameters,
)
from sfd40.transforms import Stanford40Transforms
from sfd40.dataset import Stanford40Dataset
from torchvision.io import ImageReadMode


class Stanford40DataManager:
    """
    the main class of the sfd40 responsible to handle
    all the data loading process, provide the hparams,
    the data loaders and coordinate with the rest of
    the classes inside the package.
    """

    def __init__(
        self,
        # File paths
        image_files_path: "str" = IMAGE_FILES_PATH,
        xml_files_path: "str" = XML_FILES_PATH,
        # Data ratio
        test_ratio: "float" = TEST_RATIO,
        validation_ratio: "float" = VALIDATION_RATIO,
        # NN Parameters
        in_channels: "int" = NN_IN_CHANNELS,
        learning_rate: "float" = NN_LEARNING_RATE,
        num_epochs: "int" = NN_NUM_EPOCHS,
        # Data Load
        resize: "int" = NN_TRANSFORM_RESIZE,
        image_read_mode: "ImageReadMode" = NN_IMAGE_READ_MODE,
        # Batch Sizes
        train_batch_size: "int" = NN_TRAIN_BATCH_SIZE,
        test_batch_size: "int" = NN_TEST_BATCH_SIZE,
        val_batch_size: "int" = NN_VAL_BATCH_SIZE,
    ) -> "None":
        self._splitter = Stanford40DataSplitter(
            image_files_path=image_files_path,
            xml_files_path=xml_files_path,
            test_ratio=test_ratio,
            validation_ratio=validation_ratio,
        )
        self._collection = self._splitter.seperate()
        self._image_read_mode = image_read_mode
        self._transforms = Stanford40Transforms(resize=resize)
        self.hparams = self._get_hparams(
            in_channels,
            learning_rate,
            resize,
            train_batch_size,
            test_batch_size,
            val_batch_size,
            num_epochs,
            image_read_mode,
        )
        self.labels = self._splitter.generate_labels()

    def _get_dataset(
        self, transform: "transforms.Compose", image_items: "list[Stanford40DataItem]"
    ) -> "Stanford40Dataset":
        return Stanford40Dataset(
            image_items=image_items,
            labels=self.labels,
            transform=transform,
            read_mode=self._image_read_mode,
        )

    def _get_loader(
        self,
        transform: "transforms.Compose",
        image_items: "list[Stanford40DataItem]",
        batch_size: "int",
        shuffle: "bool",
    ) -> "DataLoader":
        return DataLoader(
            self._get_dataset(transform, image_items),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def _get_hparams(
        self,
        in_channels: "int",
        learning_rate: "float",
        resize: "int",
        train_batch_size: "int",
        test_batch_size: "int",
        val_batch_size: "int",
        num_epochs: "int",
        image_read_mode: "ImageReadMode",
    ) -> "Stanford40HyperParameters":
        return Stanford40HyperParameters(
            in_channels=in_channels,
            learning_rate=learning_rate,
            resize=resize,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            val_batch_size=val_batch_size,
            num_epochs=num_epochs,
            image_read_mode=image_read_mode,
        )

    def print_hparams(self) -> "None":
        print("Loader:: initializing hyperparameters")
        print(f"Loader:: in_channels: {self.hparams.in_channels}")
        print(f"Loader:: learning_rate: {self.hparams.learning_rate}")
        print(f"Loader:: resize: {self.hparams.resize}")
        print(f"Loader:: train_batch_size: {self.hparams.train_batch_size}")
        print(f"Loader:: test_batch_size: {self.hparams.test_batch_size}")
        print(f"Loader:: val_batch_size: {self.hparams.val_batch_size}")
        print(f"Loader:: num_epochs: {self.hparams.num_epochs}")

    @property
    def num_classes(self) -> "int":
        return len(self.labels.keys())

    @property
    def train_loader(self) -> "DataLoader":
        return self._get_loader(
            self._transforms.train,
            self._collection.train,
            self.hparams.train_batch_size,
            shuffle=True,
        )

    @property
    def test_loader(self) -> "DataLoader":
        return self._get_loader(
            self._transforms.test_val,
            self._collection.test,
            self.hparams.test_batch_size,
            shuffle=True,
        )

    @property
    def validation_loader(self) -> "DataLoader":
        return self._get_loader(
            self._transforms.test_val,
            self._collection.validation,
            self.hparams.val_batch_size,
            shuffle=True,
        )
