from dataclasses import dataclass
import random
import torch
import torch.nn as nn
import torch.optim as optim
from xml.etree import ElementTree
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
import os


# File paths
ROOT_PATH = os.getenv("ROOT_PATH", "Stanford40")
IMAGE_FILES_PATH = f"{ROOT_PATH}/{os.getenv('IMAGE_FILES_PATH', 'JPEGImages')}"
XML_FILES_PATH = f"{ROOT_PATH}/{os.getenv('XML_FILES_PATH', 'XMLAnnotations')}"

# NN config
RAW_IMAGE_READ_MODE = os.getenv("NN_IMAGE_READ_MODE", "GRAY")
NN_LEARNING_RATE = float(os.getenv("NN_LEARNING_RATE", 0.001))
NN_TRANSFORM_RESIZE = int(os.getenv("NN_TRANSFORM_RESIZE", 256))
NN_TRAIN_BATCH_SIZE = int(os.getenv("NN_TRAIN_BATCH_SIZE", 100))
NN_TEST_BATCH_SIZE = int(os.getenv("NN_TEST_BATCH_SIZE", 50))
NN_VALIDATION_BATCH_SIZE = int(os.getenv("NN_VALIDATION_BATCH_SIZE", 50))
NN_NUM_EPOCHS = int(os.getenv("NN_NUM_EPOCHS", 5))

# Data Ratio
TEST_RATIO = float(os.getenv("TEST_RATIO", 0.20))
VALIDATION_RATIO = float(os.getenv("VALIDATION_RATIO", 0.15))


class DataSeparationError(Exception):
    pass


def get_image_read_mode(
    img_read_mode: "str" = RAW_IMAGE_READ_MODE,
) -> "ImageReadMode":
    if img_read_mode == "RGB":
        return ImageReadMode.RGB
    else:
        return ImageReadMode.GRAY


NN_IMAGE_READ_MODE = get_image_read_mode()


def get_nn_in_channels(
    img_read_mode: "ImageReadMode" = NN_IMAGE_READ_MODE,
) -> "int":
    if img_read_mode == ImageReadMode.RGB:
        return 3
    else:
        return 1


NN_IN_CHANNELS = get_nn_in_channels()


@dataclass
class Stanford40DataItem:
    image: "str"
    xml: "str"


DATA_ITEMS = list[Stanford40DataItem]


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

    def seperate(self) -> "tuple[DATA_ITEMS, DATA_ITEMS, DATA_ITEMS]":
        if not self.data_separation_is_valid:
            raise DataSeparationError("Sets not equal to full size of images")
        print("Splitter:: separated dataset into 3 datasets")
        print(f"Splitter:: train: {len(self.train_items)} items")
        print(f"Splitter:: test: {len(self.test_items)} items")
        print(f"Splitter:: validation: {len(self.validation_items)} items")
        return (self.train_items, self.test_items, self.validation_items)


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


@dataclass
class Stanford40DataLoader:
    num_classes: "int"
    train: "DataLoader"
    test: "DataLoader"
    validation: "DataLoader"


@dataclass
class Stanford40HyperParameters:
    in_channels: "int"
    learning_rate: "float"
    resize: "int"
    train_batch_size: "int"
    test_batch_size: "int"
    validation_batch_size: "int"
    num_epochs: "int"
    image_read_mode: "ImageReadMode"


class ActionRecogntionNN(nn.Module):
    def __init__(self, in_channels: "int", num_classes: "int") -> "None":
        super(ActionRecogntionNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (256 // 2) * (256 // 2), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x: "nn.Sequential") -> "nn.Sequential":
        x = self.encoder(x)
        x = self.fc(x)
        return x


def get_hyperparameters() -> "Stanford40HyperParameters":
    print("Config:: initializing hyperparameters")
    print(f"Config:: in_channels: {NN_IN_CHANNELS}")
    print(f"Config:: learning_rate: {NN_LEARNING_RATE}")
    print(f"Config:: resize: {NN_TRANSFORM_RESIZE}")
    print(f"Config:: train_batch_size: {NN_TRAIN_BATCH_SIZE}")
    print(f"Config:: test_batch_size: {NN_TEST_BATCH_SIZE}")
    print(f"Config:: validation_batch_size: {NN_VALIDATION_BATCH_SIZE}")
    print(f"Config:: num_epochs: {NN_NUM_EPOCHS}")
    return Stanford40HyperParameters(
        in_channels=NN_IN_CHANNELS,
        learning_rate=NN_LEARNING_RATE,
        resize=NN_TRANSFORM_RESIZE,
        train_batch_size=NN_TRAIN_BATCH_SIZE,
        test_batch_size=NN_TEST_BATCH_SIZE,
        validation_batch_size=NN_VALIDATION_BATCH_SIZE,
        num_epochs=NN_NUM_EPOCHS,
        image_read_mode=NN_IMAGE_READ_MODE,
    )


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


def train(
    model: "ActionRecogntionNN",
    train_loader: "DataLoader",
    device: "torch.device",
    criterion: "nn.CrossEntropyLoss",
    optimizer: "optim.Adam",
) -> "tuple[ActionRecogntionNN, float]":
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # Forward pass
        outputs = model(inputs)
        # Compute loss
        loss = criterion(outputs, targets)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, loss.item()


def test(
    model: "ActionRecogntionNN",
    loader: "DataLoader",
    device: "torch.device",
    criterion: "nn.CrossEntropyLoss",
) -> "float":
    # Test stage
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for test_inputs, test_targets in loader:
            test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
            test_outputs = model(test_inputs)
            test_loss += criterion(test_outputs, test_targets)

        return test_loss / len(loader)


def validate(
    model: "ActionRecogntionNN",
    loader: "DataLoader",
    device: "torch.device",
    criterion: "nn.CrossEntropyLoss",
) -> "tuple[ActionRecogntionNN, float]":
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for val_inputs, val_targets in loader:
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_targets)

        avg_val_loss = val_loss / len(loader)
        return model, avg_val_loss


def main() -> "None":
    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n", "-" * 8, "HyperParameters Config", "-" * 8)
    hparams = get_hyperparameters()

    print("\n", "-" * 8, "Data Split and Load", "-" * 8)
    splitter = Stanford40DataSplitter()

    # Data transformations
    transform = transforms.Compose(
        [transforms.Resize((hparams.resize, hparams.resize)), transforms.ToTensor()]
    )

    train_items, test_items, validation_items = splitter.seperate()

    stanford_loader = load_data(
        train_items, test_items, validation_items, transform, hparams
    )
    print("\n", "-" * 8, "Initializing NN", "-" * 8)
    model = ActionRecogntionNN(hparams.in_channels, stanford_loader.num_classes).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
    print("Model:: Initialized NN, optimizer and criterion")
    print("\n", "-" * 8, "Model Training and Validation", "-" * 8)
    for epoch in range(hparams.num_epochs):
        # training stage
        model, loss = train(model, stanford_loader.train, device, criterion, optimizer)
        print(f"Epoch {epoch + 1}/{hparams.num_epochs}, Training Loss: {loss:.4f}")

        # validation stage
        model, avg_val_loss = validate(
            model, stanford_loader.validation, device, criterion
        )
        print(
            f"Epoch {epoch + 1}/{hparams.num_epochs}, Validation Loss: {avg_val_loss:.4f}"
        )

    # test stage
    avg_test_loss = test(model, stanford_loader.test, device, criterion)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # Save the trained model (Optional)
    torch.save(model.state_dict(), "segmentation_model.pth")


if __name__ == "__main__":
    main()
