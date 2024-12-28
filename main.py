import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from sfd40 import Stanford40DataSplitter, load_data, get_hyperparameters
from models import PretrainedNN, CustomActionRecogntionNN, validate, train, test


def main() -> "None":
    print("\n", "-" * 8, "Torch Device Selection", "-" * 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:: Selected device: {device.type}")

    print("\n", "-" * 8, "HyperParameters Config", "-" * 8)
    hparams = get_hyperparameters()

    print("\n", "-" * 8, "Data Split and Load", "-" * 8)
    splitter = Stanford40DataSplitter()
    transform = transforms.Compose(
        [
            transforms.Resize((hparams.resize, hparams.resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    labels = splitter.generate_labels()
    train_items, test_items, validation_items = splitter.seperate()
    stanford_loader = load_data(
        labels, train_items, test_items, validation_items, transform, hparams
    )

    print("\n", "-" * 8, "Initializing NN", "-" * 8)
    model = CustomActionRecogntionNN(
        hparams.in_channels, stanford_loader.num_classes, hparams.resize
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
    print("Model:: Initialized NN, optimizer and criterion")

    print("\n", "-" * 8, "Model Training and Validation", "-" * 8)
    for epoch in range(hparams.num_epochs):
        model, loss = train(model, stanford_loader.train, device, criterion, optimizer)
        print(f"Epoch {epoch + 1}/{hparams.num_epochs}, Training Loss: {loss:.4f}")

        model, avg_val_loss = validate(
            model, stanford_loader.validation, device, criterion
        )
        print(
            f"Epoch {epoch + 1}/{hparams.num_epochs}, Validation Loss: {avg_val_loss:.4f}"
        )

    avg_test_loss = test(model, stanford_loader.test, device, criterion)
    print(f"Test Loss: {avg_test_loss:.4f}")

    torch.save(model.state_dict(), "segmentation_model.pth")


if __name__ == "__main__":
    main()
