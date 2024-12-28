import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from sfd40 import Stanford40DataSplitter, load_data, get_hyperparameters
from models import PretrainedNN, CustomActionRecogntionNN, validate, train, test
import os

MODEL = os.getenv("MODEL", "")


class ModelChoice:
    RESNET = "pretrained"
    CUSTOM = "custom"


def get_chosen_models(model: "str" = MODEL) -> "list[str]":
    if model == ModelChoice.CUSTOM:
        print("Model:: Only custom model is included")
        return [ModelChoice.CUSTOM]
    elif model == ModelChoice.RESNET:
        print("Model:: Only resnet model is included")
        return [ModelChoice.RESNET]
    else:
        print("Model:: Both models are included")
        return [ModelChoice.CUSTOM, ModelChoice.RESNET]


CHOSEN_MODELS = get_chosen_models()


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
            transforms.Grayscale(),
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

    for idx, model_name in enumerate(CHOSEN_MODELS):
        print(
            "\n",
            "-" * 8,
            f"[{idx+1}/{len(CHOSEN_MODELS)}] Initializing {model_name} NN",
            "-" * 8,
        )
        if model_name == ModelChoice.CUSTOM:
            model = CustomActionRecogntionNN(
                hparams.in_channels, stanford_loader.num_classes, hparams.resize
            ).to(device)
        else:
            model = PretrainedNN(hparams.in_channels, stanford_loader.num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
        print(f"Model:: Initialized {model_name} NN, optimizer and criterion")

        print(
            "\n",
            "-" * 8,
            f"[{idx+1}/{len(CHOSEN_MODELS)}] Model Training and Validation",
            "-" * 8,
        )
        for epoch in range(hparams.num_epochs):
            model, loss = train(
                model, stanford_loader.train, device, criterion, optimizer
            )
            print(f"Epoch {epoch + 1}/{hparams.num_epochs}, Training Loss: {loss:.4f}")

            model, avg_val_loss = validate(
                model, stanford_loader.validation, device, criterion
            )
            print(
                f"Epoch {epoch + 1}/{hparams.num_epochs}, Validation Loss: {avg_val_loss:.4f}"
            )

        avg_test_loss, accuracy = test(model, stanford_loader.test, device, criterion)
        print(f"Test Loss: {avg_test_loss:.4f} | Accuracy: {accuracy:.2f}%")

        torch.save(model.state_dict(), f"segmentation_{model_name}_.pth")


if __name__ == "__main__":
    main()
