import torch
import torch.nn as nn
import torch.optim as optim

from sfd40 import (
    Stanford40DataSplitter,
    Stanford40Transform,
    load_data,
    get_hyperparameters,
)
from models import PretrainedNN, CustomActionRecogntionNN, validate, train, test
import os
from plot import plot, save_as_yaml

MODEL = os.getenv("MODEL", "")
SAVE_AS_YAML = bool(os.getenv("SAVE_AS_YAML", True))


class ModelChoice:
    """
    Covers the model decision between pretrained or custom NN
    """

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


def initialize_model(
    model_name: "str",
    in_channels: "int",
    num_classes: "int",
    device: "torch.device",
) -> "CustomActionRecogntionNN | PretrainedNN":
    if model_name == ModelChoice.CUSTOM:
        return CustomActionRecogntionNN(in_channels, num_classes).to(device)
    else:
        return PretrainedNN(in_channels, num_classes).to(device)


def main() -> "None":
    print("\n", "-" * 8, "Torch Device Selection", "-" * 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:: Selected device: {device.type}")

    print("\n", "-" * 8, "HyperParameters Config", "-" * 8)
    hparams = get_hyperparameters()

    print("\n", "-" * 8, "Data Split and Load", "-" * 8)
    splitter = Stanford40DataSplitter()
    transform = Stanford40Transform(hparams.resize)
    labels = splitter.generate_labels()
    items_collection = splitter.seperate()
    stanford_loader = load_data(labels, items_collection, transform, hparams)

    for idx, model_name in enumerate(CHOSEN_MODELS):
        print(
            "\n",
            "-" * 8,
            f"[{idx+1}/{len(CHOSEN_MODELS)}] Initializing {model_name} NN",
            "-" * 8,
        )

        model = initialize_model(
            model_name,
            hparams.in_channels,
            stanford_loader.num_classes,
            device,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
        print(f"Model:: Initialized {model_name} NN, optimizer and criterion")

        print(
            "\n",
            "-" * 8,
            f"[{idx+1}/{len(CHOSEN_MODELS)}] Model Training and Validation",
            "-" * 8,
        )
        train_losses: "list[float]" = []
        val_losses: "list[float]" = []
        for epoch in range(hparams.num_epochs):
            # Training Stage
            model, avg_train_loss = train(
                model, stanford_loader.train, device, criterion, optimizer
            )
            train_losses.append(avg_train_loss)

            # Validation Stage
            model, avg_val_loss = validate(
                model, stanford_loader.validation, device, criterion
            )
            val_losses.append(avg_val_loss)
            print(
                f"[Epoch {epoch + 1}/{hparams.num_epochs}]\tTrain Loss: {avg_train_loss:.4f}\t|\tValidation Loss: {avg_val_loss:.4f}"
            )

        # Testing Stage
        avg_test_loss, accuracy = test(model, stanford_loader.test, device, criterion)
        print(f"Test Loss: {avg_test_loss:.4f} | Accuracy: {accuracy:.2f}%")

        if device.type == "cpu":
            plot(accuracy, train_losses, val_losses)

        if SAVE_AS_YAML:
            save_as_yaml(accuracy, model_name, train_losses, val_losses, hparams)


if __name__ == "__main__":
    main()
