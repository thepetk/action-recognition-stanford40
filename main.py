import os
import torch

from models import ModelManager
from sfd40 import Stanford40DataManager

from plot import plot, save_as_yaml

_TEST_MODE = os.getenv("TEST_MODE", 0)


def main() -> "None":
    print("\n", "-" * 8, "Torch Device Selection", "-" * 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:: Selected device: {device.type}")

    print("\n", "-" * 8, "Setup Data Loaders", "-" * 8)
    data_manager = Stanford40DataManager()
    print("DataManager:: Set up 3 DataLoaders: Train, Test, Validation")

    print("\n", "-" * 8, "HyperParameters Config", "-" * 8)
    hparams = data_manager.hparams
    data_manager.print_hparams()

    print("\n", "-" * 8, "Setup Models", "-" * 8)
    model_manager = ModelManager(device=device)
    chosen_models = model_manager.chosen_models
    for idx, model_name in enumerate(chosen_models):
        print(
            "\n",
            "-" * 8,
            f"[{idx+1}/{len(chosen_models)}] Initializing {model_name} NN",
            "-" * 8,
        )
        model_manager.init_model(
            model_name,
            hparams.in_channels,
            data_manager.num_classes,
            hparams.learning_rate,
        )

        print(
            "\n",
            "-" * 8,
            f"[{idx+1}/{len(chosen_models)}] Model Training and Validation",
            "-" * 8,
        )

        train_losses: "list[float]" = []
        val_losses: "list[float]" = []
        for epoch in range(hparams.num_epochs):
            # Training Stage
            avg_train_loss = model_manager.train(data_manager.train_loader)
            train_losses.append(avg_train_loss)

            # Validation Stage
            avg_val_loss = model_manager.validate(data_manager.validation_loader)
            val_losses.append(avg_val_loss)

            print(
                f"[Epoch {epoch + 1}/{hparams.num_epochs}]",
                "\t",
                "Train Loss: {avg_train_loss:.4f}",
                "\t|\t",
                "Validation Loss: {avg_val_loss:.4f}",
            )

        # Testing Stage
        accuracy = model_manager.test(data_manager.test_loader)
        print(f"Accuracy: {accuracy:.2f}%")

        if device.type == "cpu" and not _TEST_MODE:
            plot(accuracy, train_losses, val_losses)

        if model_manager.save_as_yaml and not _TEST_MODE:
            save_as_yaml(accuracy, model_name, train_losses, val_losses, hparams)


if __name__ == "__main__":
    main()
