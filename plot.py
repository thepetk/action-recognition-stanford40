import matplotlib.pyplot as plt
from sfd40 import Stanford40HyperParameters
import yaml


def plot(
    accuracy: "float",
    train_losses: "list[float]",
    val_losses: "list[float]",
    save_not_show: "bool" = False,
) -> None:
    """
    plot is responsible for all the plots generated
    after the execution of the models. In this version
    it only includes the evolution of training and
    validation errors.
    """
    _num_of_epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 8))

    plt.plot(_num_of_epochs, train_losses, label="Training Error", marker="o")
    plt.plot(_num_of_epochs, val_losses, label="Validation Error", marker="x")

    # Add labels, title, legend, and grid
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title(f"Loss Evolution - Accuracy: {accuracy:.2f} %")
    plt.legend()
    if save_not_show:
        plt.savefig("fig.png")
    else:
        plt.grid(True)
        plt.show()


def save_as_yaml(
    accuracy: "float",
    model_name: "str",
    training_losses: "list[float]",
    validation_losses: "list[float]",
    hparams: "Stanford40HyperParameters",
):
    data = {
        "eid": 0,
        "nn": model_name,
        "hyperparameters": {
            "in_channels": hparams.in_channels,
            "learning_rate": hparams.learning_rate,
            "resize": hparams.resize,
            "train_batch_size": hparams.train_batch_size,
            "test_batch_size": hparams.test_batch_size,
            "val_batch_size": hparams.val_batch_size,
            "num_epochs": hparams.num_epochs,
        },
        "results": {
            "accuracy": accuracy,
            "training_losses": training_losses,
            "validation_losses": validation_losses,
        },
    }
    with open("_example.yml", "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
