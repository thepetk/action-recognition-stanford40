import matplotlib.pyplot as plt


def plot(
    accuracy: "float",
    train_losses: "list[float]",
    val_losses: "list[float]",
    save_not_show: "bool" = False,
) -> None:
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
