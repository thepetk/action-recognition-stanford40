import matplotlib.pyplot as plt

PLOT_FIGURE_SIZE = (12, 6)


# method to plot the evolution of the neural network
def plot(
    accuracy: "float",
    train_losses: "list[float]",
    val_losses: "list[float]",
    figsize: "tuple[int, int]" = PLOT_FIGURE_SIZE,
    save_not_show: "bool" = False,
) -> None:
    _num_of_epochs = range(1, len(train_losses) + 1)
    plt.figure(1, 1, figsize=figsize)

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
