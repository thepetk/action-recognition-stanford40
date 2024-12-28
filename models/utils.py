from models.custom_nn import CustomActionRecogntionNN
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch import optim


def train(
    model: "CustomActionRecogntionNN",
    train_loader: "DataLoader",
    device: "torch.device",
    criterion: "nn.CrossEntropyLoss",
    optimizer: "optim.Adam",
) -> "tuple[CustomActionRecogntionNN, float]":
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
    model: "CustomActionRecogntionNN",
    loader: "DataLoader",
    device: "torch.device",
    criterion: "nn.CrossEntropyLoss",
) -> "float":
    # Test stage
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for test_inputs, test_targets in loader:
            test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
            outputs = model(test_inputs)
            test_loss += criterion(outputs, test_targets)
            _, predicted = torch.max(outputs.data, 1)
            total += test_targets.size(0)
            correct += (predicted == test_targets).sum().item()

    accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(loader)
    return avg_test_loss, accuracy


def validate(
    model: "CustomActionRecogntionNN",
    loader: "DataLoader",
    device: "torch.device",
    criterion: "nn.CrossEntropyLoss",
) -> "tuple[CustomActionRecogntionNN, float]":
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for val_inputs, val_targets in loader:
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_targets)

        avg_val_loss = val_loss / len(loader)
        return model, avg_val_loss
