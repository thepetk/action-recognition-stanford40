from models.custom_nn import CustomActionRecogntionNN
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch import optim

from models import PretrainedNN


def train(
    model: "CustomActionRecogntionNN | PretrainedNN",
    train_loader: "DataLoader",
    device: "torch.device",
    criterion: "nn.CrossEntropyLoss",
    optimizer: "optim.Adam",
) -> "tuple[CustomActionRecogntionNN | PretrainedNN, float]":
    model.train()
    total_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # Forward pass
        outputs = model(inputs)
        # Compute loss
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_train_loss = total_loss / len(train_loader)
    return model, avg_train_loss


def test(
    model: "CustomActionRecogntionNN | PretrainedNN",
    loader: "DataLoader",
    device: "torch.device",
    criterion: "nn.CrossEntropyLoss",
) -> "tuple[float, float]":
    # Test stage
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for test_inputs, test_targets in loader:
            test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
            outputs = model(test_inputs)
            test_loss = criterion(outputs, test_targets)
            total_loss += test_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += test_targets.size(0)
            correct += (predicted == test_targets).sum().item()

    accuracy = 100 * correct / total
    avg_test_loss = total_loss / len(loader)
    return avg_test_loss, accuracy


def validate(
    model: "CustomActionRecogntionNN | PretrainedNN",
    loader: "DataLoader",
    device: "torch.device",
    criterion: "nn.CrossEntropyLoss",
) -> "tuple[CustomActionRecogntionNN | PretrainedNN, float]":
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for val_inputs, val_targets in loader:
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)
            total_loss += val_loss

        avg_val_loss = total_loss / len(loader)
        return model, avg_val_loss
