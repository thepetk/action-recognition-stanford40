from models.custom_nn import CustomActionRecogntionNN
from models.defaults import CHOSEN_MODELS, SAVE_AS_YAML
from models.errors import ModelNotInitializedError
from models.pretrained_nn import PretrainedNN
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch import optim

from models.utils import ModelChoice, ModelOperation

NN_MODEL = CustomActionRecogntionNN | PretrainedNN


class ModelManager:
    """
    is the main class of the package responsible
    to initialize every mode, provide train, test
    and validation functionality - as well as the
    avg losses and accuracy results.
    """

    def __init__(
        self,
        device: "torch.device",
        chosen_models: "list[str]" = CHOSEN_MODELS,
        save_as_yaml: "bool" = SAVE_AS_YAML,
    ):
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer: "optim.Adam | None" = None
        self.current_model: "NN_MODEL | None" = None
        self.chosen_models = chosen_models
        self.save_as_yaml = save_as_yaml
        self.device = device

    def init_model(
        self,
        model_name: "str",
        in_channels: "int",
        num_classes: "int",
        learning_rate: "float",
    ) -> "None":
        if model_name == ModelChoice.CUSTOM:
            self.current_model = CustomActionRecogntionNN(in_channels, num_classes).to(
                self.device
            )
        else:
            self.current_model = PretrainedNN(in_channels, num_classes).to(self.device)
        # now that the model is set, initialize_the optimizer
        self.optimizer = optim.Adam(self.current_model.parameters(), lr=learning_rate)  # type: ignore
        print(f"ModelManager:: Initialized {model_name} NN, optimizer and criterion")

    def _operate(self, loader: "DataLoader", mode: "int") -> "float":
        correct = 0
        total = 0
        total_loss = 0.0

        # check if the model has been initialized
        if self.current_model is None or self.optimizer is None:
            raise ModelNotInitializedError("Model and Optimizer not initialized yet")

        if mode == ModelOperation.TRAIN:
            self.current_model.train()
        else:
            # otherwise run evaluation
            self.current_model.eval()

        # same pattern for all cases
        for inputs, targets in loader:
            # forward
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.current_model(inputs)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()
            # if train then backwards too
            if mode == ModelOperation.TRAIN:
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            elif mode == ModelOperation.TEST:
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        return (
            # return average train/val loss
            round(total_loss / len(loader), 6)
            # if test mode return accuracy
            if mode != ModelOperation.TEST
            else round(100 * correct / total, 6)
        )

    def train(
        self,
        train_loader: "DataLoader",
    ) -> "float":
        """
        covers the training stage for a given loader
        following the basic operation with grad
        calculation.

        ::returns:: avg train loss (float).
        """
        return self._operate(train_loader, mode=ModelOperation.TRAIN)

    def test(
        self,
        test_loader: "DataLoader",
    ) -> "float":
        """
        covers the testing stage for a given test loader following
        the basic operation with no grad calculation.

        ::returns:: accuracy (float).
        """
        with torch.no_grad():
            return self._operate(test_loader, mode=ModelOperation.TEST)

    def validate(
        self,
        validation_loader: "DataLoader",
    ) -> "float":
        """
        covers the validation stage for a given loader following
        the basic operation with no grad calculation.

        ::returns:: avg val loss (float).
        """
        with torch.no_grad():
            return self._operate(validation_loader, mode=ModelOperation.VALIDATE)
