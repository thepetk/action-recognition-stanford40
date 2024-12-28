import torch.nn as nn
from torchvision import models


class PretrainedNN(nn.Module):
    def __init__(self, in_channels: "int", num_classes: "int"):
        super(PretrainedNN, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.conv2 = nn.Conv2d(
            64, 128, kernel_size=5, stride=2, padding=3, bias=False
        )
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet(x)
