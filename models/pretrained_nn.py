import torch.nn as nn
from torchvision import models


class PretrainedNN(nn.Module):
    """
    a pretrained solution that utilizes the resnet18 pretrained
    model, aiming to classify human action from still images
    """

    def __init__(self, in_channels: "int", num_classes: "int"):
        super(PretrainedNN, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.conv1.weight.data = self.resnet.conv1.weight.data.mean(
            dim=1, keepdim=True
        )
        self.resnet.layer3 = nn.Sequential(self.resnet.layer3, nn.Dropout(p=0.3))
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)
