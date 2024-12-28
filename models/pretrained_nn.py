import torch.nn as nn
from torchvision import models


class PretrainedNN(nn.Module):
    def __init__(self):
        super(PretrainedNN, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.conv2 = nn.Conv2d(
            64, 128, kernel_size=5, stride=2, padding=3, bias=False
        )
        self.resnet.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.resnet(x)
