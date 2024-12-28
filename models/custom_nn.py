import torch.nn as nn


class CustomActionRecogntionNN(nn.Module):
    def __init__(
        self, in_channels: "int", num_classes: "int", image_size: "int"
    ) -> "None":
        super(CustomActionRecogntionNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (image_size // 2) * (image_size // 2), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x: "nn.Sequential") -> "nn.Sequential":
        x = self.encoder(x)
        x = self.fc(x)
        return x
