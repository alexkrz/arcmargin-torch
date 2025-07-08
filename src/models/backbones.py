import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    """ConvNet

    Adapted from: https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """

    def __init__(
        self,
        in_channels: int = 1,
        input_size: int = 28,
        n_features: int = 3,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # Dynamically compute the flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            self._flattened_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self._flattened_size, 120)
        self.fc2 = nn.Linear(120, n_features)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # NOTE: Important: Do not use ReLU on the last FC layer!
        return x


if __name__ == "__main__":
    from torchinfo import summary

    # For MNIST (1 channel)
    print("Backbone MNIST:")
    backbone_mnist = ConvNet(in_channels=1, input_size=28)
    summary(backbone_mnist, input_size=(1, 1, 28, 28))

    # For CIFAR (3 channels)
    print("Backbone CIFAR10:")
    backbone_cifar = ConvNet(in_channels=3, input_size=32)
    summary(backbone_cifar, input_size=(1, 3, 32, 32))
