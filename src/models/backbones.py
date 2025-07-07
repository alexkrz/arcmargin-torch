import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    """ConvNet

    Adapted from: https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)  # 28*28->32*32-->28*28  # fmt: skip
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)  # 10*10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, n_features)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    backbone = ConvNet(n_features=3)
    summary(backbone, input_size=(1, 1, 28, 28))
