import torch
import torch.nn as nn


class LinearHeader(nn.Module):
    def __init__(self, in_features: int, n_classes: int):
        super().__init__()

        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.fc(x)
        return x
