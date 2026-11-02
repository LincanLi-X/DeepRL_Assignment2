"""Question 2 architectures, with PyTorch default initialization and all biases."""

import torch
from torch import nn

from prepare_data import SEED


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1296, 72, bias=True),
            nn.ReLU(),
            nn.Linear(72, 10, bias=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10, bias=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


def build_models() -> dict[str, nn.Module]:
    """Reset the initialization seed separately immediately before each model."""
    torch.manual_seed(SEED)
    mlp = MLP()
    torch.manual_seed(SEED)
    cnn = CNN()
    return {"MLP": mlp, "CNN": cnn}
