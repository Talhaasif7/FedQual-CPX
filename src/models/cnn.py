"""
Neural network models for FedQual-CPX experiments.

Models:
    - SmallCNN: Lightweight CNN for rapid iteration on CIFAR-10
    - (ResNet18 wrapper will be added for final experiments)

Per Section 27: keep models small — client-selection research should
not become primarily a model-training benchmark.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """Small CNN for CIFAR-10 federated experiments.

    Architecture:
        Conv2d(3, 32, 3) → ReLU → MaxPool
        Conv2d(32, 64, 3) → ReLU → MaxPool
        Conv2d(64, 64, 3) → ReLU
        Linear(1024, 128) → ReLU → Dropout
        Linear(128, num_classes)

    This is intentionally small for rapid iteration.
    Final experiments should use ResNet-18.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, 3, 32, 32]
        x = self.pool(F.relu(self.conv1(x)))   # [B, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))   # [B, 64, 8, 8]
        x = self.pool(F.relu(self.conv3(x)))   # [B, 64, 4, 4]
        x = x.view(x.size(0), -1)             # [B, 1024]
        x = F.relu(self.fc1(x))                # [B, 128]
        x = self.dropout(x)
        x = self.fc2(x)                        # [B, num_classes]
        return x


class FEMNISTCNN(nn.Module):
    """Convolutional neural network for FEMNIST character classification (Section 27.2).

    Input: [B, 1, 28, 28] grayscale images
    Output: [B, num_classes] logits (62 classes: 10 digits + 26 upper + 26 lower)
    """

    def __init__(self, num_classes: int = 62) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, 1, 28, 28]
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 7, 7]
        x = x.view(x.size(0), -1)             # [B, 3136]
        x = F.relu(self.fc1(x))               # [B, 128]
        x = self.dropout(x)
        x = self.fc2(x)                       # [B, num_classes]
        return x


def create_model(name: str, num_classes: int = 10) -> nn.Module:
    """Factory function to create models by name.

    Args:
        name: Model name — 'SmallCNN', 'FEMNISTCNN', 'ShakespeareLSTM'.
        num_classes: Number of output classes (or vocab_size for LSTM).

    Returns:
        PyTorch nn.Module.
    """
    if name == "SmallCNN":
        return SmallCNN(num_classes=num_classes)
    elif name == "FEMNISTCNN":
        return FEMNISTCNN(num_classes=num_classes)
    elif name == "ShakespeareLSTM":
        from src.models.lstm import ShakespeareLSTM
        return ShakespeareLSTM(vocab_size=num_classes)

    raise ValueError(f"Unknown model: {name}. Available: ['SmallCNN', 'FEMNISTCNN', 'ShakespeareLSTM']")


