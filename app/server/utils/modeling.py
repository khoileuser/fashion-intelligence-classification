"""Image models shared by training and inference."""

from __future__ import annotations

import torch
import torch
from torch import nn


class TrainingBatchNorm2d(nn.BatchNorm2d):
    """Use native CUDA training kernels; retain ordinary checkpoint/inference semantics.

    cuDNN's small-channel training kernels are unusually slow on the project's
    RTX 3060. Native batch normalization computes the same operation and keeps
    the same parameters and running statistics. Evaluation uses the default path.
    """

    def forward(self, inputs):
        if self.training and inputs.is_cuda:
            with torch.backends.cudnn.flags(enabled=False):
                return super().forward(inputs)
        return super().forward(inputs)


class FashionMLP(nn.Module):
    """Fully connected ANN: one or three hidden layers on the same RGB input."""

    def __init__(self, num_classes, dropout=0.2, image_size=(96, 128), deep=False):
        super().__init__()
        widths = (256, 128, 64) if deep else (256,)
        layers = [nn.Flatten()]
        inputs = 3 * image_size[0] * image_size[1]
        for width in widths:
            layers.extend([nn.Linear(inputs, width), nn.ReLU(), nn.Dropout(dropout)])
            inputs = width
        layers.append(nn.Linear(inputs, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, images):
        return self.layers(images)


class TunedCNN(nn.Module):
    """Sample-inspired fourth convolution block and wider dense head."""

    def __init__(self, num_classes, dropout=0.2):
        super().__init__()
        layers = []
        inputs = 3
        for channels in (32, 64, 128, 256):
            layers.extend([nn.Conv2d(inputs, channels, 3, padding=1),
                           TrainingBatchNorm2d(channels), nn.ReLU(), nn.MaxPool2d(2)])
            inputs = channels
        layers.extend([nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
                       nn.Linear(256 * 2 * 2, 256), nn.ReLU(), nn.Dropout(dropout),
                       nn.Linear(256, num_classes)])
        self.layers = nn.Sequential(*layers)

    def forward(self, images):
        return self.layers(images)


class SimpleCNN(nn.Module):
    """Three convolution blocks, followed by a small classification head."""

    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            TrainingBatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            TrainingBatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            TrainingBatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, 3, stride, 1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return self.relu(outputs + residual)


class CompactCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            ResidualBlock(3, 32),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
            ResidualBlock(64, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, num_classes)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class EmbeddingCNN(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            ResidualBlock(3, 32),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
            ResidualBlock(64, 128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.features(inputs), dim=1)
