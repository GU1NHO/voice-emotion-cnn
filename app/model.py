from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

class EmotionCNN(nn.Module):
    """Four-block CNN from the assignment, for (batch, 1, mel, frames)."""
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(1, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(256, num_classes))

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Two 3x3 conv-BN-ReLU operations followed by 2x2 max pooling."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class TransferEmotionCNN(nn.Module):
    """Frozen ImageNet ResNet-18 with a trainable emotion-classification head."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The one-channel log-Mel image becomes RGB, as specified in the slides.
        return self.backbone(x.repeat(1, 3, 1, 1))
