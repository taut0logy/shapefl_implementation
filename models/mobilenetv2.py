"""
MobileNetV2 Model for CIFAR-10 Classification
==============================================
Implementation of MobileNetV2 adapted for CIFAR-10 (32x32 colour images, 10 classes).

Reference: Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
           (CVPR 2018)
Used in ShapeFL paper for CIFAR-10 experiments.

Note: The standard ImageNet MobileNetV2 uses a 7x7 global average pool
      after a 224x224 input. For 32x32 CIFAR inputs the feature map before
      the pool is 1x1, so we replace the classifier head accordingly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden = in_channels * expand_ratio

        layers = []
        if expand_ratio != 1:
            # Point-wise expansion
            layers.extend([
                nn.Conv2d(in_channels, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ])
        # Depth-wise convolution
        layers.extend([
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
        ])
        # Point-wise linear projection
        layers.extend([
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2 architecture adapted for CIFAR-10 (32x32).

    Key differences from ImageNet version:
    - First conv stride is 1 (not 2) because input is 32x32 not 224x224
    - Last block strides adjusted so final feature map is 1x1 after avg pool

    The output (classifier) layer is named ``classifier`` and is used for
    ShapeFL similarity computation.
    """

    # (expansion, out_channels, num_blocks, stride)
    _cfg = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),   # stride 1 for CIFAR (orig: 2)
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.linear_layer_name = "classifier"  # For ShapeFL similarity

        # First conv
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        # Inverted residual blocks
        in_ch = 32
        layers = []
        for t, c, n, s in self._cfg:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(in_ch, c, stride, t))
                in_ch = c
        self.blocks = nn.Sequential(*layers)

        # Last conv + pool
        self.last_conv = nn.Sequential(
            nn.Conv2d(in_ch, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.blocks(x)
        x = self.last_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_linear_layer_params(self) -> torch.Tensor:
        """Get classifier weights + biases (for ShapeFL similarity)."""
        weight = self.classifier.weight.data.flatten()
        bias = self.classifier.bias.data.flatten()
        return torch.cat([weight, bias])

    def get_linear_layer_size(self) -> int:
        return self.classifier.weight.numel() + self.classifier.bias.numel()


if __name__ == "__main__":
    model = MobileNetV2(num_classes=10)
    print("MobileNetV2 (CIFAR-10):")
    num_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"Total parameters: {num_params:,}")
    print(f"Model size: {size_mb:.3f} MB")
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Linear layer params: {model.get_linear_layer_size()}")
