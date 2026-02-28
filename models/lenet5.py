"""
LeNet-5 Model for Fashion-MNIST Classification
===============================================
Implementation following the original LeNet-5 architecture adapted for
Fashion-MNIST (28x28 grayscale images, 10 classes).

Reference: LeCun et al., "Gradient-based learning applied to document recognition"
Used in ShapeFL paper for FMNIST experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    LeNet-5 architecture for Fashion-MNIST.

    Architecture:
        Input: 1x28x28
        Conv1: 1->6 channels, 5x5 kernel, padding=2 -> 6x28x28
        Pool1: 2x2 max pool -> 6x14x14
        Conv2: 6->16 channels, 5x5 kernel -> 16x10x10
        Pool2: 2x2 max pool -> 16x5x5
        FC1: 400 -> 120
        FC2: 120 -> 84
        FC3: 84 -> 10 (output/linear layer)

    Total parameters: ~61,706
    """

    def __init__(self, num_classes: int = 10, input_channels: int = 1):
        super(LeNet5, self).__init__()
        self.linear_layer_name = "fc3"  # For ShapeFL similarity

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)  # Linear layer for similarity computation

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Conv layers
        x = self.pool(F.relu(self.conv1(x)))  # -> 6x14x14
        x = self.pool(F.relu(self.conv2(x)))  # -> 16x5x5

        # Flatten
        x = x.view(-1, 16 * 5 * 5)

        # FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def get_linear_layer_params(self) -> torch.Tensor:
        """
        Get the parameters of the linear (output) layer.
        Used for computing data distribution similarity S_ij in ShapeFL.

        Returns:
            Flattened tensor of fc3 weights and biases.
        """
        weight = self.fc3.weight.data.flatten()
        bias = self.fc3.bias.data.flatten()
        return torch.cat([weight, bias])

    def get_linear_layer_size(self) -> int:
        """Get the number of parameters in the linear layer."""
        return self.fc3.weight.numel() + self.fc3.bias.numel()


if __name__ == "__main__":
    model = LeNet5()
    print("LeNet-5 Architecture:")
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Linear layer params: {model.get_linear_layer_size()}")
