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
from typing import Dict, Tuple, Optional


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


def get_model(
    model_name: str = "lenet5",
    num_classes: int = 10,
    input_channels: int = 1,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Factory function to get a model by name.

    Args:
        model_name: Name of the model ("lenet5")
        num_classes: Number of output classes
        input_channels: Number of input channels
        device: Device to place the model on

    Returns:
        Initialized model
    """
    if model_name.lower() == "lenet5":
        model = LeNet5(num_classes=num_classes, input_channels=input_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if device is not None:
        model = model.to(device)

    return model


def get_model_size(model: nn.Module) -> Tuple[int, float]:
    """
    Calculate the number of parameters and size in MB.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (num_parameters, size_in_mb)
    """
    num_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (
        1024 * 1024
    )
    return num_params, size_mb


def model_to_dict(model: nn.Module) -> Dict[str, list]:
    """
    Convert model parameters to a dictionary of lists (for JSON serialization).

    Args:
        model: PyTorch model

    Returns:
        Dictionary mapping parameter names to lists
    """
    return {
        name: param.cpu().detach().numpy().tolist()
        for name, param in model.state_dict().items()
    }


def dict_to_model(model: nn.Module, params_dict: Dict[str, list]) -> nn.Module:
    """
    Load model parameters from a dictionary.

    Args:
        model: PyTorch model to load parameters into
        params_dict: Dictionary mapping parameter names to lists

    Returns:
        Model with loaded parameters
    """
    state_dict = {name: torch.tensor(param) for name, param in params_dict.items()}
    model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    # Test the model
    model = LeNet5()

    # Print architecture
    print("LeNet-5 Architecture:")
    print(model)

    # Calculate size
    num_params, size_mb = get_model_size(model)
    print(f"\nTotal parameters: {num_params:,}")
    print(f"Model size: {size_mb:.3f} MB")

    # Test forward pass
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Test linear layer extraction
    linear_params = model.get_linear_layer_params()
    print(f"\nLinear layer parameters: {linear_params.shape[0]}")
    print(f"  (84*10 weights + 10 biases = {84*10 + 10})")
