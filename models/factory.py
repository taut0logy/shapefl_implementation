"""
Model Factory & Utilities
=========================
Centralised factory for creating models by name, plus serialisation helpers.
Each model file (lenet5.py, mobilenetv2.py, resnet18.py) stays purely about
its own architecture; this module handles selection and shared utilities.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_model(
    model_name: str = "lenet5",
    num_classes: int = 10,
    input_channels: int = 1,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Create and return a model by name.

    Args:
        model_name: One of ``"lenet5"``, ``"mobilenetv2"``, ``"resnet18"``.
        num_classes: Number of output classes.
        input_channels: Number of input channels (used by LeNet-5).
        device: Device to place the model on.

    Returns:
        Initialised model.
    """
    name = model_name.lower()

    if name == "lenet5":
        from .lenet5 import LeNet5
        model = LeNet5(num_classes=num_classes, input_channels=input_channels)
    elif name == "mobilenetv2":
        from .mobilenetv2 import MobileNetV2
        model = MobileNetV2(num_classes=num_classes)
    elif name == "resnet18":
        from .resnet18 import ResNet18
        model = ResNet18(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from: lenet5, mobilenetv2, resnet18"
        )

    if device is not None:
        model = model.to(device)

    return model


# ---------------------------------------------------------------------------
# Size / serialisation helpers
# ---------------------------------------------------------------------------

def get_model_size(model: nn.Module) -> Tuple[int, float]:
    """
    Calculate the number of parameters and size in MB.

    Args:
        model: PyTorch model.

    Returns:
        Tuple of (num_parameters, size_in_mb).
    """
    num_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 * 1024)
    return num_params, size_mb


def model_to_dict(model: nn.Module) -> Dict[str, list]:
    """
    Convert model parameters to a dictionary of lists (JSON-serialisable).

    Args:
        model: PyTorch model.

    Returns:
        Dictionary mapping parameter names to lists.
    """
    return {
        name: param.cpu().detach().numpy().tolist()
        for name, param in model.state_dict().items()
    }


def dict_to_model(model: nn.Module, params_dict: Dict[str, list]) -> nn.Module:
    """
    Load model parameters from a dictionary.

    Args:
        model: PyTorch model to load parameters into.
        params_dict: Dictionary mapping parameter names to lists.

    Returns:
        Model with loaded parameters.
    """
    state_dict = {name: torch.tensor(param) for name, param in params_dict.items()}
    model.load_state_dict(state_dict)
    return model
