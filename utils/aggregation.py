"""
Model Aggregation Utilities for ShapeFL
=======================================
Functions for aggregating model updates using FedAvg and weighted averaging.
"""

import torch
import torch.nn as nn
from typing import List, Dict
import copy


def federated_averaging(
    models: List[nn.Module], weights: List[float] = None
) -> nn.Module:
    """
    Perform Federated Averaging (FedAvg) on a list of models.

    Args:
        models: List of models to aggregate
        weights: Optional weights for weighted averaging (default: uniform)

    Returns:
        Aggregated model
    """
    if len(models) == 0:
        raise ValueError("No models to aggregate")

    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

    # Create a copy of the first model
    aggregated_model = copy.deepcopy(models[0])
    aggregated_state = aggregated_model.state_dict()

    # Initialize aggregated parameters to zero
    for key in aggregated_state:
        aggregated_state[key] = torch.zeros_like(
            aggregated_state[key], dtype=torch.float32
        )

    # Weighted sum of all model parameters
    for model, weight in zip(models, weights):
        model_state = model.state_dict()
        for key in aggregated_state:
            aggregated_state[key] += weight * model_state[key].float()

    aggregated_model.load_state_dict(aggregated_state)
    return aggregated_model


def weighted_averaging(
    state_dicts: List[Dict[str, torch.Tensor]], weights: List[float]
) -> Dict[str, torch.Tensor]:
    """
    Perform weighted averaging on state dictionaries.

    Args:
        state_dicts: List of state dictionaries
        weights: Weights for each state dict (e.g., data sizes)

    Returns:
        Averaged state dictionary
    """
    if len(state_dicts) == 0:
        raise ValueError("No state dicts to aggregate")

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Initialize averaged state dict
    averaged_state = {}
    for key in state_dicts[0]:
        averaged_state[key] = torch.zeros_like(state_dicts[0][key], dtype=torch.float32)

    # Weighted sum
    for state_dict, weight in zip(state_dicts, normalized_weights):
        for key in averaged_state:
            averaged_state[key] += weight * state_dict[key].float()

    return averaged_state


def compute_model_update(
    new_model: nn.Module, old_model: nn.Module
) -> Dict[str, torch.Tensor]:
    """
    Compute the difference between two models (model update).

    Args:
        new_model: Model after training
        old_model: Model before training

    Returns:
        Dictionary of parameter updates (new - old)
    """
    new_state = new_model.state_dict()
    old_state = old_model.state_dict()

    update = {}
    for key in new_state:
        update[key] = new_state[key] - old_state[key]

    return update


def apply_model_update(model: nn.Module, update: Dict[str, torch.Tensor]) -> nn.Module:
    """
    Apply a model update to a base model.

    Args:
        model: Base model
        update: Model update to apply

    Returns:
        Model with update applied
    """
    state = model.state_dict()
    for key in state:
        if key in update:
            state[key] = state[key] + update[key]
    model.load_state_dict(state)
    return model


def get_linear_layer_update(new_model: nn.Module, old_model: nn.Module) -> torch.Tensor:
    """
    Get the update of the linear (output) layer for similarity computation.

    This is used in ShapeFL for computing data distribution similarity S_ij
    based on the cosine distance of linear layer updates.

    Args:
        new_model: Model after training
        old_model: Model before training

    Returns:
        Flattened tensor of linear layer update
    """
    # For LeNet5, the linear layer is 'fc3'
    new_state = new_model.state_dict()
    old_state = old_model.state_dict()

    # Get weight and bias updates for the output layer
    weight_update = new_state["fc3.weight"] - old_state["fc3.weight"]
    bias_update = new_state["fc3.bias"] - old_state["fc3.bias"]

    # Flatten and concatenate
    return torch.cat([weight_update.flatten(), bias_update.flatten()])


if __name__ == "__main__":
    # Test aggregation utilities
    import sys

    sys.path.insert(0, "..")
    from models.lenet5 import LeNet5

    print("Testing Aggregation Utilities")
    print("=" * 50)

    # Create test models
    models = [LeNet5() for _ in range(3)]
    weights = [100, 150, 200]  # Data sizes

    # Modify models slightly to simulate training
    for i, model in enumerate(models):
        for param in model.parameters():
            param.data += (i + 1) * 0.01

    # Test FedAvg
    aggregated = federated_averaging(models, weights)
    print("FedAvg completed successfully")

    # Test weighted averaging
    state_dicts = [m.state_dict() for m in models]
    avg_state = weighted_averaging(state_dicts, weights)
    print("Weighted averaging completed successfully")

    # Test model update computation
    old_model = LeNet5()
    new_model = LeNet5()
    for param in new_model.parameters():
        param.data += 0.1

    update = compute_model_update(new_model, old_model)
    print(f"Model update keys: {list(update.keys())}")

    # Test linear layer update
    linear_update = get_linear_layer_update(new_model, old_model)
    print(f"Linear layer update size: {linear_update.shape[0]}")

    print("\nAll tests passed!")
