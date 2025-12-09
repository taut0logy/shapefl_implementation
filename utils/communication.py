"""
Communication Utilities for ShapeFL
====================================
HTTP-based communication helpers for exchanging models and data
between cloud server, edge aggregators, and computing nodes.
"""

import json
import gzip
import base64
import requests
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from flask import request
import io


def model_to_bytes(model: nn.Module) -> bytes:
    """
    Serialize a PyTorch model to bytes.

    Args:
        model: PyTorch model

    Returns:
        Serialized model as bytes
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def bytes_to_model(model_bytes: bytes, model_template: nn.Module) -> nn.Module:
    """
    Deserialize bytes back to a PyTorch model.

    Args:
        model_bytes: Serialized model bytes
        model_template: Model instance to load state into

    Returns:
        Model with loaded state
    """
    buffer = io.BytesIO(model_bytes)
    state_dict = torch.load(buffer, map_location="cpu", weights_only=True)
    model_template.load_state_dict(state_dict)
    return model_template


def compress_model(model_bytes: bytes) -> str:
    """
    Compress model bytes using gzip and encode as base64 string.

    Args:
        model_bytes: Serialized model bytes

    Returns:
        Base64-encoded compressed string
    """
    compressed = gzip.compress(model_bytes)
    return base64.b64encode(compressed).decode("utf-8")


def decompress_model(compressed_str: str) -> bytes:
    """
    Decompress base64-encoded gzipped model.

    Args:
        compressed_str: Base64-encoded compressed string

    Returns:
        Original model bytes
    """
    compressed = base64.b64decode(compressed_str.encode("utf-8"))
    return gzip.decompress(compressed)


def state_dict_to_json(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Convert state dict to JSON-serializable format.

    Args:
        state_dict: PyTorch state dictionary

    Returns:
        JSON-serializable dictionary
    """
    return {
        name: {
            "data": param.cpu().numpy().tolist(),
            "shape": list(param.shape),
            "dtype": str(param.dtype),
        }
        for name, param in state_dict.items()
    }


def json_to_state_dict(json_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Convert JSON dictionary back to state dict.

    Args:
        json_dict: JSON dictionary with parameter data

    Returns:
        PyTorch state dictionary
    """
    state_dict = {}
    for name, param_info in json_dict.items():
        data = np.array(param_info["data"])
        tensor = torch.tensor(data)
        state_dict[name] = tensor
    return state_dict


def send_model(
    url: str,
    model: nn.Module,
    metadata: Optional[Dict[str, Any]] = None,
    compress: bool = True,
    timeout: int = 300,
) -> requests.Response:
    """
    Send a model to a remote endpoint.

    Args:
        url: Target URL
        model: PyTorch model to send
        metadata: Additional metadata to include
        compress: Whether to compress the model
        timeout: Request timeout in seconds

    Returns:
        Response object
    """
    model_bytes = model_to_bytes(model)

    if compress:
        model_data = compress_model(model_bytes)
        payload = {
            "model": model_data,
            "compressed": True,
            "size_bytes": len(model_bytes),
        }
    else:
        # Use JSON serialization for uncompressed
        state_dict = model.state_dict()
        payload = {"model": state_dict_to_json(state_dict), "compressed": False}

    if metadata:
        payload["metadata"] = metadata

    response = requests.post(
        url, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout
    )
    return response


def receive_model(
    request_data: Dict[str, Any], model_template: nn.Module
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Receive and deserialize a model from request data.

    Args:
        request_data: Request JSON data
        model_template: Model instance to load state into

    Returns:
        Tuple of (loaded model, metadata)
    """
    compressed = request_data.get("compressed", False)
    metadata = request_data.get("metadata", {})

    if compressed:
        model_str = request_data["model"]
        model_bytes = decompress_model(model_str)
        model = bytes_to_model(model_bytes, model_template)
    else:
        state_dict = json_to_state_dict(request_data["model"])
        model_template.load_state_dict(state_dict)
        model = model_template

    return model, metadata


def send_json(url: str, data: Dict[str, Any], timeout: int = 300) -> requests.Response:
    """
    Send JSON data to a remote endpoint.

    Args:
        url: Target URL
        data: Data to send
        timeout: Request timeout

    Returns:
        Response object
    """
    response = requests.post(
        url, json=data, headers={"Content-Type": "application/json"}, timeout=timeout
    )
    return response


def receive_json(request_obj: request) -> Dict[str, Any]:
    """
    Receive JSON data from a Flask request.

    Args:
        request_obj: Flask request object

    Returns:
        Parsed JSON data
    """
    return request_obj.get_json()


def get_model_size_bytes(model: nn.Module) -> int:
    """
    Calculate the size of a model in bytes.

    Args:
        model: PyTorch model

    Returns:
        Size in bytes
    """
    return len(model_to_bytes(model))


def get_compressed_size_bytes(model: nn.Module) -> int:
    """
    Calculate the compressed size of a model in bytes.

    Args:
        model: PyTorch model

    Returns:
        Compressed size in bytes
    """
    model_bytes = model_to_bytes(model)
    compressed = gzip.compress(model_bytes)
    return len(compressed)


if __name__ == "__main__":
    # Test communication utilities
    import sys

    sys.path.insert(0, "..")
    from models.lenet5 import LeNet5

    print("Testing Communication Utilities")
    print("=" * 50)

    # Create a test model
    model = LeNet5()

    # Test serialization
    model_bytes = model_to_bytes(model)
    print(f"Model size: {len(model_bytes):,} bytes")

    # Test compression
    compressed = compress_model(model_bytes)
    print(f"Compressed size: {len(compressed):,} bytes")
    print(f"Compression ratio: {len(model_bytes) / len(compressed):.2f}x")

    # Test decompression
    decompressed = decompress_model(compressed)
    assert decompressed == model_bytes, "Decompression mismatch!"

    # Test model restoration
    model2 = LeNet5()
    model2 = bytes_to_model(decompressed, model2)

    # Verify parameters match
    for (n1, p1), (n2, p2) in zip(
        model.state_dict().items(), model2.state_dict().items()
    ):
        assert torch.equal(p1, p2), f"Parameter mismatch: {n1}"

    print("\nAll tests passed!")
