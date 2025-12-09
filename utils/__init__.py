"""
Utils module for ShapeFL implementation.
"""

from .communication import (
    send_model,
    receive_model,
    send_json,
    receive_json,
    compress_model,
    decompress_model,
)
from .aggregation import federated_averaging, weighted_averaging, compute_model_update
from .metrics import MetricsTracker, compute_accuracy, compute_communication_cost
from .similarity import compute_cosine_similarity, compute_similarity_matrix

__all__ = [
    "send_model",
    "receive_model",
    "send_json",
    "receive_json",
    "compress_model",
    "decompress_model",
    "federated_averaging",
    "weighted_averaging",
    "compute_model_update",
    "MetricsTracker",
    "compute_accuracy",
    "compute_communication_cost",
    "compute_cosine_similarity",
    "compute_similarity_matrix",
]
