"""
Metrics Tracking Utilities for ShapeFL
======================================
Functions for tracking accuracy, communication cost, and other metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any
import json
import os
from datetime import datetime


class MetricsTracker:
    """
    Track and record metrics throughout the training process.
    """

    def __init__(self, save_dir: str = "metrics"):
        """
        Initialize the metrics tracker.

        Args:
            save_dir: Directory to save metrics
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Metrics storage
        self.metrics: Dict[str, List[Any]] = {
            "round": [],
            "accuracy": [],
            "loss": [],
            "communication_cost": [],
            "cumulative_comm_cost": [],
            "timestamp": [],
        }

        self.cumulative_comm_cost = 0
        self.start_time = datetime.now()

    def record(
        self,
        round_num: int,
        accuracy: float,
        loss: float = 0.0,
        communication_cost: int = 0,
    ):
        """
        Record metrics for a training round.

        Args:
            round_num: Current round number
            accuracy: Test accuracy
            loss: Training loss
            communication_cost: Bytes transferred this round
        """
        self.cumulative_comm_cost += communication_cost

        self.metrics["round"].append(round_num)
        self.metrics["accuracy"].append(accuracy)
        self.metrics["loss"].append(loss)
        self.metrics["communication_cost"].append(communication_cost)
        self.metrics["cumulative_comm_cost"].append(self.cumulative_comm_cost)
        self.metrics["timestamp"].append(str(datetime.now() - self.start_time))

        print(
            f"Round {round_num}: Accuracy={accuracy:.4f}, "
            f"Loss={loss:.4f}, "
            f"CommCost={communication_cost:,} bytes, "
            f"Cumulative={self.cumulative_comm_cost:,} bytes"
        )

    def save(self, filename: str = "training_metrics.json"):
        """Save metrics to a JSON file."""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {filepath}")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the training metrics."""
        if len(self.metrics["accuracy"]) == 0:
            return {}

        return {
            "final_accuracy": self.metrics["accuracy"][-1],
            "best_accuracy": max(self.metrics["accuracy"]),
            "total_rounds": len(self.metrics["round"]),
            "total_communication_cost": self.cumulative_comm_cost,
            "total_time": str(datetime.now() - self.start_time),
        }

    def get_latest(self) -> Dict[str, Any]:
        """Get the latest training metrics."""
        if len(self.metrics["accuracy"]) == 0:
            return {
                "round": 0,
                "accuracy": 0.0,
                "loss": 0.0,
                "communication_cost": 0,
                "cumulative_comm_cost": 0,
            }

        return {
            "round": self.metrics["round"][-1],
            "accuracy": self.metrics["accuracy"][-1],
            "loss": self.metrics["loss"][-1],
            "communication_cost": self.metrics["communication_cost"][-1],
            "cumulative_comm_cost": self.metrics["cumulative_comm_cost"][-1],
        }

    def get_all(self) -> Dict[str, List[Any]]:
        """Get all training metrics history."""
        return self.metrics


def compute_accuracy(
    model: nn.Module, data_loader: DataLoader, device: torch.device = None
) -> float:
    """
    Compute model accuracy on a dataset.

    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation
        device: Device to use

    Returns:
        Accuracy as a float (0-1)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total if total > 0 else 0.0


def compute_communication_cost(
    model: nn.Module,
    num_nodes: int,
    num_edges: int,
    kappa_e: int,
    kappa_c: int,
    compressed: bool = True,
) -> int:
    """
    Compute the communication cost per cloud aggregation round.

    Based on paper's communication model:
    - Nodes send updates to edges: num_nodes * model_size * kappa_e
    - Edges send updates to cloud: num_edges * model_size * 1
    - Cloud broadcasts back to edges: num_edges * model_size * 1
    - Edges broadcast to nodes: num_nodes * model_size * 1

    Args:
        model: PyTorch model
        num_nodes: Number of computing nodes
        num_edges: Number of edge aggregators
        kappa_e: Edge epochs per cloud aggregation
        kappa_c: Not used directly here but for reference
        compressed: Whether models are compressed

    Returns:
        Total bytes transferred per round
    """
    import gzip
    import io

    # Calculate model size
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    model_bytes = buffer.getvalue()

    if compressed:
        compressed_bytes = gzip.compress(model_bytes)
        model_size = len(compressed_bytes)
    else:
        model_size = len(model_bytes)

    # Communication pattern per cloud aggregation round:
    # 1. Each node sends to its edge: num_nodes * kappa_e * model_size
    # 2. Each edge sends to cloud: num_edges * model_size
    # 3. Cloud sends to each edge: num_edges * model_size
    # 4. Each edge sends to its nodes: num_nodes * model_size

    node_to_edge = num_nodes * kappa_e * model_size
    edge_to_cloud = num_edges * model_size
    cloud_to_edge = num_edges * model_size
    edge_to_node = num_nodes * model_size

    total = node_to_edge + edge_to_cloud + cloud_to_edge + edge_to_node

    return total


if __name__ == "__main__":
    # Test metrics utilities
    print("Testing Metrics Utilities")
    print("=" * 50)

    # Create tracker
    tracker = MetricsTracker(save_dir="test_metrics")

    # Record some test metrics
    for i in range(5):
        tracker.record(
            round_num=i + 1,
            accuracy=0.7 + i * 0.05,
            loss=0.5 - i * 0.08,
            communication_cost=250000,
        )

    # Print summary
    print("\nSummary:")
    for k, v in tracker.get_summary().items():
        print(f"  {k}: {v}")

    # Save metrics
    tracker.save("test_metrics.json")

    # Cleanup
    import shutil

    shutil.rmtree("test_metrics")

    print("\nAll tests passed!")
