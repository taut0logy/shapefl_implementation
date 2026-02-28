"""
ShapeFL Configuration
=====================
Hyperparameters and network configuration for the ShapeFL implementation.
Based on: "A Communication-Efficient Hierarchical Federated Learning Framework
          via Shaping Data Distribution at Edge" (IEEE/ACM ToN, 2024)
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict
import json

# =============================================================================
# HYPERPARAMETERS (from Paper Section V-A)
# =============================================================================


@dataclass
class TrainingConfig:
    """Training hyperparameters following the paper."""

    # Model and Dataset
    model_name: str = "lenet5"
    dataset_name: str = "fmnist"
    num_classes: int = 10
    input_channels: int = 1
    input_size: tuple = (28, 28)

    # Training parameters (Paper Section V-A: batch_size=32, lr stated as 0.001)
    # Using lr=0.01 because it matches the paper's reported ~83% FMNIST accuracy
    batch_size: int = 32
    learning_rate: float = 0.01

    # ShapeFL specific parameters
    kappa_p: int = 30  # Pre-training epochs per node (Algorithm 3, line 2)
    kappa_e: int = 2  # Local epochs before edge aggregation
    kappa_c: int = 10  # Edge epochs before cloud aggregation
    kappa: int = 100  # Total communication rounds to reach target accuracy

    # Algorithm parameters
    gamma: float = 2800.0  # Trade-off weight (Paper: gamma=2800 for best performance)
    T_max: int = 30  # Max iterations for LoS algorithm (Algorithm 2)
    B_e: int = 10  # Max nodes per edge aggregator (constraint 8)

    # Non-IID data partitioning (Paper: s=12, k=4 for FMNIST/CIFAR-10;
    #                                       s=100, k=20 for CIFAR-100)
    shard_size: int = 15  # Size of each shard
    shards_per_node: int = 12  # s: number of shards per node
    classes_per_node: int = 4  # k: number of classes per node


@dataclass
class NetworkConfig:
    """Network configuration for distributed deployment."""

    # Cloud server (your laptop/desktop with GPU)
    cloud_host: str = "192.168.0.100"
    cloud_port: int = 5000

    # Edge aggregators (Raspberry Pis acting as aggregators)
    edge_aggregators: List[Dict] = field(
        default_factory=lambda: [
            {"id": "edge_0", "host": "192.168.0.101", "port": 5001},
            {"id": "edge_1", "host": "192.168.0.102", "port": 5001},
            {"id": "edge_2", "host": "192.168.0.103", "port": 5001},
        ]
    )

    # Computing nodes (Raspberry Pis as worker nodes)
    computing_nodes: List[Dict] = field(
        default_factory=lambda: [
            {"id": "node_0", "host": "192.168.0.111", "port": 5002},
            {"id": "node_1", "host": "192.168.0.112", "port": 5002},
            {"id": "node_2", "host": "192.168.0.113", "port": 5002},
            {"id": "node_3", "host": "192.168.0.114", "port": 5002},
            {"id": "node_4", "host": "192.168.0.115", "port": 5002},
        ]
    )

    # Communication settings
    timeout: int = 300  # Request timeout in seconds
    max_retries: int = 3  # Max retries for failed requests
    compress: bool = True  # Enable gzip compression for model transfer


@dataclass
class PathConfig:
    """File paths configuration."""

    # Base directories
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    data_dir: str = os.path.join(os.path.dirname(base_dir), "..", "dataset")

    # Model checkpoints
    checkpoint_dir: str = os.path.join(base_dir, "checkpoints")

    # Logs and metrics
    log_dir: str = os.path.join(base_dir, "logs")
    metrics_dir: str = os.path.join(base_dir, "metrics")

    # Data partitions (pre-computed non-IID splits)
    partitions_dir: str = os.path.join(base_dir, "partitions")

    def ensure_dirs(self):
        """Create directories if they don't exist."""
        for d in [
            self.checkpoint_dir,
            self.log_dir,
            self.metrics_dir,
            self.partitions_dir,
        ]:
            os.makedirs(d, exist_ok=True)


# =============================================================================
# GLOBAL CONFIG INSTANCES
# =============================================================================

TRAINING_CONFIG = TrainingConfig()
NETWORK_CONFIG = NetworkConfig()
PATH_CONFIG = PathConfig()


# =============================================================================
# CONFIG FILE I/O
# =============================================================================


def save_config(filepath: str = "shapefl_config.json"):
    """Save current configuration to JSON file."""
    config = {
        "training": TRAINING_CONFIG.__dict__,
        "network": NETWORK_CONFIG.__dict__,
        "paths": {
            k: v for k, v in PATH_CONFIG.__dict__.items() if not k.startswith("_")
        },
    }
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Configuration saved to {filepath}")


def load_config(filepath: str = "shapefl_config.json"):
    """Load configuration from JSON file."""
    global TRAINING_CONFIG, NETWORK_CONFIG, PATH_CONFIG

    with open(filepath, "r") as f:
        config = json.load(f)

    # Update training config
    for key, value in config.get("training", {}).items():
        if hasattr(TRAINING_CONFIG, key):
            setattr(TRAINING_CONFIG, key, value)

    # Update network config
    for key, value in config.get("network", {}).items():
        if hasattr(NETWORK_CONFIG, key):
            setattr(NETWORK_CONFIG, key, value)

    print(f"Configuration loaded from {filepath}")


def print_config():
    """Print current configuration."""
    print("\n" + "=" * 60)
    print("ShapeFL Configuration")
    print("=" * 60)

    print("\n[Training Config]")
    for key, value in TRAINING_CONFIG.__dict__.items():
        print(f"  {key}: {value}")

    print("\n[Network Config]")
    print(f"  Cloud Server: {NETWORK_CONFIG.cloud_host}:{NETWORK_CONFIG.cloud_port}")
    print(f"  Edge Aggregators: {len(NETWORK_CONFIG.edge_aggregators)}")
    for e in NETWORK_CONFIG.edge_aggregators:
        print(f"    - {e['id']}: {e['host']}:{e['port']}")
    print(f"  Computing Nodes: {len(NETWORK_CONFIG.computing_nodes)}")
    for n in NETWORK_CONFIG.computing_nodes:
        print(f"    - {n['id']}: {n['host']}:{n['port']}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    print_config()
    PATH_CONFIG.ensure_dirs()
    save_config()
