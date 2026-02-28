"""
Data Loader and Non-IID Partitioning for ShapeFL
=================================================
Implements data loading and non-IID partitioning following the paper's methodology.

Paper Method (Section V-A):
- Each training dataset is divided into shards of size 15
- Each computing node is distributed with s shards from k classes
- For FMNIST/CIFAR-10: s=12, k=4
- For CIFAR-100: s=100, k=20

Supported datasets:
- fmnist   : Fashion-MNIST (1×28×28, 10 classes)
- cifar10  : CIFAR-10      (3×32×32, 10 classes)
- cifar100 : CIFAR-100     (3×32×32, 100 classes)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Dict, Tuple, Optional
import pandas as pd


# ── Dataset metadata (used by the simulation script) ────────────────────────
DATASET_INFO = {
    "fmnist": {
        "num_classes": 10,
        "input_channels": 1,
        "input_size": (28, 28),
        "shards_per_node": 12,   # s in paper
        "classes_per_node": 4,   # k in paper
    },
    "cifar10": {
        "num_classes": 10,
        "input_channels": 3,
        "input_size": (32, 32),
        "shards_per_node": 12,
        "classes_per_node": 4,
    },
    "cifar100": {
        "num_classes": 100,
        "input_channels": 3,
        "input_size": (32, 32),
        "shards_per_node": 100,
        "classes_per_node": 20,
    },
}

# Default model for each dataset (paper pairings)
DATASET_DEFAULT_MODEL = {
    "fmnist": "lenet5",
    "cifar10": "mobilenetv2",
    "cifar100": "resnet18",
}


class FMNISTDataset(Dataset):
    """Fashion-MNIST dataset wrapper."""

    def __init__(self, data: np.ndarray, targets: np.ndarray, transform=None):
        self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(1) / 255.0
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            img = self.transform(img)

        return img, target


def load_fmnist_data(
    data_dir: str = None, use_csv: bool = True
) -> Tuple[Dataset, Dataset]:
    """
    Load Fashion-MNIST dataset.

    Args:
        data_dir: Directory containing the data files
        use_csv: If True, load from CSV files; otherwise use torchvision

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    if data_dir is None:
        # Default to the dataset directory in the project
        data_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")

    if use_csv and os.path.exists(os.path.join(data_dir, "fashion-mnist_train.csv")):
        print("Loading Fashion-MNIST from CSV files...")

        # Load training data
        train_df = pd.read_csv(os.path.join(data_dir, "fashion-mnist_train.csv"))
        train_labels = train_df.iloc[:, 0].values
        train_images = train_df.iloc[:, 1:].values.reshape(-1, 28, 28)

        # Load test data
        test_df = pd.read_csv(os.path.join(data_dir, "fashion-mnist_test.csv"))
        test_labels = test_df.iloc[:, 0].values
        test_images = test_df.iloc[:, 1:].values.reshape(-1, 28, 28)

        train_dataset = FMNISTDataset(train_images, train_labels)
        test_dataset = FMNISTDataset(test_images, test_labels)

    else:
        print("Loading Fashion-MNIST from torchvision...")

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        train_dataset = datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=transform
        )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")

    return train_dataset, test_dataset


def load_cifar10_data(
    data_dir: str = None,
) -> Tuple[Dataset, Dataset]:
    """
    Load CIFAR-10 dataset via torchvision (auto-downloads if needed).

    Args:
        data_dir: Root directory for the dataset cache.

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")

    print("Loading CIFAR-10 from torchvision...")

    # Paper does not specify data augmentation – use simple normalisation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test,
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    return train_dataset, test_dataset


def load_cifar100_data(
    data_dir: str = None,
) -> Tuple[Dataset, Dataset]:
    """
    Load CIFAR-100 dataset via torchvision (auto-downloads if needed).

    Args:
        data_dir: Root directory for the dataset cache.

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")

    print("Loading CIFAR-100 from torchvision...")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train,
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test,
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    return train_dataset, test_dataset


def load_data(
    dataset_name: str = "fmnist",
    data_dir: str = None,
) -> Tuple[Dataset, Dataset]:
    """
    Unified data-loading entry point.

    Args:
        dataset_name: One of "fmnist", "cifar10", "cifar100".
        data_dir: Root directory for dataset cache (default: ../dataset).

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    name = dataset_name.lower()
    if name == "fmnist":
        return load_fmnist_data(data_dir=data_dir)
    elif name == "cifar10":
        return load_cifar10_data(data_dir=data_dir)
    elif name == "cifar100":
        return load_cifar100_data(data_dir=data_dir)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Choose from: fmnist, cifar10, cifar100"
        )


def create_non_iid_partitions(
    dataset: Dataset,
    num_nodes: int,
    shard_size: int = 15,
    shards_per_node: int = 12,
    classes_per_node: int = 4,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Create non-IID data partitions following the paper's methodology.

    Paper Method:
    - Divide dataset into shards of size `shard_size`
    - Each node gets `shards_per_node` shards from `classes_per_node` classes
    - This creates heterogeneous (non-IID) data distribution

    Args:
        dataset: The full training dataset
        num_nodes: Number of computing nodes
        shard_size: Size of each shard (default: 15)
        shards_per_node: Number of shards per node (s in paper, default: 12)
        classes_per_node: Number of classes per node (k in paper, default: 4)
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping node_id to list of sample indices
    """
    np.random.seed(seed)

    # Get all labels
    if hasattr(dataset, "targets"):
        if isinstance(dataset.targets, torch.Tensor):
            labels = dataset.targets.numpy()
        else:
            labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = len(np.unique(labels))

    # Group indices by class
    class_indices = {c: np.where(labels == c)[0].tolist() for c in range(num_classes)}

    # Shuffle indices within each class
    for c in range(num_classes):
        np.random.shuffle(class_indices[c])

    # Create shards for each class
    class_shards = {}
    for c in range(num_classes):
        indices = class_indices[c]
        num_shards = len(indices) // shard_size
        shards = [
            indices[i * shard_size : (i + 1) * shard_size] for i in range(num_shards)
        ]
        class_shards[c] = shards

    print(
        f"Created shards per class: {[len(class_shards[c]) for c in range(num_classes)]}"
    )

    # Assign shards to nodes
    partitions = {n: [] for n in range(num_nodes)}

    for node_id in range(num_nodes):
        # Select k random classes for this node
        available_classes = [c for c in range(num_classes) if len(class_shards[c]) > 0]

        if len(available_classes) < classes_per_node:
            selected_classes = available_classes
        else:
            selected_classes = np.random.choice(
                available_classes, classes_per_node, replace=False
            )

        # Calculate shards per class
        shards_per_class = shards_per_node // len(selected_classes)
        extra_shards = shards_per_node % len(selected_classes)

        # Assign shards from selected classes
        for i, c in enumerate(selected_classes):
            n_shards = shards_per_class + (1 if i < extra_shards else 0)
            for _ in range(n_shards):
                if len(class_shards[c]) > 0:
                    shard = class_shards[c].pop()
                    partitions[node_id].extend(shard)

    # Print partition statistics
    print("\nPartition Statistics:")
    for node_id in range(num_nodes):
        node_labels = labels[partitions[node_id]]
        unique, counts = np.unique(node_labels, return_counts=True)
        print(
            f"  Node {node_id}: {len(partitions[node_id])} samples, "
            f"classes: {dict(zip(unique.tolist(), counts.tolist()))}"
        )

    return partitions


def get_node_dataloader(
    dataset: Dataset, indices: List[int], batch_size: int = 32, shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader for a specific node's data partition.

    Args:
        dataset: Full dataset
        indices: Indices assigned to this node
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader for the node
    """
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def save_partitions(partitions: Dict[int, List[int]], filepath: str):
    """Save partitions to a JSON file."""
    # Convert keys to strings for JSON
    partitions_str = {str(k): v for k, v in partitions.items()}

    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(partitions_str, f)
    print(f"Partitions saved to {filepath}")


def load_partitions(filepath: str) -> Dict[int, List[int]]:
    """Load partitions from a JSON file."""
    with open(filepath, "r") as f:
        partitions_str = json.load(f)
    # Convert keys back to integers
    partitions = {int(k): v for k, v in partitions_str.items()}
    print(f"Partitions loaded from {filepath}")
    return partitions


def get_data_distribution(dataset: Dataset, indices: List[int]) -> Dict[int, int]:
    """
    Get the class distribution for a subset of the dataset.

    Args:
        dataset: Full dataset
        indices: Indices of the subset

    Returns:
        Dictionary mapping class_id to count
    """
    if hasattr(dataset, "targets"):
        if isinstance(dataset.targets, torch.Tensor):
            labels = dataset.targets[indices].numpy()
        else:
            labels = np.array(dataset.targets)[indices]
    else:
        labels = np.array([dataset[i][1] for i in indices])

    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


if __name__ == "__main__":
    # Test data loading and partitioning
    print("Testing Data Loader and Partitioning")
    print("=" * 50)

    # Load data
    train_dataset, test_dataset = load_fmnist_data()

    # Create non-IID partitions for 5 nodes (simulating 5 computing nodes)
    num_nodes = 5
    partitions = create_non_iid_partitions(
        train_dataset,
        num_nodes=num_nodes,
        shard_size=15,
        shards_per_node=12,
        classes_per_node=4,
    )

    # Test dataloader creation
    print("\nTesting DataLoader creation:")
    for node_id in range(num_nodes):
        loader = get_node_dataloader(train_dataset, partitions[node_id], batch_size=32)
        print(
            f"  Node {node_id}: {len(loader)} batches, {len(partitions[node_id])} samples"
        )

    # Save and load test
    save_partitions(
        partitions,
        os.path.join(os.path.dirname(__file__), "..", "partitions", "test_partitions.json"),
    )
    loaded = load_partitions(os.path.join(os.path.dirname(__file__), "..", "partitions", "test_partitions.json"))
    assert partitions == loaded, "Partition save/load mismatch!"
    print("\nPartition save/load test passed!")

    # Cleanup
    os.remove(os.path.join(os.path.dirname(__file__), "..", "partitions", "test_partitions.json"))
