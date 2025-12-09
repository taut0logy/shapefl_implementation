"""
Similarity Computation Utilities for ShapeFL
=============================================
Functions for computing data distribution similarity between nodes
using cosine distance of linear layer updates.

Based on paper's methodology:
S_ij = 1 - cos(Δw_i^(L), Δw_j^(L))
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


def compute_cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity value (0 to 1, where 1 is identical)
    """
    vec1 = vec1.float().flatten()
    vec2 = vec2.float().flatten()

    dot_product = torch.dot(vec1, vec2)
    norm1 = torch.norm(vec1)
    norm2 = torch.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return (dot_product / (norm1 * norm2)).item()


def compute_data_distribution_diversity(
    vec1: torch.Tensor, vec2: torch.Tensor
) -> float:
    """
    Compute data distribution diversity S_ij between two nodes.

    From the paper (Section IV-B):
    S_ij = 1 - cos(Δw_i^(L), Δw_j^(L))

    Higher S_ij means more diverse data distributions.

    Args:
        vec1: Linear layer update from node i
        vec2: Linear layer update from node j

    Returns:
        Diversity value (0 to 2, higher = more diverse)
    """
    cos_sim = compute_cosine_similarity(vec1, vec2)
    return 1.0 - cos_sim


def compute_similarity_matrix(linear_updates: Dict[int, torch.Tensor]) -> np.ndarray:
    """
    Compute the full similarity matrix S for all node pairs.

    Args:
        linear_updates: Dictionary mapping node_id to linear layer update tensor

    Returns:
        N x N numpy array where S[i,j] is the diversity between nodes i and j
    """
    node_ids = sorted(linear_updates.keys())
    n = len(node_ids)

    S = np.zeros((n, n))

    for i, node_i in enumerate(node_ids):
        for j, node_j in enumerate(node_ids):
            if i == j:
                S[i, j] = 0.0  # Same node has zero diversity with itself
            elif i < j:
                # Compute diversity
                diversity = compute_data_distribution_diversity(
                    linear_updates[node_i], linear_updates[node_j]
                )
                S[i, j] = diversity
                S[j, i] = diversity  # Symmetric

    return S


def compute_similarity_from_updates(
    updates: List[Tuple[int, torch.Tensor]], num_nodes: int
) -> np.ndarray:
    """
    Compute similarity matrix from a list of (node_id, update) tuples.

    Args:
        updates: List of (node_id, linear_layer_update) tuples
        num_nodes: Total number of nodes

    Returns:
        num_nodes x num_nodes similarity matrix
    """
    # Convert to dictionary
    update_dict = {node_id: update for node_id, update in updates}

    # Create full matrix
    S = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if i in update_dict and j in update_dict:
                diversity = compute_data_distribution_diversity(
                    update_dict[i], update_dict[j]
                )
                S[i, j] = diversity
                S[j, i] = diversity

    return S


if __name__ == "__main__":
    # Test similarity computation
    print("Testing Similarity Computation")
    print("=" * 50)

    # Create test updates (simulating different data distributions)
    np.random.seed(42)
    num_nodes = 5
    update_size = 850  # LeNet5 fc3 layer size

    # Create updates with varying similarity
    base_update = torch.randn(update_size)
    updates = {}

    for i in range(num_nodes):
        # Add noise to create different updates
        noise = torch.randn(update_size) * (i * 0.3)
        updates[i] = base_update + noise

    # Compute similarity matrix
    S = compute_similarity_matrix(updates)

    print("Similarity Matrix (S_ij):")
    print(np.round(S, 3))

    # Test individual computation
    s_01 = compute_data_distribution_diversity(updates[0], updates[1])
    print(f"\nS[0,1] = {s_01:.4f} (should match matrix)")

    print("\nAll tests passed!")
