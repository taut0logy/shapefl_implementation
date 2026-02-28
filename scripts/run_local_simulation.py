#!/usr/bin/env python3
"""
ShapeFL Local Simulation
========================
Clean single-process simulation of the full ShapeFL pipeline (Algorithm 3).
Runs all components in memory without Flask/HTTP for accurate verification
against paper results.

Supports all three model/dataset combinations from the paper:
  1) LeNet-5      + Fashion-MNIST  (10 classes, s=12, k=4)
  2) MobileNetV2  + CIFAR-10       (10 classes, s=12, k=4)
  3) ResNet18     + CIFAR-100     (100 classes, s=100, k=20)

Usage:
    python scripts/run_local_simulation.py --model lenet5   --dataset fmnist   [options]
    python scripts/run_local_simulation.py --model mobilenetv2 --dataset cifar10 [options]
    python scripts/run_local_simulation.py --model resnet18 --dataset cifar100 [options]

Paper reference hyperparameters:
    --kappa-p 30 --kappa-e 1 --kappa-c 10 --kappa 50 --gamma 2800 --batch-size 32 --lr 0.001
"""

import os
import sys
import copy
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Set, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRAINING_CONFIG, PATH_CONFIG
from models.factory import get_model, get_model_size
from data.data_loader import (
    load_data,
    create_non_iid_partitions,
    get_node_dataloader,
    DATASET_INFO,
    DATASET_DEFAULT_MODEL,
)
from algorithms.goa import run_goa
from algorithms.los import run_los
from utils.similarity import compute_similarity_matrix
from utils.aggregation import federated_averaging, get_linear_layer_update


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def local_update(model, data_loader, epochs, lr, device):
    """
    LocalUpdate function from Algorithm 3 (lines 27-33).

    Args:
        model: Model to train (will be modified in-place)
        data_loader: Node's local data loader
        epochs: Number of local epochs (kappa_e)
        lr: Learning rate
        device: Torch device

    Returns:
        Tuple of (trained model, data_size, avg_loss)
    """
    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    data_size = len(data_loader.dataset)

    return model, data_size, avg_loss


def evaluate_model(model, test_loader, device):
    """Evaluate model accuracy on test set."""
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total if total > 0 else 0.0


def weighted_average_models(models, weights, device="cpu"):
    """
    Weighted FedAvg: w_agg = Sum(D_i / D_total * w_i)

    This is used for both edge aggregation (Algorithm 3 line 21)
    and cloud aggregation (Algorithm 3 line 12).
    """
    return federated_averaging(models, weights)


def generate_communication_costs(num_nodes, model_size_bytes):
    """
    Generate simulated communication costs for a LAN setup.

    For a small-scale local experiment, we use simplified costs.
    The paper uses distance-based costs from real network topologies.

    Returns:
        Tuple of (c_ne dict, c_ec dict)
    """
    c_ne = {}
    c_ec = {}

    # Simulate node positions in a line/grid for cost calculation
    np.random.seed(123)
    positions = np.random.rand(num_nodes, 2) * 100  # Random 2D positions

    # Cloud position (fixed)
    cloud_pos = np.array([50.0, 150.0])  # Cloud is "far away"

    for n in range(num_nodes):
        for e in range(num_nodes):
            if n == e:
                c_ne[(n, e)] = 0.0  # Self-association cost = 0
            else:
                # Cost proportional to distance * model_size
                dist = np.linalg.norm(positions[n] - positions[e])
                c_ne[(n, e)] = 0.002 * dist * model_size_bytes

    for e in range(num_nodes):
        # Edge-to-cloud cost (higher multiplier, matching paper's c_ec = 0.02 * d * S_m)
        dist = np.linalg.norm(positions[e] - cloud_pos)
        c_ec[e] = 0.02 * dist * model_size_bytes

    return c_ne, c_ec


def run_simulation(args):
    """
    Run the full ShapeFL simulation implementing Algorithm 3.
    """
    # =========================================================================
    # Setup
    # =========================================================================
    device = get_device()

    print("\n" + "=" * 70)
    print("ShapeFL Local Simulation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Nodes: {args.num_nodes}")
    print(f"kappa_p (pre-train epochs): {args.kappa_p}")
    print(f"kappa_e (local epochs per edge round): {args.kappa_e}")
    print(f"kappa_c (edge rounds per cloud round): {args.kappa_c}")
    print(f"kappa   (total cloud rounds): {args.kappa}")
    print(f"gamma: {args.gamma}")
    print(f"B_e (max nodes per edge): {args.B_e}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Shard size: {args.shard_size}, Shards/node: {args.shards_per_node}, Classes/node: {args.classes_per_node}")
    print("=" * 70)

    # =========================================================================
    # Step 1: Initialize global model (Algorithm 3, line 1)
    # =========================================================================
    print("\n[Step 1] Initializing global model...")

    # Resolve dataset metadata
    ds_info = DATASET_INFO[args.dataset]
    num_classes = ds_info["num_classes"]
    input_channels = ds_info["input_channels"]

    global_model = get_model(
        model_name=args.model,
        num_classes=num_classes,
        input_channels=input_channels,
        device=device,
    )
    num_params, size_mb = get_model_size(global_model)
    print(f"  {args.model}: {num_params:,} parameters, {size_mb:.3f} MB")

    # Save initial model weights for pre-training reference
    initial_state = copy.deepcopy(global_model.state_dict())

    # =========================================================================
    # Step 2: Load data and create non-IID partitions
    # =========================================================================
    print(f"\n[Step 2] Loading {args.dataset.upper()} and creating non-IID partitions...")
    train_dataset, test_dataset = load_data(dataset_name=args.dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    partitions = create_non_iid_partitions(
        train_dataset,
        num_nodes=args.num_nodes,
        shard_size=args.shard_size,
        shards_per_node=args.shards_per_node,
        classes_per_node=args.classes_per_node,
        seed=42,
    )

    # Create data loaders for each node
    node_loaders = {}
    data_sizes = {}
    for node_id in range(args.num_nodes):
        node_loaders[node_id] = get_node_dataloader(
            train_dataset, partitions[node_id], batch_size=args.batch_size
        )
        data_sizes[node_id] = len(partitions[node_id])
        print(f"  Node {node_id}: {data_sizes[node_id]} samples")

    # =========================================================================
    # Step 3: Offline Pre-training Phase (Algorithm 3, lines 2-5)
    # =========================================================================
    print(f"\n[Step 3] Pre-training phase ({args.kappa_p} epochs per node)...")
    linear_updates = {}

    for node_id in range(args.num_nodes):
        print(f"  Pre-training node {node_id}...", end=" ", flush=True)

        # Each node downloads initial model and trains locally
        node_model = copy.deepcopy(global_model)
        node_model, _, _ = local_update(
            node_model, node_loaders[node_id], args.kappa_p, args.lr, device
        )

        # Compute linear layer update: Delta_w_n^l = w_n^l - w^(0)^l (line 4)
        linear_update = get_linear_layer_update(node_model, global_model)
        linear_updates[node_id] = linear_update
        print(f"done (update norm: {linear_update.norm():.4f})")

    # =========================================================================
    # Step 4: Compute similarity matrix S_ij (Algorithm 3, line 5)
    # =========================================================================
    print("\n[Step 4] Computing data distribution diversity matrix S_ij...")
    S = compute_similarity_matrix(linear_updates)
    print(f"  Similarity matrix shape: {S.shape}")
    print(f"  Mean diversity: {S.mean():.4f}, Max: {S.max():.4f}")

    # =========================================================================
    # Step 5: Edge Selection (LoS) and Node Association (GoA) (lines 6-7)
    # =========================================================================
    print(f"\n[Step 5] Running LoS + GoA algorithms...")

    # Generate communication costs
    model_size_bytes = num_params * 4  # float32
    c_ne, c_ec = generate_communication_costs(args.num_nodes, model_size_bytes)

    # All nodes are candidates for edge aggregator (paper: N_e subset of N)
    candidate_edges = list(range(args.num_nodes))
    all_nodes = list(range(args.num_nodes))

    # Run LoS (which internally calls GoA)
    los_result = run_los(
        candidate_edges=candidate_edges,
        all_nodes=all_nodes,
        communication_costs_ne=c_ne,
        communication_costs_ec=c_ec,
        similarity_matrix=S,
        data_sizes=data_sizes,
        kappa_c=args.kappa_c,
        gamma=args.gamma,
        B_e=args.B_e,
        T_max=args.T_max,
    )

    selected_edges = los_result.selected_edges
    associations = los_result.node_associations.associations
    edge_nodes = los_result.node_associations.edge_nodes

    # Normalise numpy ints to native Python ints for clean printing / JSON
    selected_edges = {int(e) for e in selected_edges}
    associations = {int(k): int(v) for k, v in associations.items()}
    edge_nodes = {int(e): [int(n) for n in nodes] for e, nodes in edge_nodes.items()}

    print(f"\n  Selected edge aggregators: {sorted(selected_edges)}")
    for e in sorted(selected_edges):
        nodes_at_e = sorted(edge_nodes[e])
        print(f"    Edge {e}: nodes {nodes_at_e} (data size: {los_result.node_associations.edge_data_sizes[e]})")

    # Verify all nodes assigned
    assigned = set(associations.keys())
    assert assigned == set(all_nodes), f"Not all nodes assigned! Missing: {set(all_nodes) - assigned}"

    # =========================================================================
    # Step 6: Hierarchical Training Loop (Algorithm 3, lines 8-13)
    # =========================================================================
    print(f"\n[Step 6] Starting hierarchical training ({args.kappa} cloud rounds)...")
    print(f"  Structure: {args.kappa} cloud rounds x {args.kappa_c} edge epochs x {args.kappa_e} local epochs")
    total_local_epochs = args.kappa * args.kappa_c * args.kappa_e
    print(f"  Total local training epochs per node: {total_local_epochs}")

    # Restore the initial model w^(0) for the training phase (Algorithm 3, line 1)
    # Same initialization used for pre-training — NOT a new random model
    global_model.load_state_dict(copy.deepcopy(initial_state))

    # Metrics tracking
    metrics = {
        "round": [],
        "accuracy": [],
        "loss": [],
        "local_epochs": [],
    }

    start_time = time.time()

    # Main training loop: for each communication round r = 1, ..., kappa (line 8)
    for cloud_round in range(1, args.kappa + 1):
        round_start = time.time()

        # Edge models start from global model (line 9: send w^(r-1) to edges)
        edge_models = {}
        for e in selected_edges:
            edge_models[e] = copy.deepcopy(global_model)

        # For each edge epoch t = 1, ..., kappa_c (line 16)
        round_losses = []
        for edge_epoch in range(1, args.kappa_c + 1):

            # Each edge broadcasts model to its nodes (line 17)
            # Then each node trains locally for kappa_e epochs (lines 18-19)
            node_models = {}
            node_data_sizes = {}
            for e in selected_edges:
                for n in edge_nodes[e]:
                    # Node n receives edge model (line 28: w_n^(0) <- w_e)
                    node_model = copy.deepcopy(edge_models[e])

                    # LocalUpdate: train for kappa_e epochs (line 30-32)
                    node_model, d_n, loss = local_update(
                        node_model, node_loaders[n], args.kappa_e, args.lr, device
                    )
                    node_models[n] = node_model
                    node_data_sizes[n] = d_n
                    round_losses.append(loss)

            # Edge aggregation (lines 20-21):
            # w_e^(t) = Sum_{n in M_e} (D_n / D_e) * w_n
            for e in selected_edges:
                models_for_edge = [node_models[n] for n in edge_nodes[e]]
                weights_for_edge = [node_data_sizes[n] for n in edge_nodes[e]]
                edge_models[e] = weighted_average_models(
                    models_for_edge, weights_for_edge, device
                )

        # Cloud aggregation (line 12): w^(r) = Sum_{e} (D_e / Sum D_e) * w_e
        edge_model_list = [edge_models[e] for e in sorted(selected_edges)]
        edge_weight_list = [
            los_result.node_associations.edge_data_sizes[e]
            for e in sorted(selected_edges)
        ]
        global_model = weighted_average_models(edge_model_list, edge_weight_list, device)

        # Evaluate (line 13: return w^(kappa))
        accuracy = evaluate_model(global_model, test_loader, device)
        avg_loss = np.mean(round_losses) if round_losses else 0.0
        elapsed = time.time() - round_start
        local_epochs_so_far = cloud_round * args.kappa_c * args.kappa_e

        metrics["round"].append(cloud_round)
        metrics["accuracy"].append(accuracy)
        metrics["loss"].append(avg_loss)
        metrics["local_epochs"].append(local_epochs_so_far)

        print(
            f"  Round {cloud_round:3d}/{args.kappa} | "
            f"Acc: {accuracy:.4f} | "
            f"Loss: {avg_loss:.4f} | "
            f"Local epochs: {local_epochs_so_far} | "
            f"Time: {elapsed:.1f}s"
        )

    # =========================================================================
    # Results Summary
    # =========================================================================
    total_time = time.time() - start_time
    final_accuracy = metrics["accuracy"][-1]
    best_accuracy = max(metrics["accuracy"])
    best_round = metrics["round"][metrics["accuracy"].index(best_accuracy)]

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"Final accuracy:  {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"Best accuracy:   {best_accuracy:.4f} ({best_accuracy*100:.2f}%) at round {best_round}")
    print(f"Total time:      {total_time:.1f}s")
    print(f"Selected edges:  {sorted(selected_edges)}")
    print(f"Node associations:")
    for e in sorted(selected_edges):
        print(f"  Edge {e}: {sorted(edge_nodes[e])}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert numpy types to native Python types for JSON serialization
    def to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return sorted([to_native(x) for x in obj])
        elif isinstance(obj, dict):
            return {str(k): to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_native(x) for x in obj]
        return obj

    results = {
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "num_classes": num_classes,
            "num_nodes": args.num_nodes,
            "kappa_p": args.kappa_p,
            "kappa_e": args.kappa_e,
            "kappa_c": args.kappa_c,
            "kappa": args.kappa,
            "gamma": args.gamma,
            "B_e": args.B_e,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "shard_size": args.shard_size,
            "shards_per_node": args.shards_per_node,
            "classes_per_node": args.classes_per_node,
        },
        "results": {
            "final_accuracy": final_accuracy,
            "best_accuracy": best_accuracy,
            "best_round": best_round,
            "total_time_seconds": total_time,
        },
        "selected_edges": to_native(sorted(list(selected_edges))),
        "node_associations": {str(k): int(v) for k, v in associations.items()},
        "metrics": to_native(metrics),
    }

    results_path = os.path.join(args.output_dir, "simulation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save final model
    model_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(global_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="ShapeFL Local Simulation (Algorithm 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / dataset selection
    parser.add_argument(
        "--model", type=str, default="lenet5",
        choices=["lenet5", "mobilenetv2", "resnet18"],
        help="Model architecture",
    )
    parser.add_argument(
        "--dataset", type=str, default="fmnist",
        choices=["fmnist", "cifar10", "cifar100"],
        help="Dataset to train on",
    )

    # Node configuration
    parser.add_argument("--num-nodes", type=int, default=8, help="Total number of computing nodes")

    # ShapeFL hyperparameters (Paper Section V-A)
    parser.add_argument("--kappa-p", type=int, default=30, help="Pre-training epochs per node")
    parser.add_argument("--kappa-e", type=int, default=1, help="Local epochs per edge round")
    parser.add_argument("--kappa-c", type=int, default=10, help="Edge rounds per cloud round")
    parser.add_argument("--kappa", type=int, default=50, help="Total cloud aggregation rounds")
    parser.add_argument("--gamma", type=float, default=2800.0, help="Trade-off weight")
    parser.add_argument("--B-e", type=int, default=10, help="Max nodes per edge aggregator")
    parser.add_argument("--T-max", type=int, default=30, help="Max LoS iterations")

    # Training parameters
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (SGD)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    # Data partitioning (Paper: s=12, k=4 for FMNIST/CIFAR-10; s=100, k=20 for CIFAR-100)
    # Set to None to auto-detect from dataset
    parser.add_argument("--shard-size", type=int, default=15, help="Shard size")
    parser.add_argument("--shards-per-node", type=int, default=None, help="Shards per node (s). Auto-set if omitted.")
    parser.add_argument("--classes-per-node", type=int, default=None, help="Classes per node (k). Auto-set if omitted.")

    # Output
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # ── Auto-set partitioning params from dataset if not provided ────────
    ds_info = DATASET_INFO.get(args.dataset)
    if ds_info is None:
        parser.error(f"Unknown dataset: {args.dataset}")
    if args.shards_per_node is None:
        args.shards_per_node = ds_info["shards_per_node"]
    if args.classes_per_node is None:
        args.classes_per_node = ds_info["classes_per_node"]

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    run_simulation(args)


if __name__ == "__main__":
    main()
