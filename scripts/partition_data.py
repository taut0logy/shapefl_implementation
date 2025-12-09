"""
Data Partitioning Script for ShapeFL
====================================
Creates non-IID data partitions for computing nodes and saves them to files.
Run this on the cloud server before starting training.
"""

import os
import sys
import json
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRAINING_CONFIG, PATH_CONFIG
from data.data_loader import (
    load_fmnist_data,
    create_non_iid_partitions,
    save_partitions,
)


def main():
    parser = argparse.ArgumentParser(description="Create non-IID data partitions")
    parser.add_argument(
        "--num-nodes", type=int, default=5, help="Number of computing nodes"
    )
    parser.add_argument(
        "--shard-size", type=int, default=15, help="Size of each data shard"
    )
    parser.add_argument(
        "--shards-per-node",
        type=int,
        default=12,
        help="Number of shards per node (s in paper)",
    )
    parser.add_argument(
        "--classes-per-node",
        type=int,
        default=4,
        help="Number of classes per node (k in paper)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for partition files",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set output directory
    output_dir = args.output_dir or PATH_CONFIG.partitions_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("ShapeFL Data Partitioning")
    print("=" * 60)
    print(f"Nodes: {args.num_nodes}")
    print(f"Shard size: {args.shard_size}")
    print(f"Shards per node: {args.shards_per_node}")
    print(f"Classes per node: {args.classes_per_node}")
    print(f"Output directory: {output_dir}")
    print("=" * 60 + "\n")

    # Load dataset
    train_dataset, test_dataset = load_fmnist_data()

    # Create partitions
    partitions = create_non_iid_partitions(
        train_dataset,
        num_nodes=args.num_nodes,
        shard_size=args.shard_size,
        shards_per_node=args.shards_per_node,
        classes_per_node=args.classes_per_node,
        seed=args.seed,
    )

    # Save all partitions to a single file
    all_partitions_file = os.path.join(output_dir, "partitions.json")
    save_partitions(partitions, all_partitions_file)

    # Also save individual partition files for each node
    for node_id, indices in partitions.items():
        node_file = os.path.join(output_dir, f"node_{node_id}_partition.json")
        with open(node_file, "w") as f:
            json.dump({"indices": indices}, f)
        print(f"Saved partition for node {node_id}: {len(indices)} samples")

    print(f"\nPartitions saved to {output_dir}")
    print("Transfer the partitions.json file to each Raspberry Pi node.")


if __name__ == "__main__":
    main()
