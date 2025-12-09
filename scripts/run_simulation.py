#!/usr/bin/env python3
"""
ShapeFL Simulation Mode
=======================
Run all components (cloud, edges, nodes) on a single machine for testing.
"""

import os
import sys
import time
import argparse
from multiprocessing import Process

# Add implementation directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATH_CONFIG
from data.data_loader import (
    load_fmnist_data,
    create_non_iid_partitions,
    save_partitions,
)


def run_cloud(port):
    """Run cloud server in subprocess."""
    from cloud.cloud_server import CloudServer

    server = CloudServer(host="127.0.0.1", port=port, use_gpu=True)
    server.run()


def run_edge(edge_id, port, cloud_port):
    """Run edge aggregator in subprocess."""
    from edge.edge_aggregator import EdgeAggregator

    edge = EdgeAggregator(
        edge_id=edge_id,
        host="127.0.0.1",
        port=port,
        cloud_host="127.0.0.1",
        cloud_port=cloud_port,
    )
    edge.run(training=False)  # Don't auto-start training loop


def run_node(node_id, port, cloud_port, partitions_file):
    """Run computing node in subprocess."""
    from node.computing_node import ComputingNode

    node = ComputingNode(
        node_id=node_id,
        host="127.0.0.1",
        port=port,
        cloud_host="127.0.0.1",
        cloud_port=cloud_port,
        partitions_file=partitions_file,
    )
    node.run(training=False)  # Don't auto-start training loop


def main():
    parser = argparse.ArgumentParser(description="ShapeFL Simulation Mode")
    parser.add_argument(
        "--num-edges", type=int, default=2, help="Number of edge aggregators"
    )
    parser.add_argument(
        "--num-nodes", type=int, default=4, help="Number of computing nodes"
    )
    parser.add_argument(
        "--cloud-port", type=int, default=5000, help="Cloud server port"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ShapeFL Simulation Mode")
    print("=" * 60)
    print(f"Cloud server: localhost:{args.cloud_port}")
    print(f"Edge aggregators: {args.num_edges}")
    print(f"Computing nodes: {args.num_nodes}")
    print("=" * 60 + "\n")

    # Create data partitions
    print("Creating data partitions...")
    train_dataset, _ = load_fmnist_data()
    partitions = create_non_iid_partitions(
        train_dataset,
        num_nodes=args.num_nodes,
        shard_size=15,
        shards_per_node=12,
        classes_per_node=4,
    )

    partitions_file = os.path.join(PATH_CONFIG.partitions_dir, "partitions.json")
    PATH_CONFIG.ensure_dirs()
    save_partitions(partitions, partitions_file)

    processes = []

    try:
        # Start cloud server
        print("\nStarting cloud server...")
        cloud_proc = Process(target=run_cloud, args=(args.cloud_port,))
        cloud_proc.start()
        processes.append(cloud_proc)
        time.sleep(3)  # Wait for cloud to start

        # Start edge aggregators
        print("Starting edge aggregators...")
        edge_port_base = 5100
        for i in range(args.num_edges):
            edge_port = edge_port_base + i
            edge_proc = Process(
                target=run_edge, args=(f"edge_{i}", edge_port, args.cloud_port)
            )
            edge_proc.start()
            processes.append(edge_proc)
            time.sleep(1)

        time.sleep(2)  # Wait for edges to register

        # Start computing nodes
        print("Starting computing nodes...")
        node_port_base = 5200
        for i in range(args.num_nodes):
            node_port = node_port_base + i
            node_proc = Process(
                target=run_node,
                args=(f"node_{i}", node_port, args.cloud_port, partitions_file),
            )
            node_proc.start()
            processes.append(node_proc)
            time.sleep(0.5)

        print("\n" + "=" * 60)
        print("All components started!")
        print("=" * 60)
        print("\nCloud server: http://127.0.0.1:5000")
        print("Check status: curl http://127.0.0.1:5000/status")
        print("\nPress Ctrl+C to stop all components...")

        # Wait for interrupt
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.join(timeout=5)
        print("All components stopped.")


if __name__ == "__main__":
    main()
