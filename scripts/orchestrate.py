#!/usr/bin/env python3
"""
ShapeFL Main Orchestrator
=========================
Coordinates the complete ShapeFL training workflow:
1. Initialize cloud server
2. Wait for nodes and edges to register
3. Run pre-training phase
4. Execute LoS + GoA algorithms
5. Coordinate hierarchical training
6. Save results and metrics

This script should be run on the cloud server.
"""

import os
import sys
import time
import json
import argparse
import requests

# Add implementation directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRAINING_CONFIG, NETWORK_CONFIG


def wait_for_registrations(cloud_url, expected_nodes, expected_edges, timeout=300):
    """Wait for all nodes and edges to register."""
    print(
        f"\nWaiting for {expected_nodes} nodes and {expected_edges} edges to register..."
    )

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{cloud_url}/status", timeout=10)
            if response.status_code == 200:
                status = response.json()
                nodes = status.get("registered_nodes", 0)
                edges = status.get("registered_edges", 0)

                print(
                    f"  Registered: {nodes}/{expected_nodes} nodes, {edges}/{expected_edges} edges"
                )

                if nodes >= expected_nodes and edges >= expected_edges:
                    print("All devices registered!")
                    return True
        except Exception as e:
            print(f"  Error checking status: {e}")

        time.sleep(5)

    print("Timeout waiting for registrations")
    return False


def trigger_pretrain(nodes_config, cloud_url):
    """Trigger pre-training on all nodes."""
    print("\n" + "=" * 60)
    print("Phase 1: Pre-training")
    print("=" * 60)

    for node in nodes_config:
        node_url = f"http://{node['host']}:{node['port']}"
        try:
            print(f"Triggering pre-train on {node['id']}...")
            response = requests.post(
                f"{node_url}/pretrain/start",
                json={"epochs": TRAINING_CONFIG.kappa_p},
                timeout=30,
            )
            if response.status_code == 200:
                print(f"  {node['id']}: Pre-training started")
            else:
                print(f"  {node['id']}: Failed - {response.text}")
        except Exception as e:
            print(f"  {node['id']}: Error - {e}")

    # Wait for pre-training to complete (check cloud status)
    print("\nWaiting for pre-training to complete...")
    timeout = TRAINING_CONFIG.kappa_p * 60  # Estimate based on epochs
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{cloud_url}/status", timeout=10)
            if response.status_code == 200:
                status = response.json()
                if status.get("pretrain_complete", False):
                    print("Pre-training complete!")
                    return True
        except Exception as e:
            pass
        time.sleep(10)

    print("Pre-training timeout")
    return False


def run_edge_selection(cloud_url):
    """Trigger LoS + GoA algorithm execution on cloud."""
    print("\n" + "=" * 60)
    print("Phase 2: Edge Selection and Node Association")
    print("=" * 60)

    try:
        response = requests.post(f"{cloud_url}/algorithms/run", timeout=120)
        if response.status_code == 200:
            result = response.json()
            print(f"Selected edges: {result.get('selected_edges', [])}")
            print(f"Node associations: {result.get('node_associations', {})}")
            return result
        else:
            print(f"Failed: {response.text}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def distribute_edge_assignments(nodes_config, edges_config, associations):
    """Tell each node which edge to connect to."""
    print("\nDistributing edge assignments...")

    # Build edge host mapping
    edge_hosts = {edge["id"]: (edge["host"], edge["port"]) for edge in edges_config}

    for node in nodes_config:
        node_id = node["id"]
        edge_id = associations.get(node_id)

        if edge_id and edge_id in edge_hosts:
            edge_host, edge_port = edge_hosts[edge_id]
            node_url = f"http://{node['host']}:{node['port']}"

            try:
                response = requests.post(
                    f"{node_url}/edge/assign",
                    json={"edge_host": edge_host, "edge_port": edge_port},
                    timeout=30,
                )
                if response.status_code == 200:
                    print(f"  {node_id} -> {edge_id}")
                else:
                    print(f"  {node_id}: Failed to assign")
            except Exception as e:
                print(f"  {node_id}: Error - {e}")


def start_training(cloud_url, num_rounds):
    """Start the hierarchical training process."""
    print("\n" + "=" * 60)
    print("Phase 3: Hierarchical Training")
    print("=" * 60)

    try:
        response = requests.post(
            f"{cloud_url}/training/start", json={"rounds": num_rounds}, timeout=30
        )
        if response.status_code == 200:
            print(f"Training started for {num_rounds} rounds")
            return True
        else:
            print(f"Failed to start training: {response.text}")
            return False
    except Exception as e:
        print(f"Error starting training: {e}")
        return False


def monitor_training(cloud_url, num_rounds, check_interval=30):
    """Monitor training progress."""
    print("\nMonitoring training progress...")

    while True:
        try:
            response = requests.get(f"{cloud_url}/status", timeout=10)
            if response.status_code == 200:
                status = response.json()
                current_round = status.get("current_round", 0)

                # Get latest accuracy
                try:
                    metrics_response = requests.get(
                        f"{cloud_url}/metrics/latest", timeout=10
                    )
                    if metrics_response.status_code == 200:
                        metrics = metrics_response.json()
                        accuracy = metrics.get("accuracy", 0)
                        comm_cost = metrics.get("cumulative_comm_cost", 0)
                        print(
                            f"Round {current_round}/{num_rounds}: "
                            f"Accuracy = {accuracy:.4f}, "
                            f"CommCost = {comm_cost:,} bytes"
                        )
                except:
                    print(f"Round {current_round}/{num_rounds}")

                if current_round >= num_rounds:
                    print("\nTraining complete!")
                    return True
        except Exception as e:
            print(f"Error checking status: {e}")

        time.sleep(check_interval)


def save_results(cloud_url, output_dir):
    """Fetch and save final results."""
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Fetch metrics
    try:
        response = requests.get(f"{cloud_url}/metrics/all", timeout=30)
        if response.status_code == 200:
            metrics = response.json()
            with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            print("Saved training_metrics.json")
    except Exception as e:
        print(f"Error saving metrics: {e}")

    # Fetch final model
    try:
        response = requests.get(f"{cloud_url}/model/global", timeout=60)
        if response.status_code == 200:
            with open(os.path.join(output_dir, "final_model.pt"), "wb") as f:
                f.write(response.content)
            print("Saved final_model.pt")
    except Exception as e:
        print(f"Error saving model: {e}")

    # Fetch configuration
    try:
        response = requests.get(f"{cloud_url}/config/edges", timeout=10)
        if response.status_code == 200:
            config = response.json()
            with open(os.path.join(output_dir, "edge_config.json"), "w") as f:
                json.dump(config, f, indent=2)
            print("Saved edge_config.json")
    except Exception as e:
        print(f"Error saving config: {e}")

    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="ShapeFL Main Orchestrator")
    parser.add_argument(
        "--cloud-host", type=str, default="localhost", help="Cloud server host"
    )
    parser.add_argument(
        "--cloud-port", type=int, default=5000, help="Cloud server port"
    )
    parser.add_argument(
        "--num-rounds", type=int, default=None, help="Number of training rounds"
    )
    parser.add_argument(
        "--skip-pretrain", action="store_true", help="Skip pre-training phase"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory for results"
    )

    args = parser.parse_args()

    cloud_url = f"http://{args.cloud_host}:{args.cloud_port}"
    num_rounds = args.num_rounds or TRAINING_CONFIG.kappa

    print("=" * 60)
    print("ShapeFL Orchestrator")
    print("=" * 60)
    print(f"Cloud server: {cloud_url}")
    print(f"Training rounds: {num_rounds}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Get configuration
    nodes_config = NETWORK_CONFIG.computing_nodes
    edges_config = NETWORK_CONFIG.edge_aggregators

    # Wait for all devices to register
    if not wait_for_registrations(cloud_url, len(nodes_config), len(edges_config)):
        print("Failed to get all registrations. Exiting.")
        return

    # Phase 1: Pre-training
    if not args.skip_pretrain:
        if not trigger_pretrain(nodes_config, cloud_url):
            print("Pre-training failed. Continuing anyway...")

    # Phase 2: Edge selection and node association
    algorithm_result = run_edge_selection(cloud_url)
    if algorithm_result:
        associations = algorithm_result.get("node_associations", {})
        distribute_edge_assignments(nodes_config, edges_config, associations)

    # Phase 3: Training
    if start_training(cloud_url, num_rounds):
        monitor_training(cloud_url, num_rounds)

    # Save results
    save_results(cloud_url, args.output_dir)

    print("\n" + "=" * 60)
    print("ShapeFL Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
