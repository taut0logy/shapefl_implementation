"""
Edge Aggregator for ShapeFL
===========================
Edge aggregator that performs intermediate aggregation between
computing nodes and the cloud server.

Responsibilities:
1. Receive global model from cloud server
2. Distribute model to associated computing nodes
3. Collect and aggregate node updates after local training
4. Send aggregated updates to cloud server after kappa_e edge epochs

Based on the HFL framework described in the ShapeFL paper.
"""

import os
import sys
import torch
import requests
from flask import Flask, request, jsonify
from threading import Lock, Event, Thread
from typing import Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRAINING_CONFIG
from models.lenet5 import get_model
from utils.communication import (
    model_to_bytes,
    bytes_to_model,
    compress_model,
    decompress_model,
    json_to_state_dict,
)
from utils.aggregation import federated_averaging


class EdgeAggregator:
    """
    Edge Aggregator for ShapeFL Hierarchical Federated Learning.
    """

    def __init__(
        self,
        edge_id: str,
        host: str = "0.0.0.0",
        port: int = 5001,
        cloud_host: str = "192.168.0.100",
        cloud_port: int = 5000,
    ):
        """
        Initialize the edge aggregator.

        Args:
            edge_id: Unique identifier for this edge aggregator
            host: Host address to bind to
            port: Port to listen on
            cloud_host: Cloud server host address
            cloud_port: Cloud server port
        """
        self.edge_id = edge_id
        self.host = host
        self.port = port
        self.cloud_url = f"http://{cloud_host}:{cloud_port}"

        # Device (Raspberry Pi typically uses CPU)
        self.device = torch.device("cpu")

        # Model
        self.current_model = get_model(
            model_name=TRAINING_CONFIG.model_name,
            num_classes=TRAINING_CONFIG.num_classes,
            device=self.device,
        )

        # Associated nodes
        self.associated_nodes: Dict[str, Dict[str, Any]] = {}

        # Training state
        self.current_round = 0
        self.current_edge_epoch = 0
        self.node_updates: Dict[str, Dict[str, Any]] = {}
        self.last_aggregation_data_size = 0  # Track data size from last aggregation

        # Synchronization
        self.lock = Lock()
        self.epoch_complete = Event()

        # Flask app
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask API routes."""

        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify(
                {
                    "status": "healthy",
                    "role": "edge_aggregator",
                    "edge_id": self.edge_id,
                }
            )

        @self.app.route("/register/node", methods=["POST"])
        def register_node():
            """Register a computing node with this edge."""
            data = request.get_json()
            node_id = data["node_id"]

            # Use the actual IP from the request, not the host sent by the node
            # (the node sends 0.0.0.0 which is its bind address, not reachable)
            actual_host = request.remote_addr
            node_port = data["port"]

            with self.lock:
                self.associated_nodes[node_id] = {
                    "host": actual_host,
                    "port": node_port,
                    "registered_at": str(datetime.now()),
                }

            print(
                f"Node {node_id} registered with edge {self.edge_id} at {actual_host}:{node_port}"
            )
            return jsonify({"status": "registered", "edge_id": self.edge_id})

        @self.app.route("/model/current", methods=["GET"])
        def get_current_model():
            """Return the current edge model to nodes."""
            with self.lock:
                model_bytes = model_to_bytes(self.current_model)
                compressed = compress_model(model_bytes)

            return jsonify(
                {
                    "model": compressed,
                    "round": self.current_round,
                    "edge_epoch": self.current_edge_epoch,
                    "compressed": True,
                }
            )

        @self.app.route("/node/submit", methods=["POST"])
        def submit_node_update():
            """Receive model update from a computing node."""
            data = request.get_json()
            node_id = data["node_id"]

            with self.lock:
                self.node_updates[node_id] = {
                    "model_data": data["model"],
                    "compressed": data.get("compressed", False),
                    "data_size": data["data_size"],
                    "loss": data.get("loss", 0.0),
                }

            print(f"Received update from node {node_id}: {data['data_size']} samples")

            # Check if all nodes have submitted
            expected_nodes = len(self.associated_nodes)
            received = len(self.node_updates)

            if received >= expected_nodes:
                self.epoch_complete.set()

            return jsonify(
                {"status": "received", "nodes_remaining": expected_nodes - received}
            )

        @self.app.route("/status", methods=["GET"])
        def get_status():
            """Return current edge status."""
            return jsonify(
                {
                    "edge_id": self.edge_id,
                    "current_round": self.current_round,
                    "current_edge_epoch": self.current_edge_epoch,
                    "associated_nodes": len(self.associated_nodes),
                    "updates_received": len(self.node_updates),
                }
            )

    def register_with_cloud(self):
        """Register this edge aggregator with the cloud server."""
        try:
            response = requests.post(
                f"{self.cloud_url}/register/edge",
                json={"edge_id": self.edge_id, "host": self.host, "port": self.port},
                timeout=30,
            )
            if response.status_code == 200:
                print(f"Registered with cloud server at {self.cloud_url}")
                return True
            else:
                print(f"Failed to register: {response.text}")
                return False
        except Exception as e:
            print(f"Error registering with cloud: {e}")
            return False

    def fetch_global_model(self):
        """Fetch the latest global model from the cloud server."""
        try:
            response = requests.get(f"{self.cloud_url}/model/global", timeout=60)
            if response.status_code == 200:
                data = response.json()
                model_bytes = decompress_model(data["model"])
                self.current_model = bytes_to_model(model_bytes, self.current_model)
                self.current_round = data["round"]
                print(f"Fetched global model for round {self.current_round}")
                return True
            else:
                print(f"Failed to fetch model: {response.text}")
                return False
        except Exception as e:
            print(f"Error fetching model: {e}")
            return False

    def broadcast_to_nodes(self):
        """Broadcast current model to all associated nodes."""
        model_bytes = model_to_bytes(self.current_model)
        compressed = compress_model(model_bytes)

        success_count = 0
        for node_id, node_info in self.associated_nodes.items():
            try:
                node_url = f"http://{node_info['host']}:{node_info['port']}"
                response = requests.post(
                    f"{node_url}/model/update",
                    json={
                        "model": compressed,
                        "compressed": True,
                        "round": self.current_round,
                        "edge_epoch": self.current_edge_epoch,
                    },
                    timeout=60,
                )
                if response.status_code == 200:
                    success_count += 1
            except Exception as e:
                print(f"Error broadcasting to node {node_id}: {e}")

        print(f"Broadcast to {success_count}/{len(self.associated_nodes)} nodes")
        return success_count

    def aggregate_node_updates(self):
        """Aggregate updates from all associated nodes."""
        print(f"Aggregating updates from {len(self.node_updates)} nodes...")

        if len(self.node_updates) == 0:
            print("No updates to aggregate")
            return

        # Collect models and weights
        node_models = []
        weights = []
        total_data_size = 0

        for node_id, update_info in self.node_updates.items():
            # Deserialize model
            model = get_model(device="cpu")

            if update_info["compressed"]:
                model_bytes = decompress_model(update_info["model_data"])
                model = bytes_to_model(model_bytes, model)
            else:
                state_dict = json_to_state_dict(update_info["model_data"])
                model.load_state_dict(state_dict)

            node_models.append(model)
            weights.append(update_info["data_size"])
            total_data_size += update_info["data_size"]

        # Perform weighted averaging
        aggregated = federated_averaging(node_models, weights)

        # Update current model
        with self.lock:
            self.current_model.load_state_dict(aggregated.state_dict())
            self.last_aggregation_data_size = (
                total_data_size  # Track for submit_to_cloud
            )
            self.node_updates.clear()
            self.epoch_complete.clear()

        print(
            f"Aggregated {len(node_models)} node updates, "
            f"total data: {total_data_size} samples"
        )

    def submit_to_cloud(self):
        """Submit aggregated model to the cloud server."""
        model_bytes = model_to_bytes(self.current_model)
        compressed = compress_model(model_bytes)

        # Use the tracked data size from the last aggregation
        total_data_size = self.last_aggregation_data_size

        try:
            response = requests.post(
                f"{self.cloud_url}/edge/submit",
                json={
                    "edge_id": self.edge_id,
                    "model": compressed,
                    "compressed": True,
                    "total_data_size": total_data_size,
                    "num_nodes": len(self.associated_nodes),
                },
                timeout=120,
            )
            if response.status_code == 200:
                print("Submitted aggregated model to cloud")
                return True
            else:
                print(f"Failed to submit to cloud: {response.text}")
                return False
        except Exception as e:
            print(f"Error submitting to cloud: {e}")
            return False

    def run_training_loop(self):
        """
        Run the edge aggregation training loop.

        Per Algorithm 3 from the paper:
        For each cloud aggregation round:
            For each edge epoch (1 to kappa_c):
                1. Broadcast model to nodes
                2. Nodes train locally for kappa_e epochs
                3. Collect node updates
                4. Aggregate node updates (FedAvg)
            Submit aggregated model to cloud
        """
        print(f"\nStarting training loop for edge {self.edge_id}")

        kappa_c = TRAINING_CONFIG.kappa_c  # Edge epochs per cloud round

        for round_num in range(1, TRAINING_CONFIG.kappa + 1):
            self.current_round = round_num
            print(f"\n{'=' * 50}")
            print(f"Round {round_num}")
            print(f"{'=' * 50}")

            # Fetch latest global model from cloud
            self.fetch_global_model()

            # Run kappa_c edge aggregation epochs (Algorithm 3, line 16)
            for edge_epoch in range(1, kappa_c + 1):
                self.current_edge_epoch = edge_epoch
                print(f"\n  Edge epoch {edge_epoch}/{kappa_c}")

                # Broadcast to nodes
                self.broadcast_to_nodes()

                # Wait for all nodes to submit updates
                print("  Waiting for node updates...")
                self.epoch_complete.wait(timeout=300)

                if len(self.node_updates) < len(self.associated_nodes):
                    print(
                        f"  Warning: Only received {len(self.node_updates)}/"
                        f"{len(self.associated_nodes)} updates"
                    )

                # Aggregate node updates
                self.aggregate_node_updates()

            # Submit to cloud after completing kappa_c edge epochs
            # Per Algorithm 3: Edge submits after kappa_c edge aggregations
            print(f"\nSubmitting to cloud after round {round_num}")
            self.submit_to_cloud()

        print(f"\nTraining complete for edge {self.edge_id}")

    def run(self, training: bool = True):
        """
        Start the edge aggregator.

        Args:
            training: Whether to run the training loop after starting
        """
        print("\n" + "=" * 60)
        print(f"Starting ShapeFL Edge Aggregator: {self.edge_id}")
        print("=" * 60)
        print(f"Host: {self.host}:{self.port}")
        print(f"Cloud Server: {self.cloud_url}")
        print("=" * 60 + "\n")

        # Register with cloud
        self.register_with_cloud()

        if training:
            # Start training in a separate thread
            training_thread = Thread(target=self.run_training_loop)
            training_thread.daemon = True
            training_thread.start()

        # Start Flask server
        self.app.run(host=self.host, port=self.port, threaded=True)


def main():
    """Main entry point for the edge aggregator."""
    import argparse

    parser = argparse.ArgumentParser(description="ShapeFL Edge Aggregator")
    parser.add_argument(
        "--edge-id", type=str, required=True, help="Unique edge aggregator ID"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host address to bind to"
    )
    parser.add_argument("--port", type=int, default=5001, help="Port to listen on")
    parser.add_argument(
        "--cloud-host",
        type=str,
        default="192.168.0.100",
        help="Cloud server host address",
    )
    parser.add_argument(
        "--cloud-port", type=int, default=5000, help="Cloud server port"
    )
    parser.add_argument(
        "--no-training", action="store_true", help="Disable automatic training loop"
    )

    args = parser.parse_args()

    aggregator = EdgeAggregator(
        edge_id=args.edge_id,
        host=args.host,
        port=args.port,
        cloud_host=args.cloud_host,
        cloud_port=args.cloud_port,
    )

    aggregator.run(training=not args.no_training)


if __name__ == "__main__":
    main()
