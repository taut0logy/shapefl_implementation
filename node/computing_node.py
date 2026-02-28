"""
Computing Node for ShapeFL
==========================
Computing node that performs local training on its data partition.

Responsibilities:
1. Load local data partition
2. Perform pre-training and send linear layer updates to cloud
3. Receive model from associated edge aggregator
4. Perform local training for kappa_e epochs
5. Send trained model back to edge aggregator

Based on Algorithm 3 (LocalUpdate function) from the ShapeFL paper.
"""

import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from flask import Flask, request, jsonify
from threading import Lock, Event, Thread
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRAINING_CONFIG
from models.factory import get_model
from data.data_loader import load_fmnist_data, load_partitions, get_node_dataloader
from utils.communication import (
    model_to_bytes,
    bytes_to_model,
    compress_model,
    decompress_model,
    json_to_state_dict,
)
from utils.aggregation import get_linear_layer_update


class ComputingNode:
    """
    Computing Node for ShapeFL Hierarchical Federated Learning.
    """

    def __init__(
        self,
        node_id: str,
        host: str = "0.0.0.0",
        port: int = 5002,
        cloud_host: str = "192.168.0.100",
        cloud_port: int = 5000,
        data_indices: List[int] = None,
        partitions_file: str = None,
    ):
        """
        Initialize the computing node.

        Args:
            node_id: Unique identifier for this node
            host: Host address to bind to
            port: Port to listen on
            cloud_host: Cloud server host address
            cloud_port: Cloud server port
            data_indices: List of data indices assigned to this node
            partitions_file: Path to partitions file (alternative to data_indices)
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.cloud_url = f"http://{cloud_host}:{cloud_port}"

        # Edge aggregator info (set after registration)
        self.edge_url = None
        self.edge_id = None

        # Device (Raspberry Pi uses CPU)
        self.device = torch.device("cpu")

        # Model
        self.model = get_model(
            model_name=TRAINING_CONFIG.model_name,
            num_classes=TRAINING_CONFIG.num_classes,
            device=self.device,
        )
        self.initial_model = None  # For computing updates

        # Data
        self.train_dataset, _ = load_fmnist_data()
        self.data_indices = data_indices

        # Load from partitions file if provided
        if partitions_file and os.path.exists(partitions_file):
            partitions = load_partitions(partitions_file)
            node_idx = int(node_id.split("_")[-1]) if "_" in node_id else int(node_id)
            if node_idx in partitions:
                self.data_indices = partitions[node_idx]

        self.data_loader = None
        if self.data_indices:
            self.data_loader = get_node_dataloader(
                self.train_dataset,
                self.data_indices,
                batch_size=TRAINING_CONFIG.batch_size,
            )
            print(f"Node {node_id}: {len(self.data_indices)} samples loaded")

        # Training state
        self.current_round = 0
        self.current_edge_epoch = 0
        self.training_complete = Event()

        # Synchronization
        self.lock = Lock()
        self.model_received = Event()

        # Flask app
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask API routes."""

        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify(
                {"status": "healthy", "role": "computing_node", "node_id": self.node_id}
            )

        @self.app.route("/model/update", methods=["POST"])
        def receive_model():
            """Receive updated model from edge aggregator."""
            data = request.get_json()

            with self.lock:
                if data.get("compressed", False):
                    model_bytes = decompress_model(data["model"])
                    self.model = bytes_to_model(model_bytes, self.model)
                else:
                    state_dict = json_to_state_dict(data["model"])
                    self.model.load_state_dict(state_dict)

                self.current_round = data.get("round", self.current_round)
                self.current_edge_epoch = data.get(
                    "edge_epoch", self.current_edge_epoch
                )

            self.model_received.set()
            print(
                f"Received model update: round {self.current_round}, "
                f"edge_epoch {self.current_edge_epoch}"
            )

            return jsonify({"status": "received"})

        @self.app.route("/train/start", methods=["POST"])
        def start_training():
            """Trigger local training."""
            data = request.get_json()
            epochs = data.get("epochs", TRAINING_CONFIG.kappa_e)

            # Start training in background
            Thread(target=self.train_local, args=(epochs,)).start()

            return jsonify({"status": "training_started", "epochs": epochs})

        @self.app.route("/status", methods=["GET"])
        def get_status():
            """Return current node status."""
            return jsonify(
                {
                    "node_id": self.node_id,
                    "current_round": self.current_round,
                    "current_edge_epoch": self.current_edge_epoch,
                    "data_size": len(self.data_indices) if self.data_indices else 0,
                    "edge_id": self.edge_id,
                }
            )

        @self.app.route("/pretrain/start", methods=["POST"])
        def start_pretrain():
            """Trigger pre-training phase (Algorithm 3, offline phase)."""
            data = request.get_json()
            epochs = data.get("epochs", TRAINING_CONFIG.kappa_p)

            # Start pre-training in background thread
            Thread(target=self.pretrain, args=(epochs,)).start()

            return jsonify(
                {
                    "status": "pretrain_started",
                    "epochs": epochs,
                    "node_id": self.node_id,
                }
            )

        @self.app.route("/edge/assign", methods=["POST"])
        def assign_edge():
            """Assign this node to an edge aggregator (after GoA algorithm)."""
            data = request.get_json()
            edge_host = data["edge_host"]
            edge_port = data["edge_port"]

            success = self.register_with_edge(edge_host, edge_port)

            return jsonify(
                {
                    "status": "assigned" if success else "failed",
                    "edge_id": self.edge_id,
                    "node_id": self.node_id,
                }
            )

    def register_with_cloud(self):
        """Register this node with the cloud server."""
        try:
            response = requests.post(
                f"{self.cloud_url}/register/node",
                json={"node_id": self.node_id, "host": self.host, "port": self.port},
                timeout=30,
            )
            if response.status_code == 200:
                print(f"Registered with cloud server at {self.cloud_url}")
                return True
            else:
                print(f"Failed to register with cloud: {response.text}")
                return False
        except Exception as e:
            print(f"Error registering with cloud: {e}")
            return False

    def register_with_edge(self, edge_host: str, edge_port: int):
        """Register this node with its assigned edge aggregator."""
        self.edge_url = f"http://{edge_host}:{edge_port}"

        try:
            response = requests.post(
                f"{self.edge_url}/register/node",
                json={"node_id": self.node_id, "host": self.host, "port": self.port},
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                self.edge_id = data.get("edge_id")
                print(
                    f"Registered with edge aggregator {self.edge_id} at {self.edge_url}"
                )
                return True
            else:
                print(f"Failed to register with edge: {response.text}")
                return False
        except Exception as e:
            print(f"Error registering with edge: {e}")
            return False

    def fetch_global_model(self):
        """Fetch the initial global model from cloud server."""
        try:
            response = requests.get(f"{self.cloud_url}/model/global", timeout=60)
            if response.status_code == 200:
                data = response.json()
                model_bytes = decompress_model(data["model"])
                self.model = bytes_to_model(model_bytes, self.model)
                self.current_round = data.get("round", 0)
                print("Fetched global model from cloud")
                return True
            else:
                print(f"Failed to fetch model: {response.text}")
                return False
        except Exception as e:
            print(f"Error fetching model: {e}")
            return False

    def fetch_edge_model(self):
        """Fetch the current model from edge aggregator."""
        if not self.edge_url:
            print("No edge aggregator assigned")
            return False

        try:
            response = requests.get(f"{self.edge_url}/model/current", timeout=60)
            if response.status_code == 200:
                data = response.json()
                model_bytes = decompress_model(data["model"])
                self.model = bytes_to_model(model_bytes, self.model)
                self.current_round = data.get("round", self.current_round)
                self.current_edge_epoch = data.get(
                    "edge_epoch", self.current_edge_epoch
                )
                print(f"Fetched model from edge: round {self.current_round}")
                return True
            else:
                print(f"Failed to fetch from edge: {response.text}")
                return False
        except Exception as e:
            print(f"Error fetching from edge: {e}")
            return False

    def train_local(self, epochs: int = 1):
        """
        Perform local training on node's data.

        This implements the LocalUpdate function from Algorithm 3.

        Args:
            epochs: Number of local epochs to train
        """
        if self.data_loader is None:
            print("No data loaded for training")
            return

        print(f"\nStarting local training: {epochs} epochs")

        # Save initial model for computing update
        self.initial_model = copy.deepcopy(self.model)

        self.model.train()
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=TRAINING_CONFIG.learning_rate)

        total_loss = 0.0
        num_batches = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / len(self.data_loader)
            print(f"  Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}")
            total_loss += epoch_loss

        avg_total_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Local training complete. Avg loss: {avg_total_loss:.4f}")

        # Submit update to edge
        self.submit_to_edge(avg_total_loss)

    def pretrain(self, epochs: int = None):
        """
        Perform pre-training and submit linear layer update to cloud.

        This is the offline pre-training phase described in Section IV-D.

        Args:
            epochs: Number of pre-training epochs (default: kappa_p)
        """
        if epochs is None:
            epochs = TRAINING_CONFIG.kappa_p

        print(f"\nStarting pre-training: {epochs} epochs")

        # Fetch initial model
        self.fetch_global_model()

        # Save initial model
        initial_model = copy.deepcopy(self.model)

        # Train locally
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=TRAINING_CONFIG.learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"  Pre-train epoch {epoch + 1}/{epochs}")

        print("Pre-training complete")

        # Compute linear layer update
        linear_update = get_linear_layer_update(self.model, initial_model)

        # Submit to cloud
        self.submit_pretrain_to_cloud(linear_update)

    def submit_pretrain_to_cloud(self, linear_update: torch.Tensor):
        """Submit pre-training results to cloud server."""
        try:
            response = requests.post(
                f"{self.cloud_url}/pretrain/submit",
                json={
                    "node_id": self.node_id,
                    "linear_update": linear_update.tolist(),
                    "data_size": len(self.data_indices) if self.data_indices else 0,
                },
                timeout=60,
            )
            if response.status_code == 200:
                print("Submitted pre-training results to cloud")
                return True
            else:
                print(f"Failed to submit pre-training: {response.text}")
                return False
        except Exception as e:
            print(f"Error submitting pre-training: {e}")
            return False

    def submit_to_edge(self, loss: float = 0.0):
        """Submit trained model to edge aggregator."""
        if not self.edge_url:
            print("No edge aggregator assigned")
            return False

        model_bytes = model_to_bytes(self.model)
        compressed = compress_model(model_bytes)

        try:
            response = requests.post(
                f"{self.edge_url}/node/submit",
                json={
                    "node_id": self.node_id,
                    "model": compressed,
                    "compressed": True,
                    "data_size": len(self.data_indices) if self.data_indices else 0,
                    "loss": loss,
                },
                timeout=120,
            )
            if response.status_code == 200:
                print(f"Submitted model to edge {self.edge_id}")
                return True
            else:
                print(f"Failed to submit to edge: {response.text}")
                return False
        except Exception as e:
            print(f"Error submitting to edge: {e}")
            return False

    def run_training_loop(self):
        """
        Run the node training loop.

        Per Algorithm 3 (LocalUpdate):
        For each edge epoch, the node:
            1. Receives model from edge aggregator
            2. Trains locally for kappa_e epochs
            3. Submits trained model back to edge
        """
        print(f"\nStarting training loop for node {self.node_id}")

        total_edge_epochs = TRAINING_CONFIG.kappa * TRAINING_CONFIG.kappa_c

        for epoch_num in range(1, total_edge_epochs + 1):
            # Wait for model from edge
            self.model_received.clear()
            self.model_received.wait(timeout=300)

            # Perform local training for kappa_e epochs (Algorithm 3, line 30)
            self.train_local(epochs=TRAINING_CONFIG.kappa_e)

        print(f"\nTraining complete for node {self.node_id}")
        self.training_complete.set()

    def run(self, edge_host: str = None, edge_port: int = None, training: bool = True):
        """
        Start the computing node.

        Args:
            edge_host: Edge aggregator host (optional, can be set later)
            edge_port: Edge aggregator port (optional)
            training: Whether to run the training loop
        """
        print("\n" + "=" * 60)
        print(f"Starting ShapeFL Computing Node: {self.node_id}")
        print("=" * 60)
        print(f"Host: {self.host}:{self.port}")
        print(f"Cloud Server: {self.cloud_url}")
        print(f"Data samples: {len(self.data_indices) if self.data_indices else 0}")
        print("=" * 60 + "\n")

        # Register with cloud
        self.register_with_cloud()

        # Register with edge if specified
        if edge_host and edge_port:
            self.register_with_edge(edge_host, edge_port)

        if training:
            # Start training loop in separate thread
            training_thread = Thread(target=self.run_training_loop)
            training_thread.daemon = True
            training_thread.start()

        # Start Flask server
        self.app.run(host=self.host, port=self.port, threaded=True)


def main():
    """Main entry point for the computing node."""
    import argparse

    parser = argparse.ArgumentParser(description="ShapeFL Computing Node")
    parser.add_argument("--node-id", type=str, required=True, help="Unique node ID")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host address to bind to"
    )
    parser.add_argument("--port", type=int, default=5002, help="Port to listen on")
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
        "--edge-host", type=str, default=None, help="Edge aggregator host address"
    )
    parser.add_argument(
        "--edge-port", type=int, default=None, help="Edge aggregator port"
    )
    parser.add_argument(
        "--partitions-file", type=str, default=None, help="Path to data partitions file"
    )
    parser.add_argument(
        "--no-training", action="store_true", help="Disable automatic training loop"
    )

    args = parser.parse_args()

    node = ComputingNode(
        node_id=args.node_id,
        host=args.host,
        port=args.port,
        cloud_host=args.cloud_host,
        cloud_port=args.cloud_port,
        partitions_file=args.partitions_file,
    )

    node.run(
        edge_host=args.edge_host,
        edge_port=args.edge_port,
        training=not args.no_training,
    )


if __name__ == "__main__":
    main()
