"""
Cloud Server for ShapeFL
========================
Main cloud server that coordinates the hierarchical federated learning process.

Responsibilities:
1. Initialize and distribute the global model
2. Coordinate the offline pre-training phase
3. Compute data distribution similarity matrix S_ij
4. Run LoS + GoA algorithms to select edges and associate nodes
5. Perform cloud aggregation after each kappa_c edge epochs
6. Track and report metrics

Based on Algorithm 3 from the ShapeFL paper.
"""

import os
import sys
import torch
import numpy as np
from flask import Flask, request, jsonify
from threading import Lock, Event
from typing import Dict, Any, Optional, Set
from datetime import datetime
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRAINING_CONFIG, PATH_CONFIG
from models.lenet5 import get_model
from data.data_loader import load_fmnist_data
from algorithms.los import run_los
from utils.communication import (
    model_to_bytes,
    bytes_to_model,
    compress_model,
    decompress_model
)
from utils.aggregation import federated_averaging
from utils.similarity import compute_similarity_matrix
from utils.metrics import MetricsTracker, compute_accuracy


class CloudServer:
    """
    Cloud Server for ShapeFL Hierarchical Federated Learning.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5000, use_gpu: bool = True):
        """
        Initialize the cloud server.

        Args:
            host: Host address to bind to
            port: Port to listen on
            use_gpu: Whether to use GPU for aggregation and evaluation
        """
        self.host = host
        self.port = port

        # Device setup
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        # Model
        self.global_model = get_model(
            model_name=TRAINING_CONFIG.model_name,
            num_classes=TRAINING_CONFIG.num_classes,
            device=self.device,
        )

        # Data for evaluation
        _, self.test_dataset = load_fmnist_data()
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=TRAINING_CONFIG.batch_size, shuffle=False
        )

        # Node management
        self.registered_nodes: Dict[str, Dict[str, Any]] = {}
        self.registered_edges: Dict[str, Dict[str, Any]] = {}

        # Pre-training data
        self.pretrain_updates: Dict[
            str, torch.Tensor
        ] = {}  # node_id -> linear layer update
        self.data_sizes: Dict[str, int] = {}  # node_id -> data size

        # Algorithm results
        self.similarity_matrix: Optional[np.ndarray] = None
        self.selected_edges: Set[str] = set()
        self.node_associations: Dict[str, str] = {}  # node_id -> edge_id

        # Training state
        self.current_round = 0
        self.edge_updates: Dict[str, Dict[str, Any]] = {}  # Received edge updates

        # Synchronization
        self.lock = Lock()
        self.pretrain_complete = Event()
        self.round_complete = Event()

        # Metrics
        PATH_CONFIG.ensure_dirs()
        self.metrics = MetricsTracker(save_dir=PATH_CONFIG.metrics_dir)

        # Flask app
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask API routes."""

        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "healthy", "role": "cloud_server"})

        @self.app.route("/register/node", methods=["POST"])
        def register_node():
            """Register a computing node."""
            data = request.get_json()
            node_id = data["node_id"]

            with self.lock:
                self.registered_nodes[node_id] = {
                    "host": data["host"],
                    "port": data["port"],
                    "registered_at": str(datetime.now()),
                }

            print(f"Registered node: {node_id}")
            return jsonify({"status": "registered", "node_id": node_id})

        @self.app.route("/register/edge", methods=["POST"])
        def register_edge():
            """Register an edge aggregator."""
            data = request.get_json()
            edge_id = data["edge_id"]

            with self.lock:
                self.registered_edges[edge_id] = {
                    "host": data["host"],
                    "port": data["port"],
                    "registered_at": str(datetime.now()),
                }

            print(f"Registered edge: {edge_id}")
            return jsonify({"status": "registered", "edge_id": edge_id})

        @self.app.route("/model/global", methods=["GET"])
        def get_global_model():
            """Return the current global model."""
            model_bytes = model_to_bytes(self.global_model)
            compressed = compress_model(model_bytes)

            return jsonify(
                {"model": compressed, "round": self.current_round, "compressed": True}
            )

        @self.app.route("/pretrain/submit", methods=["POST"])
        def submit_pretrain():
            """Receive pre-training results from a node."""
            data = request.get_json()
            node_id = data["node_id"]

            # Extract linear layer update
            update_data = data["linear_update"]
            linear_update = torch.tensor(update_data)

            data_size = data["data_size"]

            with self.lock:
                self.pretrain_updates[node_id] = linear_update
                self.data_sizes[node_id] = data_size

            print(f"Received pre-train from {node_id}: {data_size} samples")

            # Check if all nodes have submitted
            expected_nodes = len(self.registered_nodes)
            received = len(self.pretrain_updates)

            if received >= expected_nodes:
                self.pretrain_complete.set()

            return jsonify(
                {"status": "received", "nodes_remaining": expected_nodes - received}
            )

        @self.app.route("/edge/submit", methods=["POST"])
        def submit_edge_update():
            """Receive aggregated model update from an edge aggregator."""
            data = request.get_json()
            edge_id = data["edge_id"]

            # Decompress and deserialize model
            model_data = data["model"]
            if data.get("compressed", False):
                model_bytes = decompress_model(model_data)
            else:
                model_bytes = None

            with self.lock:
                self.edge_updates[edge_id] = {
                    "model_data": model_data,
                    "compressed": data.get("compressed", False),
                    "data_size": data["total_data_size"],
                    "num_nodes": data["num_nodes"],
                }

            print(
                f"Received update from edge {edge_id}: {data['num_nodes']} nodes, "
                f"{data['total_data_size']} samples"
            )

            # Check if all edges have submitted
            expected_edges = len(self.selected_edges)
            received = len(self.edge_updates)

            if received >= expected_edges:
                self.round_complete.set()

            return jsonify(
                {"status": "received", "edges_remaining": expected_edges - received}
            )

        @self.app.route("/config/edges", methods=["GET"])
        def get_edge_config():
            """Return edge selection and node association results."""
            return jsonify(
                {
                    "selected_edges": list(self.selected_edges),
                    "node_associations": self.node_associations,
                }
            )

        @self.app.route("/status", methods=["GET"])
        def get_status():
            """Return current training status."""
            return jsonify(
                {
                    "current_round": self.current_round,
                    "registered_nodes": len(self.registered_nodes),
                    "registered_edges": len(self.registered_edges),
                    "selected_edges": list(self.selected_edges),
                    "pretrain_complete": self.pretrain_complete.is_set(),
                }
            )

        @self.app.route("/algorithms/run", methods=["POST"])
        def run_algorithms_endpoint():
            """Execute LoS + GoA algorithms for edge selection and node association."""
            self.run_algorithms()
            return jsonify(
                {
                    "status": "completed",
                    "selected_edges": list(self.selected_edges),
                    "node_associations": self.node_associations,
                }
            )

        @self.app.route("/training/start", methods=["POST"])
        def start_training():
            """Start the hierarchical training loop."""
            data = request.get_json()
            rounds = data.get("rounds", TRAINING_CONFIG.kappa)

            # Start training in background thread
            from threading import Thread

            Thread(target=self.run_training_loop, args=(rounds,)).start()

            return jsonify({"status": "training_started", "rounds": rounds})

        @self.app.route("/metrics/latest", methods=["GET"])
        def get_latest_metrics():
            """Get the most recent training metrics."""
            return jsonify(self.metrics.get_latest())

        @self.app.route("/metrics/all", methods=["GET"])
        def get_all_metrics():
            """Get all training metrics history."""
            return jsonify(self.metrics.get_all())

    def compute_communication_costs(self) -> tuple:
        """
        Compute communication costs for the LAN setup.

        For a local network, we use a simplified model based on bytes transmitted.

        Returns:
            Tuple of (c_ne dict, c_ec dict)
        """
        # In LAN, costs are relatively uniform
        # We can use node indices as a proxy for "distance"
        c_ne = {}  # (node, edge) -> cost
        c_ec = {}  # edge -> cost to cloud

        node_ids = list(self.registered_nodes.keys())
        edge_ids = list(self.registered_edges.keys())

        # Node to edge costs (based on model size)
        model_size = len(model_to_bytes(self.global_model))

        for n_idx, node_id in enumerate(node_ids):
            for e_idx, edge_id in enumerate(edge_ids):
                # Simplified cost: model_size * (1 + small variation)
                variation = 0.1 * abs(n_idx - e_idx) / max(len(node_ids), 1)
                c_ne[(n_idx, e_idx)] = model_size * (1 + variation)

        # Edge to cloud costs
        for e_idx, edge_id in enumerate(edge_ids):
            c_ec[e_idx] = model_size * 1.5  # Slightly higher for cloud

        return c_ne, c_ec

    def run_algorithms(self):
        """
        Run LoS and GoA algorithms to select edges and associate nodes.
        """
        print("\n" + "=" * 60)
        print("Running Edge Selection and Node Association Algorithms")
        print("=" * 60)

        # Build the similarity matrix from pre-training updates
        node_ids = sorted(self.pretrain_updates.keys())
        node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        update_dict = {
            node_id_to_idx[nid]: update for nid, update in self.pretrain_updates.items()
        }

        self.similarity_matrix = compute_similarity_matrix(update_dict)
        print(f"Computed similarity matrix: {self.similarity_matrix.shape}")

        # Compute communication costs
        c_ne, c_ec = self.compute_communication_costs()

        # Build data sizes dict with indices
        data_sizes_idx = {
            node_id_to_idx[nid]: size for nid, size in self.data_sizes.items()
        }

        # Candidate edges (all registered edges, mapped to indices)
        edge_ids = sorted(self.registered_edges.keys())

        # For simplicity, use nodes that are also edges as candidates
        # In practice, any node can be an edge aggregator
        candidate_edge_indices = list(range(len(edge_ids)))
        all_node_indices = list(range(len(node_ids)))

        # Run LoS algorithm
        print("\nRunning LoS (Local Search) for edge selection...")
        los_result = run_los(
            candidate_edges=candidate_edge_indices,
            all_nodes=all_node_indices,
            communication_costs_ne=c_ne,
            communication_costs_ec=c_ec,
            similarity_matrix=self.similarity_matrix,
            data_sizes=data_sizes_idx,
            kappa_c=TRAINING_CONFIG.kappa_c,
            gamma=TRAINING_CONFIG.gamma,
            B_e=TRAINING_CONFIG.B_e,
            T_max=TRAINING_CONFIG.T_max,
        )

        # Map indices back to IDs
        idx_to_node_id = {idx: nid for nid, idx in node_id_to_idx.items()}
        idx_to_edge_id = {idx: eid for idx, eid in enumerate(edge_ids)}

        self.selected_edges = {idx_to_edge_id[idx] for idx in los_result.selected_edges}

        # Get node associations
        associations = los_result.node_associations.associations
        self.node_associations = {}
        for node_idx, edge_idx in associations.items():
            if node_idx in idx_to_node_id:
                node_id = idx_to_node_id[node_idx]
                edge_id = idx_to_edge_id.get(
                    edge_idx, edge_ids[edge_idx % len(edge_ids)]
                )
                self.node_associations[node_id] = edge_id

        print(f"\nSelected edges: {self.selected_edges}")
        print(f"Node associations: {self.node_associations}")

    def aggregate_edge_updates(self):
        """
        Perform cloud aggregation of edge model updates.
        """
        print("\nPerforming cloud aggregation...")

        # Collect edge models and weights
        edge_models = []
        weights = []

        for edge_id, update_info in self.edge_updates.items():
            # Deserialize model
            model = get_model(device="cpu")
            model_bytes = decompress_model(update_info["model_data"])
            model = bytes_to_model(model_bytes, model)

            edge_models.append(model)
            weights.append(update_info["data_size"])

        # Perform weighted averaging
        aggregated = federated_averaging(edge_models, weights)

        # Update global model
        self.global_model.load_state_dict(aggregated.state_dict())
        self.global_model.to(self.device)

        # Clear edge updates for next round
        self.edge_updates.clear()
        self.round_complete.clear()

    def evaluate(self) -> float:
        """
        Evaluate the global model on the test set.

        Returns:
            Test accuracy
        """
        return compute_accuracy(self.global_model, self.test_loader, self.device)

    def run_training_loop(self, num_rounds: int):
        """
        Run the hierarchical federated learning training loop.

        This implements Algorithm 3 from the ShapeFL paper:
        1. Wait for edge updates after kappa_c edge epochs
        2. Perform cloud aggregation
        3. Evaluate and record metrics
        4. Repeat for specified rounds

        Args:
            num_rounds: Number of cloud aggregation rounds
        """
        from utils.metrics import compute_communication_cost

        print("\n" + "=" * 60)
        print(f"Starting Training: {num_rounds} rounds")
        print("=" * 60)

        for round_num in range(1, num_rounds + 1):
            self.current_round = round_num
            print(f"\n--- Cloud Aggregation Round {round_num}/{num_rounds} ---")

            # Wait for all selected edges to submit their updates
            print("Waiting for edge updates...")
            self.round_complete.wait(timeout=600)  # 10 minute timeout

            if len(self.edge_updates) < len(self.selected_edges):
                print(
                    f"Warning: Only received {len(self.edge_updates)}/"
                    f"{len(self.selected_edges)} edge updates"
                )

            # Perform cloud aggregation
            self.aggregate_edge_updates()

            # Evaluate the global model
            accuracy = self.evaluate()
            print(f"Round {round_num} Accuracy: {accuracy:.4f}")

            # Compute communication cost for this round
            comm_cost = compute_communication_cost(
                model=self.global_model,
                num_nodes=len(self.registered_nodes),
                num_edges=len(self.selected_edges),
                kappa_e=TRAINING_CONFIG.kappa_e,
                kappa_c=TRAINING_CONFIG.kappa_c,
            )

            # Record metrics
            self.metrics.record(
                round_num=round_num,
                accuracy=accuracy,
                loss=0.0,  # Loss is tracked at node level
                communication_cost=comm_cost,
            )

        # Save final metrics
        self.metrics.save()
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Final Accuracy: {accuracy:.4f}")
        print("=" * 60)

    def run(self):
        """
        Start the cloud server.
        """
        print("\n" + "=" * 60)
        print("Starting ShapeFL Cloud Server")
        print("=" * 60)
        print(f"Host: {self.host}:{self.port}")
        print(f"Device: {self.device}")
        print("Waiting for nodes and edges to register...")
        print("=" * 60 + "\n")

        self.app.run(host=self.host, port=self.port, threaded=True)


def main():
    """Main entry point for the cloud server."""
    import argparse

    parser = argparse.ArgumentParser(description="ShapeFL Cloud Server")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host address to bind to"
    )
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")

    args = parser.parse_args()

    server = CloudServer(host=args.host, port=args.port, use_gpu=not args.no_gpu)

    server.run()


if __name__ == "__main__":
    main()
