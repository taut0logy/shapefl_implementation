#!/usr/bin/env python3
"""
Visualize saved ShapeFL results
================================
Re-generate all plots and HTML reports from an existing JSON results file
without re-running the training.

Usage:
  python -m scripts.visualize_results results/comparison/comparison_results.json
  python -m scripts.visualize_results results/smoke_lenet5/simulation_results.json
  python -m scripts.visualize_results results/full_run      # auto-detect JSON
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualization import (
    visualize_simulation,
    visualize_comparison,
)


def find_json(path: str) -> str:
    """If path is a directory, find the first JSON file in it."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for name in ["comparison_results.json", "simulation_results.json"]:
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                return candidate
        # Fallback: first json file
        for f in os.listdir(path):
            if f.endswith(".json"):
                return os.path.join(path, f)
    raise FileNotFoundError(f"No JSON results found at: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ShapeFL results from a JSON file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", type=str, help="Path to results JSON file or directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory for plots (default: same as JSON)")
    args = parser.parse_args()

    json_path = find_json(args.path)
    output_dir = args.output_dir or os.path.dirname(json_path)

    print(f"Loading: {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    # Detect if it's a comparison or single simulation
    if "per_round_metrics" in data and "summary" in data:
        # Comparison results
        config = data.get("config", {})
        summary = data["summary"]
        all_metrics = data["per_round_metrics"]
        target = config.get("target_accuracy", 0.70)

        print(f"Detected: COMPARISON run ({len(summary)} strategies)")
        visualize_comparison(all_metrics, summary, config, target, output_dir)

    elif "metrics" in data:
        # Single simulation results
        config = data.get("config", {})
        metrics = data["metrics"]

        # Reconstruct edge_nodes from node_associations or direct field
        edge_nodes = {}
        if "node_associations" in data:
            # Invert: node -> edge  becomes  edge -> [nodes]
            inv = {}
            for node_id, edge_id in data["node_associations"].items():
                inv.setdefault(int(edge_id), []).append(int(node_id))
            edge_nodes = inv
        elif "edge_nodes" in data:
            edge_nodes = data["edge_nodes"]

        print("Detected: SINGLE SIMULATION")
        visualize_simulation(metrics, config, edge_nodes, output_dir)

    else:
        print("ERROR: Unrecognized JSON structure. Expected 'per_round_metrics' or 'metrics' key.")
        sys.exit(1)

    print(f"All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
