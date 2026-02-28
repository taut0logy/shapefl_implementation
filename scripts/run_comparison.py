#!/usr/bin/env python3
"""
ShapeFL Baseline Comparison Experiment
======================================
Runs the same hierarchical training under 4 different edge-selection /
node-association strategies and compares:

    (a) accuracy  vs.  cumulative communication cost   (paper Fig. 11)
    (b) total communication cost to reach a target accuracy (paper Table II)

Strategies (from Section V-A):

  1. **ShapeFL**     – full LoS + GoA (Algorithm 2 + Algorithm 1)
  2. **Cost First**  – minimises per-round communication cost only (γ=0)
  3. **Data First**  – maximises data distribution diversity only (γ→∞)
  4. **Random**      – randomly pick edges, randomly assign nodes
  5. **FedAvg**      – flat cloud-only FL, no edge hierarchy

Usage:
    python scripts/run_comparison.py [--kappa 50] [--target-accuracy 0.70]

Paper reference: Section V, Figures 10-12, Table II
"""

import os
import sys
import copy
import math
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRAINING_CONFIG, PATH_CONFIG
from models.factory import get_model, get_model_size
from data.data_loader import (
    load_data,
    create_non_iid_partitions,
    get_node_dataloader,
    DATASET_INFO,
)
from algorithms.goa import run_goa
from algorithms.los import run_los
from utils.similarity import compute_similarity_matrix
from utils.aggregation import federated_averaging, get_linear_layer_update


# ============================================================================
#  Helpers  (shared across strategies)
# ============================================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def local_update(model, data_loader, epochs, lr, device):
    """LocalUpdate (Algorithm 3, lines 27-33)."""
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    total_loss, num_batches = 0.0, 0
    for _ in range(epochs):
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
    return model, len(data_loader.dataset), avg_loss


def evaluate_model(model, test_loader, device):
    model.eval()
    model.to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total if total > 0 else 0.0


def weighted_average_models(models, weights, device="cpu"):
    return federated_averaging(models, weights)


def generate_communication_costs(num_nodes, model_size_bytes):
    """Simulated costs (GB units) – matches run_local_simulation.py."""
    c_ne, c_ec = {}, {}
    model_size_gb = model_size_bytes / (1024 ** 3)
    np.random.seed(123)  # Fixed positions across strategies
    positions = np.random.rand(num_nodes, 2) * 1000  # km
    cloud_pos = np.array([500.0, 3500.0])
    for n in range(num_nodes):
        for e in range(num_nodes):
            if n == e:
                c_ne[(n, e)] = 0.0
            else:
                dist = np.linalg.norm(positions[n] - positions[e])
                c_ne[(n, e)] = 0.002 * dist * model_size_gb
        c_ec[n] = 0.02 * np.linalg.norm(positions[n] - cloud_pos) * model_size_gb
    return c_ne, c_ec


def compute_per_round_cost_gb(c_ne, c_ec, edges, edge_nodes, kappa_c, is_flat=False):
    """
    Per cloud-round communication cost in GB using the paper's distance-weighted
    cost model (Eq. 6-7).

    HFL (bidirectional, per cloud round):
        Node <-> Edge:  2 * κ_c * Σ_e Σ_{n∈N_e} c_ne(n,e)
        Edge <-> Cloud: 2 * Σ_e c_ec(e)

    FedAvg (flat, per cloud round):
        Node <-> Cloud: 2 * Σ_n c_ec(n)   (backbone rate, each node talks to cloud)

    Returns cost in GB.
    """
    if is_flat:
        # All nodes talk directly to cloud — c_ec[n] gives cost from node n to cloud
        all_nodes = edge_nodes[-1]  # virtual edge = all nodes
        return 2.0 * sum(c_ec[n] for n in all_nodes)
    else:
        node_edge = sum(c_ne[(n, e)] for e in edges for n in edge_nodes[e])
        edge_cloud = sum(c_ec[e] for e in edges)
        return 2.0 * kappa_c * node_edge + 2.0 * edge_cloud


# ============================================================================
#  Strategy planners – decide edges & associations BEFORE training starts
# ============================================================================

def plan_shapefl(args, c_ne, c_ec, S, data_sizes):
    """Full ShapeFL: LoS + GoA (Algorithms 1+2)."""
    los_result = run_los(
        candidate_edges=list(range(args.num_nodes)),
        all_nodes=list(range(args.num_nodes)),
        communication_costs_ne=c_ne,
        communication_costs_ec=c_ec,
        similarity_matrix=S,
        data_sizes=data_sizes,
        kappa_c=args.kappa_c,
        gamma=args.gamma,
        B_e=args.B_e,
        T_max=args.T_max,
    )
    edges = {int(e) for e in los_result.selected_edges}
    edge_nodes = {int(e): [int(n) for n in ns] for e, ns in los_result.node_associations.edge_nodes.items()}
    edge_data_sizes = {int(e): v for e, v in los_result.node_associations.edge_data_sizes.items()}
    return edges, edge_nodes, edge_data_sizes


def plan_cost_first(args, c_ne, c_ec, S, data_sizes):
    """
    Cost First baseline: minimise per-round comm cost, ignore data diversity.
    Equivalent to ShapeFL with γ=0 (paper Section V-A).
    """
    los_result = run_los(
        candidate_edges=list(range(args.num_nodes)),
        all_nodes=list(range(args.num_nodes)),
        communication_costs_ne=c_ne,
        communication_costs_ec=c_ec,
        similarity_matrix=S,
        data_sizes=data_sizes,
        kappa_c=args.kappa_c,
        gamma=0.0,  # <--- no diversity term
        B_e=args.B_e,
        T_max=args.T_max,
    )
    edges = {int(e) for e in los_result.selected_edges}
    edge_nodes = {int(e): [int(n) for n in ns] for e, ns in los_result.node_associations.edge_nodes.items()}
    edge_data_sizes = {int(e): v for e, v in los_result.node_associations.edge_data_sizes.items()}
    return edges, edge_nodes, edge_data_sizes


def plan_data_first(args, c_ne, c_ec, S, data_sizes):
    """
    Data First baseline: maximise data distribution diversity at edges,
    ignoring communication cost (paper Section V-A).

    Equivalent to ShapeFL with γ → ∞.  We implement this by setting γ to
    a very large value (1e8) so the diversity term completely dominates.
    """
    los_result = run_los(
        candidate_edges=list(range(args.num_nodes)),
        all_nodes=list(range(args.num_nodes)),
        communication_costs_ne=c_ne,
        communication_costs_ec=c_ec,
        similarity_matrix=S,
        data_sizes=data_sizes,
        kappa_c=args.kappa_c,
        gamma=1e8,  # <--- diversity dominates
        B_e=args.B_e,
        T_max=args.T_max,
    )
    edges = {int(e) for e in los_result.selected_edges}
    edge_nodes = {int(e): [int(n) for n in ns] for e, ns in los_result.node_associations.edge_nodes.items()}
    edge_data_sizes = {int(e): v for e, v in los_result.node_associations.edge_data_sizes.items()}
    return edges, edge_nodes, edge_data_sizes


def plan_random(args, data_sizes):
    """
    Random baseline: randomly select B_e-constrained edges, randomly assign
    nodes to their nearest edge (paper Section V-A, "Random" baseline).
    """
    N = args.num_nodes
    # Pick a random subset as edges (same number as ShapeFL would typically select)
    num_edges = max(2, math.ceil(N / args.B_e))
    np.random.seed(999)  # reproducible but different from position seed
    edge_list = sorted(np.random.choice(N, num_edges, replace=False).tolist())
    edges = set(edge_list)

    # Random round-robin assignment respecting B_e capacity
    edge_nodes = {e: [] for e in edges}
    edge_data_sizes = {e: 0 for e in edges}
    nodes_shuffled = list(range(N))
    np.random.shuffle(nodes_shuffled)
    edge_cycle = list(edges)
    idx = 0
    for n in nodes_shuffled:
        # Find an edge with remaining capacity
        for attempt in range(len(edge_cycle)):
            e = edge_cycle[(idx + attempt) % len(edge_cycle)]
            if len(edge_nodes[e]) < args.B_e:
                edge_nodes[e].append(n)
                edge_data_sizes[e] += data_sizes[n]
                idx = (idx + attempt + 1) % len(edge_cycle)
                break
    return edges, edge_nodes, edge_data_sizes


def plan_fedavg_flat(args, data_sizes):
    """
    Flat FedAvg: no hierarchy, all nodes communicate directly with cloud.
    Modelled as a single "virtual edge" equal to the cloud so there is
    no edge layer.  Returns a structure compatible with the HFL loop.
    """
    # We use a sentinel edge_id = -1 to indicate the cloud acts as sole aggregator
    edges = {-1}
    edge_nodes = {-1: list(range(args.num_nodes))}
    edge_data_sizes = {-1: sum(data_sizes.values())}
    return edges, edge_nodes, edge_data_sizes


# ============================================================================
#  Training loop – one loop shared by all strategies
# ============================================================================

def run_strategy(
    strategy_name,
    args,
    global_model_state,
    node_loaders,
    data_sizes,
    test_loader,
    edges,
    edge_nodes,
    edge_data_sizes,
    per_round_cost_gb,
    device,
    is_flat=False,
):
    """
    Run the HFL training loop for a given strategy and collect metrics.

    Returns dict with per-round accuracy, loss, per_round_cost, cumulative_cost.
    Cost units are in GB (distance-weighted, paper Eq. 6-7).
    """
    print(f"\n{'='*60}")
    print(f"  STRATEGY: {strategy_name}")
    print(f"{'='*60}")
    rc_gb = per_round_cost_gb
    print(f"  Edges: {sorted(edges)}")
    for e in sorted(edges):
        print(f"    Edge {e}: nodes {sorted(edge_nodes[e])}")
    print(f"  Per-round comm cost: {rc_gb:.6f} GB ({rc_gb * 1024:.2f} MB)")

    # Build model from the SAME initial weights
    num_classes = DATASET_INFO[args.dataset]["num_classes"]
    input_channels = DATASET_INFO[args.dataset]["input_channels"]
    global_model = get_model(args.model, num_classes, input_channels, device)
    global_model.load_state_dict(copy.deepcopy(global_model_state))

    metrics = {
        "round": [],
        "accuracy": [],
        "loss": [],
        "per_round_cost": [],
        "cumulative_cost": [],
    }
    cumulative = 0

    if is_flat:
        # ---- Flat FedAvg: each cloud round = 1 round of all nodes -> cloud ----
        for rnd in range(1, args.kappa + 1):
            node_models = {}
            node_data_sz = {}
            round_losses = []
            # Each "cloud round" = kappa_c * kappa_e local epochs (same total work)
            flat_epochs = args.kappa_c * args.kappa_e
            for n in range(args.num_nodes):
                nm = copy.deepcopy(global_model)
                nm, ds, loss = local_update(nm, node_loaders[n], flat_epochs, args.lr, device)
                node_models[n] = nm
                node_data_sz[n] = ds
                round_losses.append(loss)
            # Cloud aggregation
            models_list = [node_models[n] for n in range(args.num_nodes)]
            weights_list = [node_data_sz[n] for n in range(args.num_nodes)]
            global_model = weighted_average_models(models_list, weights_list, device)

            accuracy = evaluate_model(global_model, test_loader, device)
            avg_loss = np.mean(round_losses)
            cumulative += rc_gb
            metrics["round"].append(rnd)
            metrics["accuracy"].append(accuracy)
            metrics["loss"].append(avg_loss)
            metrics["per_round_cost"].append(rc_gb)
            metrics["cumulative_cost"].append(cumulative)
            print(f"  Round {rnd:3d}/{args.kappa} | Acc: {accuracy:.4f} | Loss: {avg_loss:.4f} | CumCost: {cumulative:.4f} GB")
    else:
        # ---- HFL loop (same as run_local_simulation.py Step 6) ----
        for rnd in range(1, args.kappa + 1):
            edge_models = {e: copy.deepcopy(global_model) for e in edges}
            round_losses = []

            for edge_epoch in range(1, args.kappa_c + 1):
                node_models = {}
                node_data_sz = {}
                for e in edges:
                    for n in edge_nodes[e]:
                        nm = copy.deepcopy(edge_models[e])
                        nm, ds, loss = local_update(nm, node_loaders[n], args.kappa_e, args.lr, device)
                        node_models[n] = nm
                        node_data_sz[n] = ds
                        round_losses.append(loss)
                # Edge aggregation
                for e in edges:
                    ms = [node_models[n] for n in edge_nodes[e]]
                    ws = [node_data_sz[n] for n in edge_nodes[e]]
                    edge_models[e] = weighted_average_models(ms, ws, device)

            # Cloud aggregation
            em_list = [edge_models[e] for e in sorted(edges)]
            ew_list = [edge_data_sizes[e] for e in sorted(edges)]
            global_model = weighted_average_models(em_list, ew_list, device)

            accuracy = evaluate_model(global_model, test_loader, device)
            avg_loss = np.mean(round_losses) if round_losses else 0.0
            cumulative += rc_gb
            metrics["round"].append(rnd)
            metrics["accuracy"].append(accuracy)
            metrics["loss"].append(avg_loss)
            metrics["per_round_cost"].append(rc_gb)
            metrics["cumulative_cost"].append(cumulative)
            print(f"  Round {rnd:3d}/{args.kappa} | Acc: {accuracy:.4f} | Loss: {avg_loss:.4f} | CumCost: {cumulative:.4f} GB")

    return metrics


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ShapeFL Baseline Comparison (paper Figs 11-12, Table II)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="lenet5", choices=["lenet5", "mobilenetv2", "resnet18"])
    parser.add_argument("--dataset", type=str, default="fmnist", choices=["fmnist", "cifar10", "cifar100"])
    parser.add_argument("--num-nodes", type=int, default=30)

    parser.add_argument("--kappa-p", type=int, default=30)
    parser.add_argument("--kappa-e", type=int, default=1)
    parser.add_argument("--kappa-c", type=int, default=10)
    parser.add_argument("--kappa", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=2800.0)
    parser.add_argument("--B-e", type=int, default=None)
    parser.add_argument("--T-max", type=int, default=30)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--shard-size", type=int, default=15)
    parser.add_argument("--shards-per-node", type=int, default=None)
    parser.add_argument("--classes-per-node", type=int, default=None)

    parser.add_argument("--target-accuracy", type=float, default=0.70,
                        help="Target test accuracy (0-1) for Table-II style comparison")
    parser.add_argument("--strategies", type=str, nargs="+",
                        default=["shapefl", "cost_first", "data_first", "random", "fedavg"],
                        choices=["shapefl", "cost_first", "data_first", "random", "fedavg"],
                        help="Which strategies to run")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory. Auto-generated from config + timestamp if omitted.")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ── Auto-generate unique output directory ─────────────────────────
    if args.output_dir is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = (
            f"results/comparison_{args.model}_{args.dataset}"
            f"_n{args.num_nodes}_k{args.kappa}_{ts}"
        )
    _existing = os.path.join(args.output_dir, "comparison_results.json")
    if os.path.isfile(_existing):
        print(f"  WARNING: {_existing} already exists and will be overwritten.")

    # ── Auto-set from dataset ────────────────────────────────────────────
    ds_info = DATASET_INFO[args.dataset]
    if args.shards_per_node is None:
        args.shards_per_node = ds_info["shards_per_node"]
    if args.classes_per_node is None:
        args.classes_per_node = ds_info["classes_per_node"]
    if args.B_e is None:
        args.B_e = max(3, math.ceil(args.num_nodes / 3))

    # ── Reproducibility ──────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = get_device()

    # ====================================================================
    #  Shared setup (identical for all strategies)
    # ====================================================================
    print("\n" + "=" * 70)
    print("ShapeFL Baseline Comparison Experiment")
    print("=" * 70)
    print(f"Strategies: {args.strategies}")
    print(f"Model: {args.model}  Dataset: {args.dataset}")
    print(f"Nodes: {args.num_nodes}  B_e: {args.B_e}")
    print(f"kappa_e={args.kappa_e}  kappa_c={args.kappa_c}  kappa={args.kappa}")
    print(f"LR: {args.lr}  gamma: {args.gamma}")
    print(f"Target accuracy: {args.target_accuracy*100:.0f}%")

    # ── Global model ─────────────────────────────────────────────────────
    num_classes = ds_info["num_classes"]
    input_channels = ds_info["input_channels"]
    global_model = get_model(args.model, num_classes, input_channels, device)
    num_params, size_mb = get_model_size(global_model)
    model_size_bytes = num_params * 4  # float32
    print(f"Model: {num_params:,} params, {size_mb:.3f} MB")

    initial_state = copy.deepcopy(global_model.state_dict())

    # ── Data ─────────────────────────────────────────────────────────────
    train_dataset, test_dataset = load_data(args.dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    partitions = create_non_iid_partitions(
        train_dataset, args.num_nodes, args.shard_size,
        args.shards_per_node, args.classes_per_node, seed=42,
    )
    node_loaders, data_sizes = {}, {}
    for nid in range(args.num_nodes):
        node_loaders[nid] = get_node_dataloader(train_dataset, partitions[nid], args.batch_size)
        data_sizes[nid] = len(partitions[nid])

    # ── Pre-training + similarity (needed for ShapeFL & Cost First) ──────
    need_pretrain = any(s in args.strategies for s in ["shapefl", "cost_first", "data_first"])
    S = None
    if need_pretrain:
        print(f"\n[Pre-train] {args.kappa_p} epochs per node...")
        linear_updates = {}
        for nid in range(args.num_nodes):
            nm = copy.deepcopy(global_model)
            nm, _, _ = local_update(nm, node_loaders[nid], args.kappa_p, args.lr, device)
            linear_updates[nid] = get_linear_layer_update(nm, global_model)
            print(f"  Node {nid} done (norm {linear_updates[nid].norm():.4f})")
        S = compute_similarity_matrix(linear_updates)
        print(f"  Similarity S: shape {S.shape}, mean {S.mean():.4f}")

    # ── Communication costs ──────────────────────────────────────────────
    c_ne, c_ec = generate_communication_costs(args.num_nodes, model_size_bytes)

    # ====================================================================
    #  Plan each strategy
    # ====================================================================
    plans = {}  # strategy_name -> (edges, edge_nodes, edge_data_sizes, is_flat)

    if "shapefl" in args.strategies:
        print("\n[Plan] ShapeFL (LoS + GoA, γ={:.0f})...".format(args.gamma))
        e, en, eds = plan_shapefl(args, c_ne, c_ec, S, data_sizes)
        plans["ShapeFL"] = (e, en, eds, False)

    if "cost_first" in args.strategies:
        print("\n[Plan] Cost First (LoS + GoA, γ=0)...")
        e, en, eds = plan_cost_first(args, c_ne, c_ec, S, data_sizes)
        plans["Cost First"] = (e, en, eds, False)

    if "data_first" in args.strategies:
        print("\n[Plan] Data First (LoS + GoA, \u03b3\u2192\u221e)...")
        e, en, eds = plan_data_first(args, c_ne, c_ec, S, data_sizes)
        plans["Data First"] = (e, en, eds, False)

    if "random" in args.strategies:
        print("\n[Plan] Random edge selection...")
        e, en, eds = plan_random(args, data_sizes)
        plans["Random"] = (e, en, eds, False)

    if "fedavg" in args.strategies:
        print("\n[Plan] Flat FedAvg (no hierarchy)...")
        e, en, eds = plan_fedavg_flat(args, data_sizes)
        plans["FedAvg"] = (e, en, eds, True)

    # ====================================================================
    #  Run each strategy
    # ====================================================================
    all_metrics = {}
    for name, (edges, edge_nodes, edge_data_sizes, is_flat) in plans.items():
        # Reset seeds so SGD noise is identical across strategies
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        start = time.time()
        prc_gb = compute_per_round_cost_gb(c_ne, c_ec, edges, edge_nodes, args.kappa_c, is_flat)
        m = run_strategy(
            name, args, initial_state, node_loaders, data_sizes,
            test_loader, edges, edge_nodes, edge_data_sizes,
            prc_gb, device, is_flat,
        )
        m["time_seconds"] = time.time() - start
        all_metrics[name] = m

    # ====================================================================
    #  Analysis & summary
    # ====================================================================
    print("\n" + "=" * 70)
    print("  COMPARISON RESULTS")
    print("=" * 70)

    target = args.target_accuracy

    # Header
    print(f"\n{'Strategy':<14} {'Final Acc':>10} {'Best Acc':>10} "
          f"{'Per-Round':>14} {'Cost@{:.0f}%'.format(target*100):>14} "
          f"{'Rounds@{:.0f}%'.format(target*100):>12} {'Time':>8}")
    print("-" * 84)

    summary = {}
    for name, m in all_metrics.items():
        final_acc = m["accuracy"][-1]
        best_acc = max(m["accuracy"])
        per_round = m["per_round_cost"][0] if m["per_round_cost"] else 0

        # Cost to reach target accuracy
        cost_at_target = None
        rounds_at_target = None
        for i, acc in enumerate(m["accuracy"]):
            if acc >= target:
                cost_at_target = m["cumulative_cost"][i]
                rounds_at_target = m["round"][i]
                break

        cost_str = f"{cost_at_target:.4f} GB" if cost_at_target else "NOT REACHED"
        rounds_str = str(rounds_at_target) if rounds_at_target else "-"
        t = m.get("time_seconds", 0)

        print(f"{name:<14} {final_acc*100:>9.2f}% {best_acc*100:>9.2f}% "
              f"{per_round:.6f}GB {cost_str:>14} "
              f"{rounds_str:>12} {t:>7.1f}s")

        summary[name] = {
            "final_accuracy": final_acc,
            "best_accuracy": best_acc,
            "per_round_cost_gb": per_round,
            "cost_to_target_gb": cost_at_target,
            "rounds_to_target": rounds_at_target,
            "time_seconds": t,
        }

    # Communication cost savings (ShapeFL vs others)
    if "ShapeFL" in summary and summary["ShapeFL"]["cost_to_target_gb"]:
        shapefl_cost = summary["ShapeFL"]["cost_to_target_gb"]
        print(f"\n--- Communication cost savings (ShapeFL vs baselines to reach {target*100:.0f}% accuracy) ---")
        for name, s in summary.items():
            if name == "ShapeFL":
                continue
            if s["cost_to_target_gb"]:
                saving = (1 - shapefl_cost / s["cost_to_target_gb"]) * 100
                print(f"  vs {name:<12}: {saving:+.1f}% {'(saved)' if saving > 0 else '(worse)'}")
            else:
                print(f"  vs {name:<12}: baseline did NOT reach {target*100:.0f}%")

    # ── Save results ─────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    def to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return sorted(list(obj))
        if isinstance(obj, dict):
            return {str(k): to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_native(x) for x in obj]
        return obj

    output = {
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "num_nodes": args.num_nodes,
            "kappa_e": args.kappa_e,
            "kappa_c": args.kappa_c,
            "kappa": args.kappa,
            "gamma": args.gamma,
            "B_e": args.B_e,
            "lr": args.lr,
            "target_accuracy": target,
        },
        "summary": to_native(summary),
        "per_round_metrics": to_native(all_metrics),
    }

    out_path = os.path.join(args.output_dir, "comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Visualization ────────────────────────────────────────────────────
    try:
        from utils.visualization import visualize_comparison
        visualize_comparison(
            all_metrics=all_metrics,
            summary=summary,
            config=output["config"],
            target_accuracy=target,
            output_dir=args.output_dir,
        )
    except Exception as e:
        print(f"[Warning] Visualization failed: {e}")


if __name__ == "__main__":
    main()
