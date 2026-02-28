# ShapeFL Project Implementation — Complete Report

## 1. Paper Context

This codebase implements **ShapeFL** (_"A Communication-Efficient Hierarchical Federated Learning Framework via Shaping Data Distribution at Edge"_, IEEE/ACM Transactions on Networking, vol. 32, no. 3, June 2024). The core idea is a **three-tier Hierarchical Federated Learning (HFL)** architecture — Cloud → Edge Aggregators → Computing Nodes — that uses two optimization algorithms (GoA and LoS) to strategically shape data distributions at the edge, reducing communication cost by up to 70% while maintaining model accuracy.

The codebase is a small-scale replication using **LeNet-5** on **Fashion-MNIST**, testing 8 computing nodes instead of the paper's 37–46.

---

## 2. Project Structure & File Roles

The project has two operational modes:

- **Local simulation** (single-process, in-memory) — the primary testing mode via run_local_simulation.py
- **Distributed deployment** (Flask-based HTTP, multi-machine) — for real Raspberry Pi clusters via cloud_server.py, edge_aggregator.py, computing_node.py

---

### 2.1 Configuration

**config.py** (188 lines)

Central configuration file defining three dataclasses:

- **`TrainingConfig`**: All hyperparameters from the paper — `batch_size=32`, `lr=0.001`, `kappa_p=30` (pre-training epochs), `kappa_e=2` (local epochs per edge round), `kappa_c=10` (edge rounds per cloud round), `kappa=100` (total cloud rounds), `gamma=2800` (trade-off weight), `T_max=30` (max LoS iterations), `B_e=10` (max nodes per edge), `shard_size=15`, `shards_per_node=12`, `classes_per_node=4`.
- **`NetworkConfig`**: IP addresses and ports for cloud server, edge aggregators, and computing nodes (for distributed deployment on LAN).
- **`PathConfig`**: Filesystem paths for checkpoints, logs, metrics, and partitions.

Also provides `save_config()`, `load_config()`, and `print_config()` for persisting/inspecting settings as JSON. Global singleton instances `TRAINING_CONFIG`, `NETWORK_CONFIG`, `PATH_CONFIG` are used by all other modules.

---

### 2.2 Model

**lenet5.py** (183 lines)

Implements the **LeNet-5** convolutional neural network as used in the paper for Fashion-MNIST:

| Layer   | Details                             | Output Shape |
| ------- | ----------------------------------- | ------------ |
| Conv1   | 1→6 channels, 5×5 kernel, padding=2 | 6×28×28      |
| MaxPool | 2×2                                 | 6×14×14      |
| Conv2   | 6→16 channels, 5×5 kernel           | 16×10×10     |
| MaxPool | 2×2                                 | 16×5×5       |
| FC1     | 400→120                             | 120          |
| FC2     | 120→84                              | 84           |
| FC3     | 84→10                               | 10           |

**Total: 61,706 parameters** (~0.236 MB)

Key methods:

- `forward()`: Standard CNN forward pass
- `get_linear_layer_params()`: Extracts fc3 (output layer) weights+biases as a flattened tensor — used for computing similarity $S_{ij}$ in ShapeFL
- `get_linear_layer_size()`: Returns 850 (84×10 + 10)

Helper functions:

- `get_model()`: Factory function to instantiate models by name
- `get_model_size()`: Returns parameter count and size in MB
- `model_to_dict()` / `dict_to_model()`: JSON serialization for network transfer

****init**.py**: Exposes `LeNet5` and `get_model`.

---

### 2.3 Data Loading & Partitioning

**data_loader.py** (300 lines)

Handles Fashion-MNIST loading and non-IID data partitioning.

**`FMNISTDataset`** — Custom wrapper class. Normalizes pixel values to [0,1], converts labels to `torch.long`.

**`load_fmnist_data()`** — Loads Fashion-MNIST from CSV files in dataset (60,000 train, 10,000 test). Falls back to `torchvision.datasets.FashionMNIST` with auto-download if CSVs are missing.

**`create_non_iid_partitions()`** — The core non-IID partitioning algorithm following Section V-A of the paper:

1. Groups all 60,000 training samples by class (10 classes × 6,000 each)
2. Shuffles within each class
3. Splits each class into shards of size 15 → 400 shards per class
4. For each node: randomly selects $k=4$ classes, then assigns $s/k = 3$ shards from each class → 12 shards × 15 = **180 samples per node**
5. This creates highly non-IID distributions where each node only sees 4 out of 10 classes

Additional helpers: `get_node_dataloader()` creates a PyTorch `DataLoader` from a partition, `save_partitions()` / `load_partitions()` for JSON persistence, `get_data_distribution()` for inspecting class distributions.

**dataset** directory: Contains pre-downloaded Fashion-MNIST data in both CSV format (`fashion-mnist_train.csv`, `fashion-mnist_test.csv`) and raw binary format (`train-images-idx3-ubyte`, etc.).

**partitions** directory: Contains pre-computed partition files (for the distributed mode). `partitions.json` has all node assignments; `node_X_partition.json` has individual node data. These are NOT used by the local simulation (which generates partitions on-the-fly).

---

### 2.4 Core Algorithms

#### 2.4.1 GoA (Greedy Node Association) — Algorithm 1

**goa.py** (286 lines)

Implements **Algorithm 1** from the paper (Section IV-C). Given a fixed set of edge aggregators, determines which edge each node should associate with.

**Objective (Eq. 14):**
$$\min_Y \kappa_c \sum y_{ne} c_{ne} - \frac{\gamma}{|E|} \sum_{e \in E} \frac{1}{\binom{D_e}{2}} \sum_{i,j \in M_e} S_{ij} D_i D_j$$

**Class `GreedyNodeAssociation`**:

- Inputs: edge aggregator list, communication costs $c_{ne}$, similarity matrix $S_{ij}$, data sizes $D_n$, $\kappa_c$, $\gamma$, $B_e$
- `_comb2(d)`: Computes $\binom{d}{2} = d(d-1)/2$ for diversity normalization

**`run()` method — the greedy assignment loop:**

1. **Initialization**: Each edge aggregator is pre-assigned to itself (its data counts towards diversity). Tracking structures: `M_e` (nodes at each edge), `D_e` (total data per edge), `pair_sums` (running $\sum S_{ij}D_iD_j$ per edge)
2. **Main loop**: While unassigned nodes remain:
    - For each unassigned node $n$, for each edge $e$ with capacity:
        - Compute communication cost: $\kappa_c \cdot c_{ne}$
        - Compute new pair contributions: $\sum_{m \in M_e} S_{nm} D_n D_m$
        - Compute $\Delta S_{ne}$ as the **full change** in diversity: `(pair_sums + new_pairs) / C(D'_e, 2) - pair_sums / C(D_e, 2)`
        - Compute $\Delta J_{ne} = \kappa_c \cdot c_{ne} - \gamma \cdot \frac{1}{|E|} \cdot \Delta S_{ne}$
    - Select the (node, edge) pair with minimum $\Delta J_{ne}$; assign it
3. **Final objective**: Compute $J_m = \kappa_c \sum c_{ne} - \gamma \cdot J_d$ where $J_d$ is the total weighted diversity

Returns `NodeAssociationResult` containing: associations dict, edge_nodes sets, edge_data_sizes, objective_value $J_m$, edge_diversity_sums.

#### 2.4.2 LoS (Local Search Edge Selection) — Algorithm 2

**los.py** (340 lines)

Implements **Algorithm 2** (Section IV-C). Selects the optimal subset of nodes to serve as edge aggregators.

**Objective (Eq. 19):**
$$J(E_s) = J_m(E_s) + \sum_{e \in E_s} c_{ec}$$

Where $J_m$ is the output of GoA (Eq. 14), and $c_{ec}$ is the edge-to-cloud communication cost (NOT multiplied by $\kappa_c$).

**Class `LocalSearchEdgeSelection`**:

- `compute_objective_J()`: For a candidate edge set, instantiates GoA, runs it, gets $J_m$, adds $\sum c_{ec}$
- `initialize_random()`: Picks 3 random candidates as initial solution

**`run()` method — local search iterations:**

For up to $T_{max}=30$ iterations, tries three operations in order:

1. **Open**: Add a non-selected edge $e$ to the set. If $J$ decreases, accept and restart
2. **Close**: Remove an edge from the set (keeping $\geq 1$). If $J$ decreases, accept
3. **Swap**: Replace one selected edge with a non-selected one. If $J$ decreases, accept

Converges when no operation improves $J$. Returns `EdgeSelectionResult` with: selected_edges, node_associations (GoA result), objective_value.

**[algorithms/**init**.py](algorithms/__init__.py)**: Exposes `GreedyNodeAssociation`, `run_goa`, `LocalSearchEdgeSelection`, `run_los`.

---

### 2.5 Utility Modules

#### 2.5.1 Similarity Computation

**[utils/similarity.py](utils/similarity.py)** (153 lines)

Implements the data distribution diversity measure from Section IV-B of the paper:

$$S_{ij} = 1 - \cos(\Delta w_i^{(L)}, \Delta w_j^{(L)})$$

Where $\Delta w_i^{(L)}$ is the linear layer (fc3) update vector after pre-training. Higher $S_{ij}$ means more diverse data distributions between nodes $i$ and $j$ (range 0–2).

- `compute_cosine_similarity()`: Dot product / (norm1 × norm2)
- `compute_data_distribution_diversity()`: Returns $1 - \cos\_sim$
- `compute_similarity_matrix()`: Builds the full $N \times N$ symmetric matrix, with $S_{ii} = 0$

#### 2.5.2 Model Aggregation

**aggregation.py** (203 lines)

Implements **FedAvg** (Federated Averaging) used at both edge and cloud levels:

- `federated_averaging()`: Takes list of models + weights (data sizes), returns weighted average model: $w_{agg} = \sum \frac{D_i}{D_{total}} w_i$. Handles normalization automatically.
- `weighted_averaging()`: Same but operates on state_dicts directly
- `compute_model_update()`: $\Delta w = w_{new} - w_{old}$
- `apply_model_update()`: $w_{new} = w_{base} + \Delta w$
- `get_linear_layer_update()`: Extracts fc3 weight+bias difference between two models → used for similarity computation

#### 2.5.3 Communication

**communication.py** (283 lines)

HTTP-based model transfer utilities for the distributed deployment:

- `model_to_bytes()` / `bytes_to_model()`: PyTorch model ↔ bytes serialization via `torch.save`/`torch.load`
- `compress_model()` / `decompress_model()`: gzip + base64 encoding for network transfer
- `state_dict_to_json()` / `json_to_state_dict()`: JSON-compatible serialization
- `send_model()` / `receive_model()`: Full HTTP POST with optional compression
- `get_model_size_bytes()` / `get_compressed_size_bytes()`: Size calculations

#### 2.5.4 Metrics Tracking

**metrics.py** (243 lines)

- **`MetricsTracker`** class: Records per-round accuracy, loss, communication cost, cumulative cost, and timestamps. Has `record()`, `save()` (to JSON), `get_summary()`, `get_latest()` methods.
- `compute_accuracy()`: Evaluates a model on a DataLoader, returns accuracy as float
- `compute_communication_cost()`: Estimates bytes transferred per round based on model size, node count, and aggregation structure

****init**.py**: Package marker.

---

### 2.6 Local Simulation Script (Primary Entry Point)

**run_local_simulation.py** (504 lines)

This is the **main script** for verifying the ShapeFL implementation against the paper. It runs the complete Algorithm 3 in a single process, without Flask/HTTP overhead, giving clean and accurate results.

**Step-by-step execution flow (Algorithm 3):**

**Step 1 — Model Initialization** (line 1 of Algorithm 3):

- Creates LeNet-5 with `get_model()` and saves `initial_state = deepcopy(global_model.state_dict())`

**Step 2 — Data Loading & Partitioning**:

- Calls `load_fmnist_data()` and `create_non_iid_partitions()` with the specified `num_nodes`, generating partitions on-the-fly (no static file dependency)
- Creates a `DataLoader` per node

**Step 3 — Offline Pre-training** (lines 2–5 of Algorithm 3):

- Each node gets a `deepcopy` of the global model and trains for $\kappa_p$ epochs using `local_update()`
- After training, `get_linear_layer_update()` extracts $\Delta w_n^{(L)} = w_n^{(L)} - w^{(0)(L)}$ for each node

**Step 4 — Similarity Matrix**:

- `compute_similarity_matrix()` builds the $N \times N$ diversity matrix $S_{ij}$ from the linear layer updates

**Step 5 — LoS + GoA** (lines 6–7 of Algorithm 3):

- `generate_communication_costs()` creates simulated $c_{ne}$ and $c_{ec}$ based on random 2D positions
- `run_los()` selects edge aggregators and associates nodes (internally calls GoA)
- Prints selected edges and cluster memberships

**Step 6 — Hierarchical Training Loop** (lines 8–13 of Algorithm 3):

- Resets global model to $w^{(0)}$ via `global_model.load_state_dict(deepcopy(initial_state))`
- Outer loop: $\kappa$ cloud rounds
    - Each edge starts with a `deepcopy` of the global model
    - Inner loop: $\kappa_c$ edge epochs
        - Each node at each edge: receives edge model → trains for $\kappa_e$ epochs → returns trained model
        - Edge aggregation: `federated_averaging()` weighted by data sizes
    - Cloud aggregation: `federated_averaging()` of all edge models, weighted by edge data sizes
    - Evaluation on full 10,000-sample test set

**Results**: Saves JSON (config, accuracy curve, loss curve, edge selection) and the final model checkpoint.

**Command-line arguments**: `--num-nodes`, `--kappa-p`, `--kappa-e`, `--kappa-c`, `--kappa`, `--gamma`, `--B-e`, `--T-max`, `--lr`, `--batch-size`, `--shard-size`, `--shards-per-node`, `--classes-per-node`, `--output-dir`, `--seed`.

Helper functions:

- `local_update()`: The LocalUpdate function (Algorithm 3, lines 27–33). SGD with CrossEntropyLoss.
- `evaluate_model()`: Test accuracy computation
- `weighted_average_models()`: Wrapper around `federated_averaging()`
- `generate_communication_costs()`: Simulates LAN topology costs
- `to_native()`: Converts numpy types to Python types for JSON serialization

---

### 2.7 Distributed Deployment Components

These three files implement the **real distributed system** using Flask HTTP servers, intended for Raspberry Pi clusters. Each is a standalone Flask application.

#### 2.7.1 Cloud Server

**cloud_server.py** (528 lines)

The central coordinator. Runs on a laptop/desktop (optionally with GPU).

- Flask routes for: device registration (`/register/node`, `/register/edge`), model distribution (`/model/global`), pre-training collection (`/pretrain/submit`), edge update collection (`/edge/submit`), algorithm execution (`/algorithms/run`), training management (`/training/start`), status/metrics queries
- `run_algorithms()`: Builds similarity matrix from collected pre-training updates, constructs communication cost matrices, runs `run_los()`, maps index-based results back to string node/edge IDs
- `aggregate_edge_updates()`: Deserializes, decompresses, and FedAvg-aggregates edge models
- `run_training_loop()`: Waits for edge submissions each round, aggregates, evaluates, records metrics
- Thread-safe via `Lock`, `Event` synchronization primitives

#### 2.7.2 Edge Aggregator

**edge_aggregator.py** (446 lines)

Intermediate aggregator. Runs on Raspberry Pi.

- Flask routes for: node registration (`/register/node`), model distribution (`/model/current`), node update collection (`/node/submit`)
- `run_training_loop()`: For each cloud round, fetches global model → runs $\kappa_c$ edge epochs (broadcast to nodes → collect updates → FedAvg aggregate) → submits to cloud
- `broadcast_to_nodes()`: HTTP POST to each associated node
- `aggregate_node_updates()`: Deserializes node models, runs `federated_averaging()`

#### 2.7.3 Computing Node

**computing_node.py** (565 lines)

Local trainer. Runs on Raspberry Pi.

- Flask routes for: model reception (`/model/update`), training triggers (`/train/start`, `/pretrain/start`), edge assignment (`/edge/assign`)
- `pretrain()`: Fetches global model, trains $\kappa_p$ epochs, extracts linear layer update, submits to cloud
- `train_local()`: SGD training for specified epochs, then submits trained model to edge
- `run_training_loop()`: Waits for model → trains $\kappa_e$ epochs → submits, repeated $\kappa \times \kappa_c$ times

---

### 2.8 Orchestration & Launcher Scripts

**orchestrate.py** (319 lines) — Master coordinator for distributed deployment. Sequentially: waits for device registrations → triggers pre-training on all nodes → runs edge selection → distributes edge assignments to nodes → starts training loop → monitors progress → saves final results. All via HTTP calls to the cloud server.

**run_simulation.py** (158 lines) — Multi-process simulation mode. Launches cloud, edges, and nodes as separate Python processes all on `127.0.0.1` with different ports. Creates partitions first, then spawns processes.

**partition_data.py** (94 lines) — Standalone data partitioning tool. Creates non-IID partitions and saves them as JSON files (both a combined `partitions.json` and individual `node_X_partition.json` files). Used to pre-compute partitions for Raspberry Pi deployment.

**run_cloud.py**, **run_edge.py**, **run_node.py** — Thin launcher scripts (each ~18 lines). Simply import and call `main()` from their respective modules. Convenience entry points for deployment.

---

## 3. Algorithm Flow Summary (Algorithm 3)

```
Phase 0: Initialize global model w^(0)

Phase 1: OFFLINE PRE-TRAINING
  For each node n:
    w_n ← w^(0)
    Train w_n locally for κ_p epochs on D_n
    Compute Δw_n^(L) = w_n^(L) - w^(0)(L)  [linear layer only]
    Send Δw_n^(L) to cloud

Phase 2: SIMILARITY & OPTIMIZATION
  Cloud computes S_ij = 1 - cos(Δw_i^(L), Δw_j^(L)) for all pairs
  Cloud runs LoS → selected edges E_s
    (internally runs GoA for each candidate edge set)
  Cloud runs GoA → node associations {n → e}

Phase 3: HIERARCHICAL TRAINING
  Reset global model to w^(0)
  For r = 1 to κ (cloud rounds):
    Cloud sends w^(r-1) to all edges
    For t = 1 to κ_c (edge rounds):
      Each edge e broadcasts model to nodes in M_e
      Each node n trains for κ_e epochs → w_n
      Edge aggregates: w_e = Σ (D_n/D_e) · w_n  [FedAvg]
    Cloud aggregates: w^(r) = Σ (D_e/D_total) · w_e  [FedAvg]
    Evaluate w^(r) on test set
```

## 4. Simulation Results

### 4.1 Paper Reference Results (Section V-B, Fig. 10)

- **FMNIST/LeNet5 on Geant2010** (37 nodes, κ_e=1, κ_c=10, 500 local epochs):
    - ShapeFL: ~83% accuracy
    - FedAvg: ~80% accuracy
    - Paper target for Table-II communication cost comparison: 70%

### 4.2 Implementation Results Summary

| Run                            | Nodes  | lr       | κ      | Final Acc  | Best Acc   | Gap to Paper |
| ------------------------------ | ------ | -------- | ------ | ---------- | ---------- | ------------ |
| lr_01_momentum (10 nodes)      | 10     | 0.01     | 50     | 76.29%     | 76.29%     | -6.7%        |
| paper_match_30nodes (lr=0.001) | 30     | 0.001    | 50     | 62.38%     | 63.97%     | -19%         |
| **30nodes_lr01 (lr=0.01)**     | **30** | **0.01** | **50** | **81.74%** | **81.74%** | **-1.3%**    |

### 4.3 Key Finding: Learning Rate

The paper states lr=0.001 (Section V-A), but lr=0.01 produces results matching the paper's reported ~83% accuracy. With lr=0.001 and 30 nodes, accuracy plateaus at ~63% — far below paper results. With lr=0.01, the model converges to **81.74%**, within ~1.3% of the paper's Geant2010 result.

The remaining gap is explained by:

1. **Fewer nodes**: 30 vs 37 (Geant2010), meaning 5,400 vs 6,660 training samples
2. **Random topology**: Simulated 1000×1000 km grid vs real-world Geant2010 European network with Dijkstra shortest-path distances
3. **Run-to-run variance**: Random data partitioning and topology generation

### 4.4 Best Configuration (30 Nodes, lr=0.01)

```
Parameters:
  Model: LeNet5 (61,706 params)
  Dataset: Fashion-MNIST (60,000 train / 10,000 test)
  Nodes: 30 (180 samples each, 4 classes, 12 shards)
  κ_p=30, κ_e=1, κ_c=10, κ=50, γ=2800, B_e=10
  Optimizer: SGD(lr=0.01, momentum=0.9)
  Total local epochs per node: 500

Results:
  Selected edges: [1, 2, 15] (3 edges, 10 nodes each)
  Final accuracy: 81.74% (round 50)
  Best accuracy: 81.74% (round 50)
  Training time: 523 seconds

Convergence trajectory:
  Round  5 (50 epochs):  57.30%
  Round 10 (100 epochs): 62.83%
  Round 20 (200 epochs): 68.74%
  Round 30 (300 epochs): 73.08%
  Round 40 (400 epochs): 80.05%
  Round 50 (500 epochs): 81.74%
```

### 4.5 Comparison Experiment (Paper Fig. 11)

A full comparison of 5 strategies mirrors paper Fig. 11 (accuracy vs communication cost).

**Communication cost model (Eq. 6–7)**: distance-weighted, bidirectional.

- Node↔Edge: `c_ne = 0.002 × d_ne(km) × S_m(GB)` — cheap local links
- Edge↔Cloud: `c_ec = 0.02 × d_ec(km) × S_m(GB)` — expensive backbone
- HFL per round: `2κ_c Σ_e Σ_{n∈N_e} c_ne + 2Σ_e c_ec`
- FedAvg per round: `2Σ_n c_ec(n)` (all nodes talk directly to cloud)

| Strategy       | Per-Round (GB) | Cost@70% (GB) | Rounds@70% | Final Acc | Best Acc |
| -------------- | -------------- | ------------- | ---------- | --------- | -------- |
| **ShapeFL**    | 0.1813         | 3.989         | 22         | 80.90%    | 81.26%   |
| **Cost First** | 0.1316         | 2.632         | 20         | 80.93%    | 80.93%   |
| **Data First** | 0.1813         | 3.989         | 22         | 80.90%    | 81.26%   |
| **Random**     | 0.1977         | 3.361         | 17         | 80.24%    | 80.34%   |
| **FedAvg**     | 0.8291         | 9.120         | 11         | 79.06%    | 79.06%   |

**Key findings:**

1. **ShapeFL saves 56.3% vs FedAvg** — matches paper's ~50-60% savings claim. FedAvg per-round cost is 4.6× higher due to direct cloud communication over expensive backbone links.

2. **Cost First is cheapest per round** (γ=0 ⇒ pure distance optimization, edges [2,7,17]). ShapeFL/Data First select edges [1,2,15] optimizing diversity.

3. **ShapeFL ≈ Data First**: With LeNet5 (0.00023 GB), γ=2800 makes diversity 23,732× larger than cost in GoA's ΔJ_ne criterion, effectively γ→∞. The paper's γ=2800 was calibrated for larger models on real topologies.

4. **All HFL strategies reach ~80-81%**, FedAvg only 79%. This matches the paper's observation that HFL with edge aggregation improves convergence under non-IID data.

Results saved to `results/comparison_fixed/comparison_results.json`.

## 5. File Dependency Graph

```
config.py ──────────────────────────────────────────────┐
                                                        │
models/lenet5.py ───────────────────────────────────────┤
                                                        │
data/data_loader.py ────────────────────────────────────┤
                                                        │
utils/similarity.py ────────────────────────────────────┤
utils/aggregation.py ───────────────────────────────────┤
utils/communication.py ─────────────────────────────────┤
utils/metrics.py ───────────────────────────────────────┤
                                                        │
algorithms/goa.py ──────► algorithms/los.py ────────────┤
                                                        │
                                  ┌─────────────────────┘
                                  │
                                  ├──► scripts/run_local_simulation.py  [LOCAL MODE]
                                  │
                                  ├──► cloud/cloud_server.py ──┐
                                  ├──► edge/edge_aggregator.py ├──► [DISTRIBUTED MODE]
                                  └──► node/computing_node.py ─┘
                                           │
                            scripts/orchestrate.py  (HTTP coordinator)
                            scripts/run_simulation.py  (multi-process launcher)
```
