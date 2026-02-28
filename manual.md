# ShapeFL — User Manual

> **ShapeFL**: _A Communication-Efficient Hierarchical Federated Learning Framework via Shaping Data Distribution at Edge_
> IEEE/ACM Transactions on Networking, 2024

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Installation & Environment Setup](#2-installation--environment-setup)
3. [Dataset Guide](#3-dataset-guide)
4. [Supported Model / Dataset Combinations](#4-supported-model--dataset-combinations)
5. [Running a Simulation](#5-running-a-simulation)
6. [All Command-Line Arguments](#6-all-command-line-arguments)
7. [Paper-Exact Experiment Configurations](#7-paper-exact-experiment-configurations)
8. [Code Architecture & File Reference](#8-code-architecture--file-reference)
9. [Understanding the Output](#9-understanding-the-output)
10. [Distributed Deployment (Raspberry Pi / LAN)](#10-distributed-deployment-raspberry-pi--lan)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Project Overview

ShapeFL implements a **three-tier Hierarchical Federated Learning (HFL)** system:

```
Cloud Server         (global aggregator)
    │
    ├── Edge Aggregator 1  ──  Node A, Node B, ...
    ├── Edge Aggregator 2  ──  Node C, Node D, ...
    └── ...
```

### Key Algorithms

| Algorithm                                  | Purpose                                                                                                             | Paper Reference           |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| **GoA** (Greedy-based optimal Association) | Assign computing nodes to edge aggregators to minimise an objective combining communication cost and data diversity | Algorithm 1, Section IV-B |
| **LoS** (Local Search for edge Selection)  | Select which nodes serve as edge aggregators from a set of candidates                                               | Algorithm 2, Section IV-C |
| **Algorithm 3** (Full Pipeline)            | Combines pre-training → similarity → LoS+GoA → hierarchical FedAvg training                                         | Algorithm 3, Section IV-D |

### What the simulation does

1. **Pre-training**: Each node trains a copy of the global model for `κ_p` epochs on its local non-IID data.
2. **Similarity Matrix**: Cosine distance of output-layer updates quantifies data distribution diversity between all node pairs.
3. **Edge Selection & Node Association**: LoS selects which nodes become edge aggregators; GoA assigns every node to an edge.
4. **Hierarchical Training**: `κ` cloud rounds × `κ_c` edge epochs × `κ_e` local epochs of FedAvg, with edge-level and cloud-level aggregation.

---

## 2. Installation & Environment Setup

### Prerequisites

- **Python 3.10+** (tested with 3.14)
- **pip** (comes with Python)
- A machine with at least 8 GB RAM (16 GB recommended for ResNet18 + CIFAR-100)
- (Optional) NVIDIA GPU with CUDA for faster training

### Step-by-step

```bash
# 1. Clone or navigate to the project
cd shapefl_/src

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
#    Windows PowerShell:
.\.venv\Scripts\Activate.ps1
#    Windows cmd:
.\.venv\Scripts\activate.bat
#    Linux / macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt` contents

```
torch
torchvision
numpy
pandas
flask
requests
```

> **Note**: `pandas` is only needed for loading Fashion-MNIST from CSV. CIFAR-10/100 are handled entirely by `torchvision`.

---

## 3. Dataset Guide

### 3.1 Fashion-MNIST (FMNIST)

| Property      | Value                                                                                 |
| ------------- | ------------------------------------------------------------------------------------- |
| Classes       | 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot) |
| Image size    | 28 × 28, grayscale (1 channel)                                                        |
| Train samples | 60,000                                                                                |
| Test samples  | 10,000                                                                                |

**How to obtain:**

Fashion-MNIST is **already included** in the `dataset/` folder as CSV files:

- `fashion-mnist_train.csv`
- `fashion-mnist_test.csv`

The code will auto-detect and load them. If the CSV files are missing, `torchvision` will **auto-download** the dataset on first run.

### 3.2 CIFAR-10

| Property      | Value                                                                     |
| ------------- | ------------------------------------------------------------------------- |
| Classes       | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| Image size    | 32 × 32, colour (3 channels)                                              |
| Train samples | 50,000                                                                    |
| Test samples  | 10,000                                                                    |
| Download size | ~170 MB                                                                   |

**How to obtain:**

CIFAR-10 is **auto-downloaded** by `torchvision` on first run. No manual action needed. Files are cached in `dataset/cifar-10-batches-py/`.

If you prefer manual download:

1. Go to https://www.cs.toronto.edu/~kriz/cifar.html
2. Download "CIFAR-10 python version" (`cifar-10-python.tar.gz`)
3. Extract to `dataset/`

### 3.3 CIFAR-100

| Property      | Value                              |
| ------------- | ---------------------------------- |
| Classes       | 100 (grouped into 20 superclasses) |
| Image size    | 32 × 32, colour (3 channels)       |
| Train samples | 50,000                             |
| Test samples  | 10,000                             |
| Download size | ~170 MB                            |

**How to obtain:**

CIFAR-100 is **auto-downloaded** by `torchvision` on first run. No manual action needed. Files are cached in `dataset/cifar-100-python/`.

If you prefer manual download:

1. Go to https://www.cs.toronto.edu/~kriz/cifar.html
2. Download "CIFAR-100 python version" (`cifar-100-python.tar.gz`)
3. Extract to `dataset/`

### Summary

| Dataset   | Auto-download?           | First-run download size | Local cache location           |
| --------- | ------------------------ | ----------------------- | ------------------------------ |
| FMNIST    | ✅ (CSV already bundled) | —                       | `dataset/*.csv`                |
| CIFAR-10  | ✅ via torchvision       | ~170 MB                 | `dataset/cifar-10-batches-py/` |
| CIFAR-100 | ✅ via torchvision       | ~170 MB                 | `dataset/cifar-100-python/`    |

---

## 4. Supported Model / Dataset Combinations

The paper evaluates three model/dataset pairings:

| #   | Model                         | Dataset       | Classes | Input Size  | Paper Partitioning |
| --- | ----------------------------- | ------------- | ------- | ----------- | ------------------ |
| 1   | **LeNet-5** (61K params)      | Fashion-MNIST | 10      | 1 × 28 × 28 | s=12, k=4          |
| 2   | **MobileNetV2** (2.2M params) | CIFAR-10      | 10      | 3 × 32 × 32 | s=12, k=4          |
| 3   | **ResNet18** (11.2M params)   | CIFAR-100     | 100     | 3 × 32 × 32 | s=100, k=20        |

Where:

- **s** = number of shards per computing node
- **k** = number of classes per node (creates non-IID distribution)

---

## 5. Running a Simulation

All simulations are launched from the `src/` directory:

```bash
cd shapefl_/src
```

### Quick Start — LeNet-5 + Fashion-MNIST

```bash
python scripts/run_local_simulation.py --model lenet5 --dataset fmnist
```

### Quick Start — MobileNetV2 + CIFAR-10

```bash
python scripts/run_local_simulation.py --model mobilenetv2 --dataset cifar10
```

### Quick Start — ResNet18 + CIFAR-100

```bash
python scripts/run_local_simulation.py --model resnet18 --dataset cifar100
```

> **Tip**: On first run with CIFAR datasets, there will be a one-time download. Subsequent runs use the cached data.

### Customising Parameters

```bash
python scripts/run_local_simulation.py \
    --model mobilenetv2 \
    --dataset cifar10 \
    --num-nodes 12 \
    --kappa-p 30 \
    --kappa-e 1 \
    --kappa-c 10 \
    --kappa 100 \
    --gamma 2800 \
    --lr 0.001 \
    --batch-size 32 \
    --output-dir results/my_experiment
```

---

## 6. All Command-Line Arguments

| Argument             | Type  | Default   | Description                                                         |
| -------------------- | ----- | --------- | ------------------------------------------------------------------- |
| `--model`            | str   | `lenet5`  | Model: `lenet5`, `mobilenetv2`, `resnet18`                          |
| `--dataset`          | str   | `fmnist`  | Dataset: `fmnist`, `cifar10`, `cifar100`                            |
| `--num-nodes`        | int   | `8`       | Total number of computing nodes                                     |
| `--kappa-p`          | int   | `30`      | Pre-training epochs per node (for similarity computation)           |
| `--kappa-e`          | int   | `1`       | Local training epochs per edge round                                |
| `--kappa-c`          | int   | `10`      | Edge aggregation rounds per cloud round                             |
| `--kappa`            | int   | `50`      | Total cloud aggregation rounds                                      |
| `--gamma`            | float | `2800.0`  | Trade-off weight γ (balances communication cost vs. data diversity) |
| `--B-e`              | int   | `10`      | Maximum nodes per edge aggregator                                   |
| `--T-max`            | int   | `30`      | Maximum LoS iterations                                              |
| `--lr`               | float | `0.001`   | Learning rate (SGD)                                                 |
| `--batch-size`       | int   | `32`      | Mini-batch size                                                     |
| `--shard-size`       | int   | `15`      | Shard size for non-IID partitioning                                 |
| `--shards-per-node`  | int   | _auto_    | Shards per node (s). Auto-set from dataset if omitted.              |
| `--classes-per-node` | int   | _auto_    | Classes per node (k). Auto-set from dataset if omitted.             |
| `--output-dir`       | str   | `results` | Directory for output files                                          |
| `--seed`             | int   | `42`      | Random seed for reproducibility                                     |

### Auto-detection

If `--shards-per-node` or `--classes-per-node` are not specified, they are automatically set to the paper's values based on the selected dataset:

| Dataset  | Auto s | Auto k |
| -------- | ------ | ------ |
| fmnist   | 12     | 4      |
| cifar10  | 12     | 4      |
| cifar100 | 100    | 20     |

---

## 7. Paper-Exact Experiment Configurations

These are the exact configurations used in the paper (Section V-A):

### Experiment 1: LeNet-5 + FMNIST

```bash
python scripts/run_local_simulation.py \
    --model lenet5 --dataset fmnist \
    --num-nodes 8 \
    --kappa-p 30 --kappa-e 1 --kappa-c 10 --kappa 50 \
    --gamma 2800 --lr 0.001 --batch-size 32 \
    --output-dir results/paper_lenet5_fmnist
```

### Experiment 2: MobileNetV2 + CIFAR-10

```bash
python scripts/run_local_simulation.py \
    --model mobilenetv2 --dataset cifar10 \
    --num-nodes 8 \
    --kappa-p 30 --kappa-e 1 --kappa-c 10 --kappa 50 \
    --gamma 2800 --lr 0.001 --batch-size 32 \
    --output-dir results/paper_mobilenetv2_cifar10
```

### Experiment 3: ResNet18 + CIFAR-100

```bash
python scripts/run_local_simulation.py \
    --model resnet18 --dataset cifar100 \
    --num-nodes 8 \
    --kappa-p 30 --kappa-e 1 --kappa-c 10 --kappa 50 \
    --gamma 2800 --lr 0.001 --batch-size 32 \
    --output-dir results/paper_resnet18_cifar100
```

### Paper Hyperparameters Reference

| Parameter          | Symbol | Paper Value | Notes                                      |
| ------------------ | ------ | ----------- | ------------------------------------------ |
| Pre-train epochs   | κ_p    | 30          | Used for similarity matrix                 |
| Local epochs       | κ_e    | 1-2         | Per edge round                             |
| Edge rounds        | κ_c    | 10          | Per cloud round                            |
| Cloud rounds       | κ      | 50-100      | Total                                      |
| Trade-off weight   | γ      | 2800        | Best for FMNIST; may need tuning for CIFAR |
| Batch size         | B      | 32          | —                                          |
| Learning rate      | η      | 0.001       | SGD with no momentum                       |
| LoS max iterations | T_max  | 30          | —                                          |

---

## 8. Code Architecture & File Reference

```
src/
├── config.py                    # Global configuration dataclasses
├── requirements.txt             # Python dependencies
├── USER_MANUAL.md              # This file
│
├── models/                      # Neural network architectures
│   ├── __init__.py             # Exports: get_model(), get_model_size(), model classes
│   ├── lenet5.py               # LeNet-5 for FMNIST (28×28, 1ch, 10 classes)
│   ├── mobilenetv2.py          # MobileNetV2 for CIFAR-10 (32×32, 3ch, 10 classes)
│   └── resnet18.py             # ResNet18 for CIFAR-100 (32×32, 3ch, 100 classes)
│
├── data/                        # Data loading and partitioning
│   ├── __init__.py
│   └── data_loader.py          # load_data(), create_non_iid_partitions(), DATASET_INFO
│
├── algorithms/                  # Core ShapeFL algorithms
│   ├── goa.py                  # GoA (Algorithm 1): Greedy node association
│   └── los.py                  # LoS (Algorithm 2): Local search edge selection
│
├── utils/                       # Utility functions
│   ├── aggregation.py          # FedAvg, weighted averaging, get_linear_layer_update()
│   ├── similarity.py           # Cosine similarity matrix computation
│   ├── communication.py        # HTTP helpers for distributed deployment
│   └── metrics.py              # Metric tracking utilities
│
├── scripts/                     # Runnable scripts
│   ├── run_local_simulation.py # ★ Main entry point for local simulation
│   ├── run_simulation.py       # Distributed simulation orchestrator
│   ├── orchestrate.py          # Orchestration helpers
│   ├── partition_data.py       # Standalone data partitioning script
│   ├── run_cloud.py            # Cloud server (Flask app)
│   ├── run_edge.py             # Edge aggregator (Flask app)
│   └── run_node.py             # Computing node (Flask app)
│
├── cloud/                       # Cloud server module
│   └── cloud_server.py         # CloudServer class for distributed mode
│
├── edge/                        # Edge aggregator module
│   └── edge_aggregator.py      # EdgeAggregator class
│
├── node/                        # Computing node module
│   └── computing_node.py       # ComputingNode class
│
├── dataset/                     # Dataset files (auto-populated)
│   ├── fashion-mnist_train.csv
│   ├── fashion-mnist_test.csv
│   └── ...                     # CIFAR data cached here on first download
│
├── results/                     # Simulation outputs (created at runtime)
├── checkpoints/                 # Model checkpoints
├── logs/                        # Log files
└── metrics/                     # Training metrics JSON
```

### Key Functions

| Function                               | File                   | Purpose                                            |
| -------------------------------------- | ---------------------- | -------------------------------------------------- |
| `get_model(name, num_classes, ...)`    | `models/lenet5.py`     | Factory — returns LeNet5, MobileNetV2, or ResNet18 |
| `load_data(dataset_name)`              | `data/data_loader.py`  | Loads FMNIST, CIFAR-10, or CIFAR-100               |
| `create_non_iid_partitions(...)`       | `data/data_loader.py`  | Creates shard-based non-IID splits                 |
| `run_goa(...)`                         | `algorithms/goa.py`    | GoA node association (Algorithm 1)                 |
| `run_los(...)`                         | `algorithms/los.py`    | LoS edge selection (Algorithm 2)                   |
| `get_linear_layer_update(new, old)`    | `utils/aggregation.py` | Extracts output-layer updates (generic)            |
| `compute_similarity_matrix(updates)`   | `utils/similarity.py`  | Cosine distance matrix S_ij                        |
| `federated_averaging(models, weights)` | `utils/aggregation.py` | Weighted FedAvg aggregation                        |

---

## 9. Understanding the Output

### Console Output

During a run you will see:

```
[Step 1] Initializing global model...
  mobilenetv2: 2,236,682 parameters, 8.532 MB

[Step 2] Loading CIFAR10 and creating non-IID partitions...
  Node 0: 180 samples, classes: {0: 45, 2: 45, 6: 45, 8: 45}
  ...

[Step 3] Pre-training phase (30 epochs per node)...
  Pre-training node 0... done (update norm: 1.2345)
  ...

[Step 4] Computing data distribution diversity matrix S_ij...
  Similarity matrix shape: (8, 8)
  Mean diversity: 0.8234, Max: 1.5678

[Step 5] Running LoS + GoA algorithms...
  Selected edge aggregators: [2, 5]
    Edge 2: nodes [0, 1, 2, 3] (data size: 720)
    Edge 5: nodes [4, 5, 6, 7] (data size: 720)

[Step 6] Starting hierarchical training (50 cloud rounds)...
  Round   1/50 | Acc: 0.2500 | Loss: 2.1000 | Local epochs: 10 | Time: 5.2s
  Round   2/50 | Acc: 0.3200 | Loss: 1.8500 | Local epochs: 20 | Time: 5.1s
  ...
```

### Output Files

After completion, the `--output-dir` folder contains:

| File                      | Content                                                                     |
| ------------------------- | --------------------------------------------------------------------------- |
| `simulation_results.json` | Full configuration, per-round accuracy/loss, final results, edge selections |
| `final_model.pt`          | Saved PyTorch state dict of the final global model                          |

### `simulation_results.json` structure

```json
{
  "config": {
    "model": "mobilenetv2",
    "dataset": "cifar10",
    "num_classes": 10,
    "num_nodes": 8,
    "kappa_p": 30,
    "kappa_e": 1,
    "kappa_c": 10,
    "kappa": 50,
    "gamma": 2800.0,
    "lr": 0.001,
    ...
  },
  "results": {
    "final_accuracy": 0.6523,
    "best_accuracy": 0.6701,
    "best_round": 47,
    "total_time_seconds": 1234.5
  },
  "selected_edges": [2, 5],
  "node_associations": {"0": 2, "1": 2, ...},
  "metrics": {
    "round": [1, 2, ...],
    "accuracy": [0.25, 0.32, ...],
    "loss": [2.1, 1.85, ...],
    "local_epochs": [10, 20, ...]
  }
}
```

---

## 10. Distributed Deployment (Raspberry Pi / LAN)

For real distributed deployment across physical devices:

### 1. Configure the network

Edit `config.py` to set IP addresses:

```python
NETWORK_CONFIG = NetworkConfig(
    cloud_host="192.168.0.100",  # Your laptop/desktop
    cloud_port=5000,
    edge_aggregators=[
        {"id": "edge_0", "host": "192.168.0.101", "port": 5001},
        {"id": "edge_1", "host": "192.168.0.102", "port": 5001},
    ],
    computing_nodes=[
        {"id": "node_0", "host": "192.168.0.111", "port": 5002},
        {"id": "node_1", "host": "192.168.0.112", "port": 5002},
        ...
    ],
)
```

### 2. Start the components

On the **cloud machine**:

```bash
python scripts/run_cloud.py
```

On each **edge aggregator**:

```bash
python scripts/run_edge.py --edge-id edge_0
```

On each **computing node**:

```bash
python scripts/run_node.py --node-id node_0
```

### 3. Orchestrate

```bash
python scripts/run_simulation.py
```

> **Note**: The local simulation (`run_local_simulation.py`) is recommended for development and experimentation. It runs everything in a single process and produces identical algorithmic results.

---

## 11. Troubleshooting

### Common Issues

| Problem                                        | Solution                                                                                                                 |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `ModuleNotFoundError: No module named 'torch'` | Install dependencies: `pip install -r requirements.txt`                                                                  |
| `CIFAR download fails`                         | Check internet connection. Or manually download from https://www.cs.toronto.edu/~kriz/cifar.html and place in `dataset/` |
| `VisibleDeprecationWarning` from torchvision   | Harmless with NumPy 2.4+. Can be ignored.                                                                                |
| `Out of memory` (ResNet18)                     | Reduce `--batch-size` to 16 or `--num-nodes` to 4                                                                        |
| Very slow training on CPU                      | Normal for MobileNetV2/ResNet18. Use GPU (`--device cuda`) or reduce rounds.                                             |
| Low accuracy after many rounds                 | Try `--lr 0.01` for small-scale experiments (fewer nodes, small data per node)                                           |
| `Unknown model: ...`                           | Use one of: `lenet5`, `mobilenetv2`, `resnet18`                                                                          |
| `Unknown dataset: ...`                         | Use one of: `fmnist`, `cifar10`, `cifar100`                                                                              |

### Runtime Estimates (CPU, 8 nodes)

| Model + Dataset        | 1 cloud round | 50 rounds |
| ---------------------- | ------------- | --------- |
| LeNet-5 + FMNIST       | ~2 s          | ~2 min    |
| MobileNetV2 + CIFAR-10 | ~90 s         | ~75 min   |
| ResNet18 + CIFAR-100   | ~180 s        | ~150 min  |

GPU acceleration (CUDA) typically provides 5-10× speedup.

---

_End of User Manual_
