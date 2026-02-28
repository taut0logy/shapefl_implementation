# ShapeFL: Communication-Efficient Hierarchical Federated Learning

A complete implementation of the ShapeFL (Shaping Data Distribution at Edge for Communication-Efficient Hierarchical Federated Learning) framework using Raspberry Pi devices as edge nodes.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

ShapeFL is a hierarchical federated learning (HFL) framework that significantly reduces communication costs while maintaining high model accuracy. This implementation follows the methodology described in the paper:

> Y. Deng et al., "A Communication-Efficient Hierarchical Federated Learning Framework via Shaping Data Distribution at Edge," in IEEE/ACM Transactions on Networking, vol. 32, no. 3, pp. 2600-2615, June 2024, doi: 10.1109/TNET.2024.3363916.

**Keywords**: Federated learning, communication efficiency, edge computing, distributed edge intelligence, data distribution shaping

### Key Features

- **Hierarchical Architecture**: Three-tier structure (Cloud Server → Edge Aggregators → Computing Nodes)
- **Communication Optimization**: Up to 70% reduction in communication cost compared to traditional FL
- **Data Distribution Shaping**: LoS + GoA algorithms for optimal edge selection and node association
- **Non-IID Data Support**: Handles imbalanced and heterogeneous data distributions
- **Multiple Models & Datasets**: LeNet-5 + FMNIST, MobileNetV2 + CIFAR-10, ResNet18 + CIFAR-100
- **Configurable Simulation**: Command-line arguments for model, dataset, and all hyperparameters
- **Hardware Implementation**: Designed for deployment on Raspberry Pi 4 clusters

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Cloud Server                            │
│              (Laptop/Desktop with GPU)                       │
│   - Global Model Aggregation                                 │
│   - Edge Selection (LoS Algorithm)                           │
│   - Node Association (GoA Algorithm)                         │
└────────────────┬────────────────┬──────────────┬─────────────┘
                 │                │              │
        ┌────────▼─────┐  ┌───────▼─────┐  ┌─────▼────────┐
        │ Edge Agg 0   │  │ Edge Agg 1  │  │ Edge Agg 2   │
        │ (Raspberry   │  │ (Raspberry  │  │ (Raspberry   │
        │  Pi 4)       │  │  Pi 4)      │  │  Pi 4)       │
        └┬────────────┬┘  └┬───────────┬┘  └─┬────────────┘
         │            │    │           │     │
    ┌────▼──┐    ┌────▼──┐ │      ┌────▼──┐  │
    │Node 0 │    │Node 1 │ │      │Node 3 │  │
    │(Pi 4) │    │(Pi 4) │ │      │(Pi 4) │  │
    └───────┘    └───────┘ │      └───────┘  │
                      ┌────▼──┐         ┌────▼──┐
                      │Node 2 │         │Node 4 │
                      │(Pi 4) │         │(Pi 4) │
                      └───────┘         └───────┘

Note: Node-to-edge assignment is determined dynamically by the GoA algorithm
based on communication cost and data distribution similarity.
```

## How ShapeFL Works

### 1. Offline Pre-training Phase

- Each node performs local training (κₚ = 30 epochs)
- Nodes send linear layer updates (Δw^(L)) to cloud server
- Cloud computes data distribution similarity matrix: S_ij = 1 - cos(Δw_i^(L), Δw_j^(L))

### 2. Edge Selection & Node Association

- **LoS (Local Search)**: Selects optimal edge aggregators using open/close/swap operations
- **GoA (Greedy Node Association)**: Assigns nodes to edges minimizing:
    - J = κₑ·c_ne - γ·S_ne (communication cost vs. data diversity trade-off)

### 3. Hierarchical Training

- For each communication round (1 to κ):
    - Nodes train locally for κₑ epochs
    - Edge aggregators collect and aggregate node updates (FedAvg)
    - After κc edge epochs, edges send to cloud server
    - Cloud performs global aggregation and broadcasts

## Hardware Requirements

### Cloud Server (Laptop/Desktop)

- **CPU**: Intel i5 or equivalent (4+ cores)
- **RAM**: 8 GB minimum, 16 GB recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: 256 GB SSD
- **OS**: Ubuntu 24.04 LTS or Windows 10/11

### Edge Aggregators & Computing Nodes (Raspberry Pi 4)

- **Model**: Raspberry Pi 4 Model B
- **CPU**: ARM Cortex-A72 (1.5 GHz)
- **RAM**: 4-8 GB
- **Storage**: 32 GB MicroSD (Class 10)
- **Network**: Gigabit Ethernet or Wi-Fi
- **Quantity**: 8 devices (3 edge aggregators + 5 computing nodes)

### Network

- Local router or Ethernet switch
- All devices on the same LAN subnet (e.g., 192.168.0.0/24)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/taut0logy/shapefl_implementation.git
cd shapefl_implementation
```

### 2. Setup on Cloud Server

#### Linux (Ubuntu 24.04)

```bash
# Create virtual environment
cd src
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For GPU support (if you have NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Windows

```powershell
# Create virtual environment
cd src
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# For GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Setup on Raspberry Pi (Ubuntu Server 24.04)

```bash
# On each Raspberry Pi
mkdir -p /home/pi/shapefl
cd /home/pi/shapefl
python3 -m venv .venv
source .venv/bin/activate

# Install CPU-only PyTorch for ARM
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install flask requests numpy pandas
```

## Configuration

Edit `src/config.py` to match your network setup:

```python
@dataclass
class NetworkConfig:
    # Cloud server IP address
    cloud_host: str = "192.168.0.100"  # Your laptop's IP
    cloud_port: int = 5000

    # Edge aggregators
    edge_aggregators: List[Dict] = field(default_factory=lambda: [
        {"id": "edge_0", "host": "192.168.0.101", "port": 5001},
        {"id": "edge_1", "host": "192.168.0.102", "port": 5001},
        {"id": "edge_2", "host": "192.168.0.103", "port": 5001},
    ])

    # Computing nodes
    computing_nodes: List[Dict] = field(default_factory=lambda: [
        {"id": "node_0", "host": "192.168.0.111", "port": 5002},
        {"id": "node_1", "host": "192.168.0.112", "port": 5002},
        {"id": "node_2", "host": "192.168.0.113", "port": 5002},
        {"id": "node_3", "host": "192.168.0.114", "port": 5002},
        {"id": "node_4", "host": "192.168.0.115", "port": 5002},
    ])
```

### Hyperparameters (from Paper)

| Parameter  | Value | Description                           |
| ---------- | ----- | ------------------------------------- |
| κ_p        | 30    | Pre-training epochs per node          |
| κ_e        | 1     | Local epochs before edge aggregation  |
| κ_c        | 10    | Edge epochs before cloud aggregation  |
| κ          | 50    | Total communication rounds            |
| γ          | 2800  | Trade-off weight (optimal from paper) |
| B_e        | 10    | Max nodes per edge aggregator         |
| T_max      | 30    | Max LoS iterations                    |
| batch_size | 32    | Training batch size                   |
| lr         | 0.001 | Learning rate (SGD)                   |

#### Dataset-Specific Partitioning (from Paper)

| Dataset   | Shards/node (s) | Classes/node (k) |
| --------- | --------------- | ---------------- |
| FMNIST    | 12              | 4                |
| CIFAR-10  | 12              | 4                |
| CIFAR-100 | 100             | 20               |

> Partitioning parameters are auto-detected from the dataset when using `run_local_simulation.py`.

## Quick Start

### Local Simulation (Recommended for Development)

The local simulation runs the entire ShapeFL pipeline (Algorithm 3) in a single process — no Flask servers or network setup needed.

```bash
cd src
source .venv/bin/activate  # Linux
# .\.venv\Scripts\Activate.ps1  # Windows

# LeNet-5 + Fashion-MNIST (fastest)
python scripts/run_local_simulation.py --model lenet5 --dataset fmnist

# MobileNetV2 + CIFAR-10
python scripts/run_local_simulation.py --model mobilenetv2 --dataset cifar10

# ResNet18 + CIFAR-100
python scripts/run_local_simulation.py --model resnet18 --dataset cifar100
```

All three model/dataset combinations from the paper are supported. CIFAR datasets are auto-downloaded on first run via torchvision.

#### Customising Parameters

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

#### All Command-Line Arguments

| Argument             | Type  | Default   | Description                                  |
| -------------------- | ----- | --------- | -------------------------------------------- |
| `--model`            | str   | `lenet5`  | Model: `lenet5`, `mobilenetv2`, `resnet18`   |
| `--dataset`          | str   | `fmnist`  | Dataset: `fmnist`, `cifar10`, `cifar100`     |
| `--num-nodes`        | int   | `8`       | Total computing nodes                        |
| `--kappa-p`          | int   | `30`      | Pre-training epochs per node                 |
| `--kappa-e`          | int   | `1`       | Local epochs per edge round                  |
| `--kappa-c`          | int   | `10`      | Edge rounds per cloud round                  |
| `--kappa`            | int   | `50`      | Total cloud aggregation rounds               |
| `--gamma`            | float | `2800.0`  | Trade-off weight γ                           |
| `--B-e`              | int   | `10`      | Max nodes per edge aggregator                |
| `--T-max`            | int   | `30`      | Max LoS iterations                           |
| `--lr`               | float | `0.001`   | Learning rate (SGD)                          |
| `--batch-size`       | int   | `32`      | Mini-batch size                              |
| `--shard-size`       | int   | `15`      | Shard size for non-IID partitioning          |
| `--shards-per-node`  | int   | _auto_    | Shards per node (s). Auto-set from dataset.  |
| `--classes-per-node` | int   | _auto_    | Classes per node (k). Auto-set from dataset. |
| `--output-dir`       | str   | `results` | Output directory                             |
| `--seed`             | int   | `42`      | Random seed                                  |

### Distributed Simulation (Localhost)

Test the Flask-based distributed mode on your laptop before deploying to Raspberry Pis:

```bash
cd src
source .venv/bin/activate  # Linux
# .\.venv\Scripts\Activate.ps1  # Windows

python scripts/run_simulation.py --num-edges 3 --num-nodes 5
```

In another terminal, verify all components registered:

```bash
curl http://127.0.0.1:5000/status
```

Expected output:

```json
{
    "registered_edges": 3,
    "registered_nodes": 5,
    "current_round": 0,
    "pretrain_complete": false,
    "selected_edges": []
}
```

Press `Ctrl+C` to stop the simulation.

### Deploy to Raspberry Pi Cluster

#### Step 1: Prepare Data Partitions

On the cloud server:

```bash
cd src
source .venv/bin/activate

# Create non-IID data partitions
python scripts/partition_data.py \
    --num-nodes 5 \
    --shard-size 15 \
    --shards-per-node 12 \
    --classes-per-node 4
```

This creates `partitions/partitions.json` with non-IID data assignments.

#### Step 2: Transfer Code and Data to Raspberry Pis

```bash
# Deploy code to all Pis
PIES="192.168.0.101 192.168.0.102 192.168.0.103 192.168.0.111 192.168.0.112 192.168.0.113 192.168.0.114 192.168.0.115"

for PI in $PIES; do
    echo "Deploying to $PI..."
    rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
        src/ pi@$PI:/home/pi/shapefl/
done

# Copy dataset and partitions to computing nodes only
for NODE in 192.168.0.{111..115}; do
    scp ../dataset/*.csv pi@$NODE:/home/pi/shapefl/dataset/
    scp partitions/partitions.json pi@$NODE:/home/pi/shapefl/partitions/
done
```

#### Step 3: Start Components

**1. Cloud Server (your laptop):**

```bash
cd src
source .venv/bin/activate  # Linux
python -m cloud.cloud_server --host 0.0.0.0 --port 5000
```

**2. Edge Aggregators (each Pi):**

```bash
# SSH into edge aggregator Pi
ssh pi@192.168.0.101

cd /home/pi/shapefl
source .venv/bin/activate
python -m edge.edge_aggregator \
    --edge-id edge_0 \
    --host 0.0.0.0 \
    --port 5001 \
    --cloud-host 192.168.0.100 \
    --cloud-port 5000
```

Repeat for edge1 (`edge_1` on 192.168.0.102) and edge2 (`edge_2` on 192.168.0.103).

**3. Computing Nodes (each Pi):**

> **Important:** Do NOT specify `--edge-host`. The GoA algorithm dynamically assigns nodes to edges based on optimization.

```bash
# SSH into computing node Pi
ssh pi@192.168.0.111

cd /home/pi/shapefl
source .venv/bin/activate
python -m node.computing_node \
    --node-id node_0 \
    --host 0.0.0.0 \
    --port 5002 \
    --cloud-host 192.168.0.100 \
    --cloud-port 5000 \
    --partitions-file partitions/partitions.json
```

Repeat for all 5 nodes (`node_0` through `node_4`).

#### Step 4: Run the Orchestrator

Once all components are registered, run the orchestrator on the cloud server:

```bash
cd src
source .venv/bin/activate

python scripts/orchestrate.py \
    --cloud-host localhost \
    --cloud-port 5000 \
    --num-rounds 100
```

The orchestrator automates:

1. ✅ Waiting for all registrations
2. ✅ Triggering pre-training (κ_p = 30 epochs)
3. ✅ Computing similarity matrix S_ij
4. ✅ Running LoS algorithm (edge selection)
5. ✅ Running GoA algorithm (node-to-edge assignment)
6. ✅ Starting hierarchical training loop

#### Step 5: Monitor Training

```bash
# Check status
curl http://192.168.0.100:5000/status

# View latest metrics
curl http://192.168.0.100:5000/metrics/latest

# View all metrics history
curl http://192.168.0.100:5000/metrics/all
```

## Project Structure

```
src/
├── config.py                         # Configuration & hyperparameters
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── USER_MANUAL.md                    # Detailed user manual
│
├── models/                           # Neural network architectures
│   ├── __init__.py                   # Exports all models and factory
│   ├── factory.py                    # Model factory: get_model(), get_model_size(), serialisation
│   ├── lenet5.py                     # LeNet-5 for FMNIST (1×28×28, 61,706 params)
│   ├── mobilenetv2.py                # MobileNetV2 for CIFAR-10 (3×32×32, 2,236,682 params)
│   └── resnet18.py                   # ResNet18 for CIFAR-100 (3×32×32, 11,220,132 params)
│
├── data/
│   ├── __init__.py
│   └── data_loader.py                # load_data(), CIFAR-10/100 loaders, non-IID partitioning
│
├── algorithms/
│   ├── __init__.py
│   ├── goa.py                        # Algorithm 1: Greedy Node Association (GoA)
│   └── los.py                        # Algorithm 2: Local Search Edge Selection (LoS)
│
├── cloud/
│   ├── __init__.py
│   └── cloud_server.py               # Cloud server (Flask) + global aggregation
│
├── edge/
│   ├── __init__.py
│   └── edge_aggregator.py            # Edge aggregator (Flask)
│
├── node/
│   ├── __init__.py
│   └── computing_node.py             # Computing node (Flask) + local training
│
├── utils/
│   ├── __init__.py
│   ├── aggregation.py                # FedAvg, weighted averaging, generic get_linear_layer_update()
│   ├── similarity.py                 # S_ij = 1 − cos(Δw_i^(L), Δw_j^(L))
│   ├── communication.py              # HTTP model transfer helpers
│   └── metrics.py                    # MetricsTracker
│
├── scripts/
│   ├── run_local_simulation.py       # ★ Main entry point: single-process simulation
│   ├── run_simulation.py             # Distributed localhost simulation
│   ├── orchestrate.py                # Distributed training orchestrator
│   ├── partition_data.py             # Standalone data partitioning
│   ├── run_cloud.py                  # Start cloud server
│   ├── run_edge.py                   # Start edge aggregator
│   └── run_node.py                   # Start computing node
│
├── dataset/                          # Dataset files (auto-populated)
│   ├── fashion-mnist_train.csv       # FMNIST training data (bundled)
│   ├── fashion-mnist_test.csv        # FMNIST test data (bundled)
│   ├── cifar-10-batches-py/          # CIFAR-10 (auto-downloaded)
│   └── cifar-100-python/             # CIFAR-100 (auto-downloaded)
│
├── results/                          # Simulation outputs (generated)
├── partitions/                       # Data partitions (generated)
├── checkpoints/                      # Model checkpoints (generated)
├── logs/                             # Log files (generated)
└── metrics/                          # Training metrics (generated)
```

## Experiments and Results

### Supported Model / Dataset Combinations

All three pairings evaluated in the paper are implemented:

| #   | Model           | Dataset       | Classes | Input Size  | Parameters | Partitioning (s, k) |
| --- | --------------- | ------------- | ------- | ----------- | ---------- | ------------------- |
| 1   | **LeNet-5**     | Fashion-MNIST | 10      | 1 × 28 × 28 | 61,706     | s=12, k=4           |
| 2   | **MobileNetV2** | CIFAR-10      | 10      | 3 × 32 × 32 | 2,236,682  | s=12, k=4           |
| 3   | **ResNet18**    | CIFAR-100     | 100     | 3 × 32 × 32 | 11,220,132 | s=100, k=20         |

### Datasets

| Dataset           | Train                    | Test   | Download                                            |
| ----------------- | ------------------------ | ------ | --------------------------------------------------- |
| **Fashion-MNIST** | 60,000 (28×28 grayscale) | 10,000 | Bundled as CSV; also auto-downloads via torchvision |
| **CIFAR-10**      | 50,000 (32×32 colour)    | 10,000 | Auto-downloaded (~170 MB) on first run              |
| **CIFAR-100**     | 50,000 (32×32 colour)    | 10,000 | Auto-downloaded (~170 MB) on first run              |

### Models

**LeNet-5** — Conv(6) → Pool → Conv(16) → Pool → FC(120) → FC(84) → FC(10). Linear layer `fc3` (850 params) used for similarity.

**MobileNetV2** — Inverted residual blocks, adapted for 32×32 input (stride-1 first conv). Classifier layer `classifier` (12,810 params) used for similarity.

**ResNet18** — BasicBlock ×[2,2,2,2], adapted for 32×32 input (3×3 first conv, no maxpool). Linear layer `linear` (51,300 params) used for similarity.

### Non-IID Data Partitioning

Following the paper's methodology:

- The training dataset is divided into shards of size 15
- Each computing node receives _s_ shards from _k_ classes
- For FMNIST / CIFAR-10: 12 shards from 4 classes (180 samples/node)
- For CIFAR-100: 100 shards from 20 classes (1,500 samples/node)

### Expected Performance (from Paper)

Compared to baseline HierFAVG:

- **Communication Cost**: ~45% reduction (FMNIST), significant reductions on CIFAR-10/100
- **Accuracy**: +3-5% improvement due to better data distribution at edges
- **Convergence**: Faster convergence with fewer communication rounds

### Simulation Output

Each simulation run produces:

- `simulation_results.json` — full config, per-round accuracy/loss, edge selections, node associations
- `final_model.pt` — saved PyTorch state dict of the final global model

## Monitoring and Evaluation

### API Endpoints

| Endpoint          | Method | Description                             |
| ----------------- | ------ | --------------------------------------- |
| `/status`         | GET    | Get registration and training status    |
| `/health`         | GET    | Health check                            |
| `/metrics/latest` | GET    | Get most recent training metrics        |
| `/metrics/all`    | GET    | Get full metrics history                |
| `/config/edges`   | GET    | Get edge selection and node assignments |
| `/algorithms/run` | POST   | Trigger LoS + GoA execution             |
| `/training/start` | POST   | Start hierarchical training loop        |
| `/pretrain/start` | POST   | Trigger pre-training on nodes           |
| `/edge/assign`    | POST   | Assign node to edge aggregator          |

### Metrics Export

Simulation results are saved in `<output-dir>/simulation_results.json`:

```json
{
  "config": {
    "model": "lenet5",
    "dataset": "fmnist",
    "num_classes": 10,
    "num_nodes": 8,
    "kappa_p": 30,
    "kappa_e": 1,
    "kappa_c": 10,
    "kappa": 50,
    "gamma": 2800.0,
    "lr": 0.001,
    "batch_size": 32
  },
  "results": {
    "final_accuracy": 0.6523,
    "best_accuracy": 0.6701,
    "best_round": 47,
    "total_time_seconds": 120.5
  },
  "selected_edges": [2, 5],
  "node_associations": {"0": 2, "1": 2, "2": 5, ...},
  "metrics": {
    "round": [1, 2, 3, ...],
    "accuracy": [0.25, 0.32, 0.38, ...],
    "loss": [2.1, 1.85, 1.6, ...],
    "local_epochs": [10, 20, 30, ...]
  }
}
```

### Visualization

```python
import json
import matplotlib.pyplot as plt

with open('results/simulation_results.json', 'r') as f:
    data = json.load(f)

metrics = data['metrics']

# Plot accuracy over cloud rounds
plt.figure(figsize=(10, 6))
plt.plot(metrics['round'], metrics['accuracy'], marker='o')
plt.xlabel('Cloud Round')
plt.ylabel('Test Accuracy')
plt.title(f"ShapeFL: {data['config']['model']} + {data['config']['dataset']}")
plt.grid(True)
plt.savefig('accuracy_curve.png', dpi=300)
```

## Troubleshooting

### Common Issues

**1. Connection Refused**

```bash
# Check if firewall is blocking ports
sudo ufw allow 5000:5300/tcp

# Verify device is reachable
ping 192.168.0.101
```

**2. Out of Memory on Raspberry Pi**

```python
# Reduce batch size
batch_size: int = 16  # Instead of 32
```

ResNet18 + CIFAR-100 is large (~43 MB). Consider using LeNet-5 + FMNIST for Raspberry Pi deployments.

**3. Slow Training**

- MobileNetV2 and ResNet18 are significantly slower than LeNet-5 on CPU
- Use GPU (`torch.cuda`) when available — the simulation auto-detects CUDA
- Use Ethernet instead of Wi-Fi for distributed mode
- Check Pi temperature: `vcgencmd measure_temp`

**4. Model Mismatch Errors**

```bash
# Ensure same PyTorch version on all devices
pip list | grep torch
```

**5. Pre-training Timeout**

- Increase timeout in `orchestrate.py`
- Check if all nodes have data partitions

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this implementation in your research, please cite the original ShapeFL paper:

```bibtex
@ARTICLE{deng2024shapefl,
  author={Deng, Yongheng and Lyu, Feng and Xia, Tengyi and Zhou, Yuezhi and Zhang, Yaoxue and Ren, Ju and Yang, Yuanyuan},
  journal={IEEE/ACM Transactions on Networking},
  title={A Communication-Efficient Hierarchical Federated Learning Framework via Shaping Data Distribution at Edge},
  year={2024},
  volume={32},
  number={3},
  pages={2600-2615},
  doi={10.1109/TNET.2024.3363916},
  keywords={Costs;Data models;Servers;Computational modeling;Training data;Federated learning;Distributed databases;Hierarchical federated learning;communication efficiency;edge computing;distributed edge intelligence}
}
```

## 🔗 References

1. Y. Deng et al., "A Communication-Efficient Hierarchical Federated Learning Framework via Shaping Data Distribution at Edge," in IEEE/ACM Transactions on Networking, vol. 32, no. 3, pp. 2600-2615, June 2024, doi: 10.1109/TNET.2024.3363916.

2. H. Brendan McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," in AISTATS, 2017.

3. Y. LeCun et al., "Gradient-based learning applied to document recognition," in Proceedings of the IEEE, 1998.

4. M. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks," in CVPR, 2018.

5. K. He et al., "Deep Residual Learning for Image Recognition," in CVPR, 2016.

6. Fashion-MNIST Dataset: https://github.com/zalandoresearch/fashion-mnist

7. CIFAR-10/100 Datasets: https://www.cs.toronto.edu/~kriz/cifar.html

## 👥 Authors

- **Raufun Ahsan** - Khulna University of Engineering & Technology
- **Md. Sakibur Rahman** - Khulna University of Engineering & Technology

## Acknowledgments

- Original ShapeFL paper authors for the groundbreaking research
- PyTorch team for the excellent deep learning framework
- Raspberry Pi Foundation for affordable edge computing hardware
- Fashion-MNIST and CIFAR dataset creators for providing benchmark datasets

## Contact

For questions or support, please open an issue on GitHub or contact:

- Email: raufun.ahsan@gmail.com
- GitHub: [@taut0logy](https://github.com/taut0logy)
- Email: sakiburrahman11.msr@gmail.com
- GitHub: [@SakiburRahman07](https://github.com/SakiburRahman07)

---

**Note**: This is a research implementation. For production use, additional security, error handling, and optimization may be required.
