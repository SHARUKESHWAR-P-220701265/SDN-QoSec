# SDN-QoSec — Software-Defined Quantum Network Simulation

A Python simulation of a **Software-Defined Quantum Network (SDQN)** that uses a dynamic *Quality-of-Security (QoSec)* routing protocol to prevent service blocking caused by key exhaustion in quantum mesh networks.

---

## What It Does

| Component | Description |
|---|---|
| **SDN Controller** | Central brain — maintains a live network map and routes traffic using the QoSec cost function |
| **Quantum Data Plane** | Simulates BB84 key generation; intermediate nodes relay keys via One-Time Pad (OTP) |
| **Eve (Adversary)** | Background thread that spikes QBER on random links every 100 ticks to simulate Intercept-Resend attacks |
| **Traffic Generator** | Generates Poisson-distributed (λ=2) key requests and dispatches them through the network |

### QoSec Cost Function

Routes are chosen by minimising the cost per link:

```
C(u,v) = α·(D_uv / D_max) + β·(K_threshold / K_curr) + γ·exp(η·QBER_uv)
```

Links with fewer than **50 bits** of key material are pruned from the routing graph entirely.

---

## Project Structure

```
SDN-QoSec/
├── sdn_controller.py        # SDN Controller — Global Knowledge Map + QoSec routing
├── quantum_data_plane.py    # Key buffers + BB84 generation + OTP relay
├── adversary.py             # Eve — Intercept-Resend attack thread
├── traffic_generator.py     # Poisson key-request generator
├── simulation.py            # Main entry point
├── tests/
│   └── test_sdn_controller.py
├── requirements.txt
├── setup.cfg
└── .env.example             # All tunable simulation parameters
```

---

## Quick Setup

### Prerequisites
- Python 3.9 or higher
- Git Bash / PowerShell / any terminal

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/SDN-QoSec.git
cd SDN-QoSec
```

### 2. Create and activate a virtual environment

```bash
# Create
python -m venv .venv

# Activate — Git Bash / macOS / Linux
source .venv/Scripts/activate

# Activate — PowerShell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the simulation

```bash
# Full 500-tick simulation
python simulation.py --ticks 500 --log INFO

# Quick 150-tick smoke test (WARNING-only output)
python simulation.py --ticks 150 --log WARNING
```

### 5. Run the test suite

```bash
python -m pytest tests/ -v
```

---

## Configuration

Copy `.env.example` to `.env` and adjust any parameter:

| Parameter | Default | Description |
|---|---|---|
| `QOSEC_ALPHA` | 0.4 | Weight — normalised link distance |
| `QOSEC_BETA` | 0.4 | Weight — inverse key-buffer level |
| `QOSEC_GAMMA` | 0.2 | Weight — QBER exponential penalty |
| `QOSEC_ETA` | 10 | Exponent multiplier for QBER term |
| `QOSEC_K_MIN` | 50 | Bits below which a link is pruned |
| `SIM_TICKS` | 500 | Total simulation ticks |
| `TRAFFIC_LAMBDA` | 2 | Poisson λ — mean requests per tick |
| `EVE_INTERVAL_TICKS` | 100 | Ticks between Eve attacks |
| `EVE_QBER_SPIKE` | 0.15 | QBER injected during an attack |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `networkx` | ≥ 3.1 | Graph topology + Dijkstra routing |
| `numpy` | ≥ 1.26 | Poisson distribution |
| `qunetsim` | 0.1.3.post1 | Quantum network simulation framework |
| `scipy` | ≥ 1.11 | Statistical helpers |
| `colorlog` | ≥ 6.7 | Coloured console output |
| `pytest` | ≥ 7.4 | Test runner |

---

## License

MIT
