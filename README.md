# SDN-QoSec — Software-Defined Quantum Network Simulation

A Python simulation of a **Software-Defined Quantum Network (SDQN)** that uses a dynamic *Quality-of-Security (QoSec)* routing protocol to prevent service blocking caused by key exhaustion in quantum mesh networks. Includes a real-time browser dashboard for visualisation.

---

## What It Does

| Component | Description |
|---|---|
| **SDN Controller** | Central brain — maintains a live network map and routes traffic using the QoSec cost function |
| **Quantum Data Plane** | Simulates BB84 key generation; intermediate nodes relay keys via One-Time Pad (OTP) |
| **Eve (Adversary)** | Background thread that spikes QBER on random links every 100 ticks to simulate Intercept-Resend attacks |
| **Traffic Generator** | Generates Poisson-distributed (λ=2) key requests and dispatches them through the network |
| **Qiskit Backend** | Dummy Qiskit integration layer demonstrating BB84 key generation via quantum circuits |
| **Sim Bridge** | Flask REST API that runs the simulation in a background thread and serves live state to the frontend |
| **Frontend Dashboard** | Single-page HTML/JS/CSS dashboard with D3.js network graph and Chart.js metrics |

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
├── simulation.py            # CLI entry point — headless simulation
├── sim_bridge.py            # Flask API bridge + dashboard server
├── qiskit_backend.py        # Dummy Qiskit BB84 integration
├── frontend/
│   ├── index.html           # Dashboard page
│   ├── app.js               # D3.js graph + Chart.js metrics logic
│   └── style.css            # Dashboard styling
├── tests/
│   ├── __init__.py
│   └── test_sdn_controller.py
├── requirements.txt
├── setup.cfg
├── .env.example             # All tunable simulation parameters
└── .gitignore
```

---

## Setup

### Prerequisites

- **Python 3.9** or higher
- **pip** (comes with Python)
- **Git** (to clone the repository)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/SDN-QoSec.git
cd SDN-QoSec
```

### 2. Create and activate a virtual environment

```bash
# Create
python -m venv .venv

# Activate — Git Bash / macOS / Linux
source .venv/Scripts/activate      # Windows Git Bash
source .venv/bin/activate          # macOS / Linux

# Activate — PowerShell
.venv\Scripts\Activate.ps1

# Activate — Command Prompt
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables (optional)

```bash
cp .env.example .env
```

Edit `.env` to tune simulation parameters. See the [Configuration](#configuration) section below for details.

---

## Running

### Option A — Interactive Dashboard (recommended)

Start the Flask server which runs the simulation and serves the live dashboard:

```bash
python sim_bridge.py
```

Then open **http://localhost:5050** in your browser.

| Flag | Default | Description |
|---|---|---|
| `--port` | `5050` | Server port |
| `--host` | `127.0.0.1` | Bind address |

Example with custom port:

```bash
python sim_bridge.py --port 8080
```

Use the dashboard controls to **Start**, **Pause**, **Step**, or **Reset** the simulation in real time.

### Option B — Headless CLI Simulation

Run the simulation entirely in the terminal (no browser needed):

```bash
# Full 500-tick simulation
python simulation.py --ticks 500 --log INFO

# Quick 150-tick smoke test (WARNING-only output)
python simulation.py --ticks 150 --log WARNING
```

| Flag | Default | Description |
|---|---|---|
| `--ticks` | `500` | Total simulation ticks |
| `--dt` | `0.01` | Tick duration in seconds |
| `--log` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `--seed` | `42` | RNG seed for reproducibility |

### Running Tests

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
| `QOSEC_K_THRESHOLD` | 200 | Target key reserve (bits) for β term |
| `QOSEC_K_MIN` | 50 | Bits below which a link is pruned |
| `SIM_TICKS` | 500 | Total simulation ticks |
| `SIM_TICK_MS` | 10 | Wall-clock ms per tick (real-time pacing) |
| `KEY_GEN_RATE_BPS` | 10000 | Base quantum key generation rate (bps) |
| `KEY_BUFFER_MAX` | 5000 | Maximum key buffer per link (bits) |
| `TRAFFIC_LAMBDA` | 2 | Poisson λ — mean requests per tick |
| `EVE_INTERVAL_TICKS` | 100 | Ticks between Eve attacks |
| `EVE_QBER_SPIKE` | 0.15 | QBER injected by Intercept-Resend attack |
| `LOG_LEVEL` | INFO | Console log level |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `networkx` | ≥ 3.1 | Graph topology + Dijkstra routing |
| `numpy` | ≥ 1.26 | Poisson distribution |
| `qunetsim` | 0.1.3.post1 | Quantum network simulation framework |
| `scipy` | ≥ 1.11 | Statistical helpers |
| `colorlog` | ≥ 6.7 | Coloured console output |
| `flask` | ≥ 3.0 | REST API bridge for the dashboard |
| `flask-cors` | ≥ 4.0 | Cross-origin support for local dev |
| `pytest` | ≥ 7.4 | Test runner |
| `pytest-timeout` | ≥ 2.1 | Guard against hung threads in tests |

---

## License

MIT
