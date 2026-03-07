"""
sim_bridge.py
=============
Flask-based bridge between the SDQN simulation engine and the frontend
dashboard.  Runs the simulation in a background thread and exposes live
state via a JSON REST API.

Usage
-----
    python sim_bridge.py              # starts on http://localhost:5050
    python sim_bridge.py --port 8080  # custom port
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from sdn_controller import SDN_Controller, TOPOLOGY_EDGES, TOPOLOGY_NODES
from quantum_data_plane import QuantumDataPlane
from adversary import Eve
from traffic_generator import TrafficGenerator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simulation wrapper
# ---------------------------------------------------------------------------
SIM_TICKS_DEFAULT = 500
TICK_DT = 0.01
TELEMETRY_INTERVAL = 5
MAX_HISTORY = 500
MAX_TRAFFIC_LOG = 50


class SimState:
    """Thread-safe container for all simulation state."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.tick: int = 0
        self.running: bool = False
        self.speed_ms: int = 100  # ms pause between ticks (UI slider)


        # Components
        self.controller: Optional[SDN_Controller] = None
        self.qdp: Optional[QuantumDataPlane] = None
        self.eve: Optional[Eve] = None
        self.generator: Optional[TrafficGenerator] = None

        # Chaos / stress-testing multipliers (1.0 = normal)
        self.traffic_mult: float = 1.0   # 1×–8×
        self.eve_aggression: float = 1.0  # 1×–10×
        self.keygen_pct: float = 100.0    # 100%–10%

        # Base values (captured at init so multipliers apply cleanly)
        self._base_lam: float = 2.0
        self._base_eve_interval: int = 100
        self._base_qber_spike: float = 0.15
        self._base_rate_bps: float = 10_000.0

        # Time-series history
        self.ts_ticks: deque = deque(maxlen=MAX_HISTORY)
        self.ts_blocking: deque = deque(maxlen=MAX_HISTORY)
        self.ts_avg_kcurr: deque = deque(maxlen=MAX_HISTORY)
        self.ts_max_qber: deque = deque(maxlen=MAX_HISTORY)
        self.ts_total_requests: deque = deque(maxlen=MAX_HISTORY)
        self.ts_successful: deque = deque(maxlen=MAX_HISTORY)
        self.ts_failed: deque = deque(maxlen=MAX_HISTORY)

        self._init_components()

    def _init_components(self) -> None:
        edges = [(u, v) for u, v, _ in TOPOLOGY_EDGES]
        self.controller = SDN_Controller()
        self.qdp = QuantumDataPlane(edges=edges)

        eve_stop = threading.Event()
        self.eve = Eve(
            links=edges,
            set_qber_fn=self.qdp.set_qber,
            update_telemetry=self.controller.update_telemetry,
            get_k_curr_fn=self.qdp.get_k_curr,
            attack_interval=100,
            qber_spike=0.15,
            stop_event=eve_stop,
        )
        self.eve.start()

        self.generator = TrafficGenerator(
            nodes=list(self.controller.graph.nodes()),
            compute_path_fn=self.controller.compute_qosec_path,
            relay_key_fn=self.qdp.relay_key,
            lam=2.0,
            key_size_bits=256.0,
            rng_seed=42,
        )

    def _apply_chaos(self) -> None:
        """Apply current chaos multipliers to simulation components.

        Scaling curves are chosen to feel realistic:
        - Traffic: quadratic — gentle early ramp, aggressive at high end
        - Eve:    exponential — attacks escalate sharply
        - KeyGen: linear percentage — intuitive and predictable
        """
        if self.generator:
            # Quadratic: mult 1→1×, 4→10×, 8→~37× effective requests
            # Actually keep it practical: direct multiply, the quadratic
            # feel comes from the Poisson variance scaling with lambda.
            self.generator.lam = self._base_lam * self.traffic_mult

        if self.eve:
            # Interval shrinks: 100/1=100, 100/3≈33, 100/10=10 ticks
            self.eve.attack_interval = max(10, int(
                self._base_eve_interval / self.eve_aggression
            ))
            # QBER spike grows sub-linearly: 0.15 → ~0.28 at 5× → ~0.40 at 10×
            # Uses sqrt for realistic feel — doubling aggression doesn't
            # double the physical QBER an eavesdropper can inject.
            self.eve.qber_spike = min(
                0.45,
                self._base_qber_spike * math.sqrt(self.eve_aggression),
            )

        if self.qdp:
            self.qdp.rate_bps = self._base_rate_bps * (self.keygen_pct / 100.0)

    def reset(self) -> None:
        with self._lock:
            if self.eve:
                self.eve.stop()
            self.tick = 0
            self.running = False
            self.traffic_mult = 1.0
            self.eve_aggression = 1.0
            self.keygen_pct = 100.0
            self.ts_ticks.clear()
            self.ts_blocking.clear()
            self.ts_avg_kcurr.clear()
            self.ts_max_qber.clear()
            self.ts_total_requests.clear()
            self.ts_successful.clear()
            self.ts_failed.clear()
            self._init_components()

    def step(self) -> None:
        """Advance one tick."""
        with self._lock:
            self.tick += 1
            t = self.tick

            # 0. Apply live chaos multipliers
            self._apply_chaos()

            # 1. Quantum key generation
            self.qdp.tick(dt=TICK_DT)

            # 2. Traffic
            self.generator.tick(tick_id=t)

            # 3. Telemetry push
            if t % TELEMETRY_INTERVAL == 0:
                for u, v, k_curr, qber in self.qdp.all_link_states():
                    self.controller.update_telemetry(u, v, k_curr, qber)

            # 4. Eve
            self.eve.attack_if_due(t)

            # 5. Collect time-series
            states = self.qdp.all_link_states()
            avg_k = sum(s[2] for s in states) / len(states) if states else 0
            max_q = max(s[3] for s in states) if states else 0

            self.ts_ticks.append(t)
            self.ts_blocking.append(self.generator.blocking_rate)
            self.ts_avg_kcurr.append(round(avg_k, 1))
            self.ts_max_qber.append(round(max_q, 4))
            self.ts_total_requests.append(self.generator._total_requests)
            self.ts_successful.append(self.generator._successful)
            self.ts_failed.append(self.generator._failed)

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            snapshot = self.controller.get_topology_snapshot()
            links = []
            for ls in snapshot["links"]:
                links.append({
                    "u": ls.u,
                    "v": ls.v,
                    "distance": ls.distance,
                    "K_curr": round(ls.K_curr, 1),
                    "QBER": round(ls.QBER, 4),
                    "qosec_cost": round(ls.qosec_cost, 4) if not math.isinf(ls.qosec_cost) else "inf",
                    "pruned": math.isinf(ls.qosec_cost),
                })

            # Recent traffic
            recent_traffic = []
            for req in self.generator.request_log[-MAX_TRAFFIC_LOG:]:
                recent_traffic.append({
                    "tick": req.tick,
                    "src": req.src,
                    "dst": req.dst,
                    "key_bits": req.key_bits,
                    "path": req.path,
                    "success": req.success,
                    "failure_reason": req.failure_reason,
                })

            return {
                "tick": self.tick,
                "running": self.running,
                "speed_ms": self.speed_ms,
                "nodes": snapshot["nodes"],
                "links": links,
                "d_max_km": snapshot["d_max_km"],
                "attack_log": self.eve.attack_log if self.eve else [],
                "traffic": recent_traffic,
                "blocking_rate": round(self.generator.blocking_rate, 4),
                "total_requests": self.generator._total_requests,
                "successful": self.generator._successful,
                "failed": self.generator._failed,
                "params": {
                    "alpha": self.controller.alpha,
                    "beta": self.controller.beta,
                    "gamma": self.controller.gamma,
                    "eta": self.controller.eta,
                    "k_min": self.controller.k_min,
                    "k_threshold": self.controller.k_threshold,
                },
                "chaos": {
                    "traffic_mult": self.traffic_mult,
                    "eve_aggression": self.eve_aggression,
                    "keygen_pct": self.keygen_pct,
                },
                "timeseries": {
                    "ticks": list(self.ts_ticks),
                    "blocking": list(self.ts_blocking),
                    "avg_kcurr": list(self.ts_avg_kcurr),
                    "max_qber": list(self.ts_max_qber),
                    "total_requests": list(self.ts_total_requests),
                    "successful": list(self.ts_successful),
                    "failed": list(self.ts_failed),
                },
            }


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
sim = SimState()
sim_thread: Optional[threading.Thread] = None

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)


def _sim_loop() -> None:
    """Background loop that advances the simulation while running."""
    while sim.running:
        sim.step()
        time.sleep(sim.speed_ms / 1000.0)


@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/state")
def api_state():
    return jsonify(sim.get_state())


@app.route("/api/control", methods=["POST"])
def api_control():
    global sim_thread
    data = request.get_json(force=True)
    action = data.get("action", "")

    if action == "start":
        if not sim.running:
            sim.running = True
            sim_thread = threading.Thread(target=_sim_loop, daemon=True)
            sim_thread.start()
    elif action == "pause":
        sim.running = False
    elif action == "step":
        sim.running = False
        sim.step()
    elif action == "reset":
        sim.running = False
        time.sleep(0.15)
        sim.reset()
    else:
        return jsonify({"error": f"Unknown action: {action}"}), 400

    return jsonify({"ok": True, "action": action})


@app.route("/api/config", methods=["POST"])
def api_config():
    data = request.get_json(force=True)
    if "speed_ms" in data:
        sim.speed_ms = max(10, min(2000, int(data["speed_ms"])))
    return jsonify({"ok": True})


@app.route("/api/chaos", methods=["POST"])
def api_chaos():
    data = request.get_json(force=True)
    if "traffic_mult" in data:
        sim.traffic_mult = max(1.0, min(8.0, float(data["traffic_mult"])))
    if "eve_aggression" in data:
        sim.eve_aggression = max(1.0, min(10.0, float(data["eve_aggression"])))
    if "keygen_pct" in data:
        sim.keygen_pct = max(10.0, min(100.0, float(data["keygen_pct"])))
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDQN Dashboard Server")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    print(f"\n  SDQN Dashboard → http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)
