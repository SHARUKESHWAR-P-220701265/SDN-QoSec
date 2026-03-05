"""
simulation.py
=============
Main entry point for the SDQN QoSec simulation.

Architecture (two decoupled planes)
------------------------------------
  Classical Control Plane   →  SDN_Controller   (NetworkX + QoSec routing)
  Quantum Data Plane        →  QuantumDataPlane  (KeyBuffers + BB84 stub)

Simulation loop (per tick)
--------------------------
  1. QuantumDataPlane.tick()        — generate new key bits on all links.
  2. TrafficGenerator.tick()        — generate and route Poisson key requests.
  3. Push telemetry to controller   — SDN graph stays current.
  4. Signal Eve                     — Eve attacks a link every 100 ticks.
  5. Print per-tick summary.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import List

import colorlog

from sdn_controller import SDN_Controller, TOPOLOGY_EDGES
from quantum_data_plane import QuantumDataPlane
from adversary import Eve
from traffic_generator import TrafficGenerator

# ---------------------------------------------------------------------------
# Logging setup — coloured console output
# ---------------------------------------------------------------------------
def _configure_logging(level: str = "INFO") -> None:
    handler = colorlog.StreamHandler(sys.stdout)
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(cyan)s[tick %(tick)-4s]%(reset)s %(message)s",
        log_colors={
            "DEBUG":    "white",
            "INFO":     "green",
            "WARNING":  "yellow,bold",
            "ERROR":    "red,bold",
            "CRITICAL": "red,bg_white",
        },
        reset=True,
    ))
    logging.root.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.root.addHandler(handler)


# ---------------------------------------------------------------------------
# Tick-aware log filter (injects tick number into every log record)
# ---------------------------------------------------------------------------
class _TickFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.current_tick: int = 0

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        record.tick = self.current_tick  # type: ignore[attr-defined]
        return True


_tick_filter = _TickFilter()


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
SIM_TICKS: int = 500
TICK_DT: float = 0.01          # 10 ms per tick
TELEMETRY_INTERVAL: int = 5    # Push telemetry to controller every N ticks
PRINT_INTERVAL: int = 50       # Print link table every N ticks


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
def run_simulation(
    ticks: int = SIM_TICKS,
    tick_dt: float = TICK_DT,
    log_level: str = "INFO",
    seed: int = 42,
) -> None:
    """
    Run the full SDQN QoSec simulation.

    Parameters
    ----------
    ticks     : Total number of simulation ticks.
    tick_dt   : Time step per tick in seconds.
    log_level : Python logging level string.
    seed      : RNG seed for reproducibility.
    """
    _configure_logging(log_level)
    logging.root.addFilter(_tick_filter)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("  [SDQN] QoSec Simulation -- Starting")
    logger.info(f"     Ticks: {ticks}  |  dt: {tick_dt*1000:.0f} ms  |  Seed: {seed}")
    logger.info("=" * 60)

    # ── Initialise planes ───────────────────────────────────────────────────
    edges = [(u, v) for u, v, _ in TOPOLOGY_EDGES]

    controller = SDN_Controller()
    qdp = QuantumDataPlane(edges=edges)

    # ── Eve — adversary thread ──────────────────────────────────────────────
    eve_tick_event = threading.Event()
    eve_stop_event = threading.Event()

    eve = Eve(
        links=edges,
        set_qber_fn=qdp.set_qber,
        update_telemetry=controller.update_telemetry,
        get_k_curr_fn=qdp.get_k_curr,
        attack_interval=100,
        qber_spike=0.15,
        tick_event=eve_tick_event,
        stop_event=eve_stop_event,
    )
    eve.start()

    # ── Traffic generator ───────────────────────────────────────────────────
    generator = TrafficGenerator(
        nodes=list(controller.graph.nodes()),
        compute_path_fn=controller.compute_qosec_path,
        relay_key_fn=qdp.relay_key,
        lam=2.0,
        key_size_bits=256.0,
        rng_seed=seed,
    )

    # Print initial topology
    controller.print_link_table()

    # ── Main loop ───────────────────────────────────────────────────────────
    for tick in range(1, ticks + 1):
        _tick_filter.current_tick = tick

        # 1. Quantum key generation
        qdp.tick(dt=tick_dt)

        # 2. Route and relay key requests
        generator.tick(tick_id=tick)

        # 3. Bulk telemetry push to SDN Controller
        if tick % TELEMETRY_INTERVAL == 0:
            for u, v, k_curr, qber in qdp.all_link_states():
                controller.update_telemetry(u, v, k_curr, qber)

        # 4. Signal Eve
        eve.signal_tick()

        # 5. Periodic link table print
        if tick % PRINT_INTERVAL == 0:
            logger.info("── Tick %d — Link State Snapshot ──", tick)
            controller.print_link_table()

    # ── Shutdown ─────────────────────────────────────────────────────────────
    eve.stop()
    eve.join(timeout=2.0)

    logger.info("=" * 60)
    logger.info("  [DONE] Simulation complete after %d ticks.", ticks)
    logger.info("=" * 60)

    # Final reports
    eve.print_attack_summary()
    generator.print_stats()
    controller.print_link_table()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SDQN QoSec Simulation")
    parser.add_argument("--ticks",     type=int,   default=SIM_TICKS,  help="Total simulation ticks")
    parser.add_argument("--dt",        type=float, default=TICK_DT,    help="Tick duration (seconds)")
    parser.add_argument("--log",       type=str,   default="INFO",     help="Log level")
    parser.add_argument("--seed",      type=int,   default=42,         help="RNG seed")
    args = parser.parse_args()

    run_simulation(
        ticks=args.ticks,
        tick_dt=args.dt,
        log_level=args.log,
        seed=args.seed,
    )
