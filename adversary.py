"""
adversary.py
============
Eve — the adversary thread for the SDQN simulation.

Models an **Intercept-Resend** attack on quantum channels:
  * Every ``attack_interval`` simulation ticks, Eve targets a random link.
  * She spikes the link's QBER to ``qber_spike`` (default 0.15 = 15 %).
  * The SDN Controller automatically detects the anomaly via telemetry and
    reroutes traffic away from the compromised link.

Eve runs as a Python *daemon thread* so it exits cleanly when the main
simulation finishes.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default adversary parameters
EVE_ATTACK_INTERVAL: int = 100    # ticks between attacks
EVE_QBER_SPIKE: float = 0.15      # Intercept-Resend QBER signature


class Eve(threading.Thread):
    """
    Background adversary thread that simulates Intercept-Resend attacks.

    Parameters
    ----------
    links            : All (u, v) link pairs in the topology.
    set_qber_fn      : Callable(u, v, qber) — sets QBER on the data plane.
    update_telemetry : Callable(u, v, k_curr, qber) — pushes telemetry to
                       the SDN Controller so it can re-route immediately.
    get_k_curr_fn    : Callable(u, v) → float — retrieves K_curr for the link.
    attack_interval  : Ticks between successive attacks.
    qber_spike       : QBER value injected during an attack.
    tick_event       : threading.Event that fires once per simulation tick,
                       used so Eve stays synchronised with the clock.
    stop_event       : threading.Event; set it to stop Eve gracefully.
    """

    def __init__(
        self,
        links: List[Tuple[int, int]],
        set_qber_fn: Callable[[int, int, float], None],
        update_telemetry: Callable[[int, int, float, float], None],
        get_k_curr_fn: Callable[[int, int], float],
        attack_interval: int = EVE_ATTACK_INTERVAL,
        qber_spike: float = EVE_QBER_SPIKE,
        tick_event: Optional[threading.Event] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        super().__init__(name="Eve-Adversary", daemon=True)
        self._links = list(links)
        self._set_qber = set_qber_fn
        self._update_telemetry = update_telemetry
        self._get_k_curr = get_k_curr_fn
        self.attack_interval = attack_interval
        self.qber_spike = qber_spike
        self._tick_event = tick_event or threading.Event()
        self._stop_event = stop_event or threading.Event()

        # Attack history for analysis
        self.attack_log: List[dict] = []
        self._tick_counter: int = 0

    # -----------------------------------------------------------------------
    # Thread entry point
    # -----------------------------------------------------------------------
    def run(self) -> None:
        """Thread kept for API compatibility; attacks are driven by attack_if_due()."""
        logger.info("[EVE] Eve thread started (attack every %d ticks via main loop)",
                    self.attack_interval)
        self._stop_event.wait()   # just park until simulation calls stop()
        logger.info("[EVE] Eve thread stopped.")

    # -----------------------------------------------------------------------
    # Attack logic
    # -----------------------------------------------------------------------
    def _launch_attack(self) -> None:
        """
        Pick a random link and inject an Intercept-Resend QBER spike.

        After injecting, push telemetry to the SDN Controller so it detects
        the attack immediately and can reroute affected traffic.
        """
        if not self._links:
            return

        target_u, target_v = random.choice(self._links)
        k_curr = self._get_k_curr(target_u, target_v)

        logger.warning(
            "[EVE] ATTACK -- tick %d | link (%d<->%d) | QBER spiked to %.2f",
            self._tick_counter, target_u, target_v, self.qber_spike,
        )

        # 1. Inject QBER spike into the quantum data plane
        self._set_qber(target_u, target_v, self.qber_spike)

        # 2. Immediately notify the SDN Controller (simulates real-time telemetry)
        self._update_telemetry(target_u, target_v, k_curr, self.qber_spike)

        # 3. Record the attack for post-simulation analysis
        self.attack_log.append({
            "tick": self._tick_counter,
            "link": (target_u, target_v),
            "qber_injected": self.qber_spike,
            "k_curr_at_attack": k_curr,
        })

    # -----------------------------------------------------------------------
    # Control
    # -----------------------------------------------------------------------
    def attack_if_due(self, tick: int) -> None:
        """
        Called **synchronously** from the simulation loop once per tick.

        This is the primary attack driver.  Using the main loop's tick
        counter guarantees Eve attacks at exactly tick 100, 200, 300…
        regardless of thread scheduling latency.
        """
        self._tick_counter = tick
        if tick % self.attack_interval == 0:
            self._launch_attack()

    def stop(self) -> None:
        """Signal Eve to exit gracefully after the current tick."""
        self._stop_event.set()
        self._tick_event.set()  # unblock any waiting

    def print_attack_summary(self) -> None:
        """Print a summary of all attacks Eve performed."""
        if not self.attack_log:
            print("[EVE] No attacks performed this simulation run.")
            return
        print(f"\n{'='*55}")
        print(f"  [EVE] Attack Summary  ({len(self.attack_log)} attack(s))")
        print(f"{'='*55}")
        print(f"  {'Tick':<8} {'Link':<12} {'QBER Injected':<16} {'K_curr'}")
        print(f"  {'-'*50}")
        for entry in self.attack_log:
            u, v = entry["link"]
            print(
                f"  {entry['tick']:<8} ({u}<->{v}){'':<7} "
                f"{entry['qber_injected']:<16.3f} {entry['k_curr_at_attack']:.1f} bits"
            )
        print(f"{'='*55}\n")
