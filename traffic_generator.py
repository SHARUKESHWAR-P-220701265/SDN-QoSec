"""
traffic_generator.py
====================
Poisson-distributed key-distribution request generator for the SDQN.

Generates random key requests each simulation tick using a Poisson process
(λ = 2 requests/tick by default).  Each request:
  1. Picks a random (src, dst) node pair.
  2. Asks the SDN Controller for the optimal QoSec path.
  3. Attempts to relay the key over the data plane via OTP.
  4. Records success / failure for end-of-simulation analysis.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULT_LAMBDA: float = 2.0        # Mean requests per tick (Poisson λ)
DEFAULT_KEY_SIZE_BITS: float = 256.0  # Bits per key request (symmetric key)


# ---------------------------------------------------------------------------
# Request record
# ---------------------------------------------------------------------------
@dataclass
class KeyRequest:
    """Represents a single key-distribution request."""
    tick: int
    src: int
    dst: int
    key_bits: float
    path: Optional[List[int]] = None
    success: bool = False
    failure_reason: str = ""


# ---------------------------------------------------------------------------
# Traffic Generator
# ---------------------------------------------------------------------------
class TrafficGenerator:
    """
    Generates and dispatches quantum key requests each simulation tick.

    Parameters
    ----------
    nodes            : List of all node IDs in the topology.
    compute_path_fn  : Callable(src, dst) → List[int]  — SDN Controller routing.
    relay_key_fn     : Callable(path, bits) → bool  — Data plane OTP relay.
    lam              : Poisson λ (mean requests per tick).
    key_size_bits    : Size of each key request in bits.
    rng_seed         : Optional seed for reproducibility.
    """

    def __init__(
        self,
        nodes: List[int],
        compute_path_fn: Callable[[int, int], List[int]],
        relay_key_fn: Callable[[List[int], float], bool],
        lam: float = DEFAULT_LAMBDA,
        key_size_bits: float = DEFAULT_KEY_SIZE_BITS,
        rng_seed: Optional[int] = None,
    ) -> None:
        self._nodes = list(nodes)
        self._compute_path = compute_path_fn
        self._relay_key = relay_key_fn
        self.lam = lam
        self.key_size_bits = key_size_bits
        self._rng = np.random.default_rng(rng_seed)

        # Statistics
        self.request_log: List[KeyRequest] = []
        self._total_requests: int = 0
        self._successful: int = 0
        self._failed: int = 0

        logger.info(
            "TrafficGenerator ready: λ=%.1f req/tick  key_size=%d bits",
            self.lam, int(self.key_size_bits),
        )

    # -----------------------------------------------------------------------
    # Per-tick dispatch
    # -----------------------------------------------------------------------
    def tick(self, tick_id: int) -> List[KeyRequest]:
        """
        Generate and service all key requests arriving at tick *tick_id*.

        Uses ``numpy.random.Poisson(λ)`` to determine the number of requests,
        then routes and relays each one.

        Parameters
        ----------
        tick_id : int — Current simulation tick number.

        Returns
        -------
        List of ``KeyRequest`` objects processed this tick.
        """
        n_requests: int = int(self._rng.poisson(self.lam))
        results: List[KeyRequest] = []

        for _ in range(n_requests):
            req = self._generate_request(tick_id)
            self._service_request(req)
            self.request_log.append(req)
            results.append(req)
            self._total_requests += 1
            if req.success:
                self._successful += 1
            else:
                self._failed += 1

        if n_requests > 0:
            logger.debug(
                "Tick %d: %d request(s) generated | %d OK / %d FAIL",
                tick_id,
                n_requests,
                sum(1 for r in results if r.success),
                sum(1 for r in results if not r.success),
            )

        return results

    # -----------------------------------------------------------------------
    # Request generation and servicing
    # -----------------------------------------------------------------------
    def _generate_request(self, tick_id: int) -> KeyRequest:
        """Pick a random (src, dst) pair ensuring src ≠ dst."""
        src, dst = random.sample(self._nodes, 2)
        return KeyRequest(
            tick=tick_id,
            src=src,
            dst=dst,
            key_bits=self.key_size_bits,
        )

    def _service_request(self, req: KeyRequest) -> None:
        """
        Route and relay a single ``KeyRequest``.

        Flow
        ----
        1. Ask SDN Controller for QoSec-optimal path.
        2. Attempt OTP relay via Quantum Data Plane.
        3. Mark request success / failure with reason.
        """
        import networkx as nx  # lazy import to avoid circular deps

        try:
            path = self._compute_path(req.src, req.dst)
            req.path = path
        except nx.NetworkXNoPath:
            req.failure_reason = "No feasible QoSec path (all routes pruned)"
            logger.warning(
                "Tick %d | Request %d->%d BLOCKED: %s",
                req.tick, req.src, req.dst, req.failure_reason,
            )
            return
        except ValueError as exc:
            req.failure_reason = str(exc)
            return

        # Attempt OTP relay along the calculated path
        if self._relay_key(path, req.key_bits):
            req.success = True
            logger.debug(
                "Tick %d | Key %d->%d OK  path=%s",
                req.tick, req.src, req.dst,
                " -> ".join(map(str, path)),
            )
        else:
            req.failure_reason = "Key buffer exhaustion on one or more hops"
            logger.warning(
                "Tick %d | Key %d->%d FAIL (buffer low)  path=%s",
                req.tick, req.src, req.dst,
                " -> ".join(map(str, path)),
            )

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------
    @property
    def blocking_rate(self) -> float:
        """Fraction of requests that failed (0.0 = perfect, 1.0 = all blocked)."""
        if self._total_requests == 0:
            return 0.0
        return self._failed / self._total_requests

    def print_stats(self) -> None:
        """Print end-of-simulation traffic statistics."""
        print(f"\n{'='*50}")
        print("  [TRAFFIC] Traffic Generator -- Simulation Statistics")
        print(f"{'='*50}")
        print(f"  Total requests  : {self._total_requests}")
        print(f"  Successful      : {self._successful}")
        print(f"  Failed          : {self._failed}")
        print(f"  Blocking rate   : {self.blocking_rate:.2%}")
        print(f"{'='*50}\n")
