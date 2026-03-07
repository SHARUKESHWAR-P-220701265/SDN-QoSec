"""
quantum_data_plane.py
=====================
Quantum Data Plane for the SDQN simulation.

Responsibilities
----------------
* Model per-link **Key Buffers** (K_buf) that hold raw key material.
* Simulate hybrid PQC-QKD key generation at a configurable base rate via a
  stub that mimics BB84 photon exchange outcomes.
* Implement the **Trusted Repeater** relay: when a key traverses an
  intermediate node, it is decrypted with OTP using the incoming link key
  and re-encrypted for the outgoing link.
* Report telemetry (K_curr, QBER) to the SDN Controller after each tick.
"""

from __future__ import annotations

import logging
import math
import random
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_QBER: float = 0.01          # Nominal QBER on a clean link
MAX_KEY_BUFFER_BITS: float = 5000.0  # Hard cap per link (bits)
KEY_GEN_RATE_BPS: float = 10_000.0  # 10 kbps base generation rate


# ---------------------------------------------------------------------------
# Key Buffer
# ---------------------------------------------------------------------------
@dataclass
class KeyBuffer:
    """
    Thread-safe key buffer for a single network link.

    Attributes
    ----------
    bits     : Current stored key material (bits).
    max_bits : Hard capacity cap (bits).
    qber     : Current Quantum Bit Error Rate for the link.
    """
    bits: float = MAX_KEY_BUFFER_BITS
    max_bits: float = MAX_KEY_BUFFER_BITS
    qber: float = BASE_QBER
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def deposit(self, amount: float) -> float:
        """Add *amount* bits (capped at max_bits). Returns bits actually added."""
        with self._lock:
            space = self.max_bits - self.bits
            added = min(amount, space)
            self.bits += added
            return added

    def withdraw(self, amount: float) -> bool:
        """
        Withdraw *amount* bits (OTP key consumption).

        Returns
        -------
        True  — sufficient key material available; bits deducted.
        False — insufficient key material; buffer unchanged.
        """
        with self._lock:
            if self.bits >= amount:
                self.bits -= amount
                return True
            return False

    @property
    def level(self) -> float:
        """Current buffer level in bits (thread-safe snapshot)."""
        with self._lock:
            return self.bits

    def set_qber(self, qber: float) -> None:
        with self._lock:
            self.qber = max(0.0, min(1.0, qber))


# ---------------------------------------------------------------------------
# Quantum Data Plane
# ---------------------------------------------------------------------------
class QuantumDataPlane:
    """
    Manages the quantum-layer key generation and distribution for the SDQN.

    Each undirected link is represented by a single ``KeyBuffer``.  Key
    generation is simulated by incrementing the buffer at each tick; the
    effective rate is reduced by the QBER (error bits must be discarded
    during privacy amplification in a real BB84 protocol).

    Parameters
    ----------
    edges    : List of (u, v) tuples matching the SDN Controller topology.
    rate_bps : Base key generation rate in bits-per-second.
    """

    def __init__(
        self,
        edges: List[Tuple[int, int]],
        rate_bps: float = KEY_GEN_RATE_BPS,
        max_bits: float = MAX_KEY_BUFFER_BITS,
    ) -> None:
        self.rate_bps = rate_bps
        self._buffers: Dict[Tuple[int, int], KeyBuffer] = {}

        for u, v in edges:
            key = self._canonical(u, v)
            self._buffers[key] = KeyBuffer(
                bits=max_bits,
                max_bits=max_bits,
                qber=BASE_QBER,
            )

        logger.info(
            "QuantumDataPlane ready: %d links @ %.0f bps base rate",
            len(self._buffers), rate_bps,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _canonical(u: int, v: int) -> Tuple[int, int]:
        """Return a canonical (min, max) key so (u,v) == (v,u)."""
        return (min(u, v), max(u, v))

    def _get_buffer(self, u: int, v: int) -> Optional[KeyBuffer]:
        return self._buffers.get(self._canonical(u, v))

    # -----------------------------------------------------------------------
    # BB84-inspired key generation stub
    # -----------------------------------------------------------------------
    def _bb84_raw_bits(self, qber: float, rate_bps: float, dt: float) -> float:
        """
        Estimate usable key bits generated over interval *dt* seconds.

        In a real BB84 protocol:
          1. Alice sends n photons.
          2. After sifting, ~50 % remain as raw key.
          3. Error correction discards ~h(QBER) fraction.
          4. Privacy amplification reduces further.

        Here we approximate with a binary entropy correction:
            usable_bits ≈ rate_bps · dt · (1 − h(QBER))
        where h(p) = −p·log2(p) − (1−p)·log2(1−p).
        """
        if qber <= 0.0:
            return rate_bps * dt
        if qber >= 0.5:
            return 0.0  # no usable key above 50% QBER

        # Binary entropy
        h = -qber * math.log2(qber) - (1 - qber) * math.log2(1 - qber)
        usable_fraction = max(0.0, 1.0 - h)
        return rate_bps * dt * usable_fraction

    # -----------------------------------------------------------------------
    # Tick update — generate keys on all links
    # -----------------------------------------------------------------------
    def tick(self, dt: float = 0.01) -> None:
        """
        Advance the quantum layer by *dt* seconds.

        For every link, compute usable bits from the BB84 stub and deposit
        into the buffer.  A small amount of random noise (±5 %) is added to
        simulate realistic variance.

        Parameters
        ----------
        dt : float — Simulation time step in seconds (default 10 ms).
        """
        for (u, v), buf in self._buffers.items():
            with buf._lock:
                qber = buf.qber
            raw = self._bb84_raw_bits(qber, self.rate_bps, dt)
            # Add ±5 % stochastic variation
            noise = random.uniform(-0.05, 0.05) * raw
            generated = max(0.0, raw + noise)
            buf.deposit(generated)

            # ── QBER natural decay ──────────────────────────────────────
            # When Eve is not actively attacking, the quantum channel's
            # error rate drifts back towards the physical baseline.
            # Decay rate: 5 % of the delta per tick (exponential decay).
            with buf._lock:
                if buf.qber > BASE_QBER:
                    buf.qber -= 0.05 * (buf.qber - BASE_QBER)
                    buf.qber = max(BASE_QBER, buf.qber)

    # -----------------------------------------------------------------------
    # OTP Trusted Repeater relay
    # -----------------------------------------------------------------------
    def relay_key(
        self,
        path: List[int],
        key_bits: float,
    ) -> bool:
        """
        Attempt to forward *key_bits* bits along *path* via OTP relay.

        Each hop consumes *key_bits* from the incoming link buffer (decrypt)
        and *key_bits* from the outgoing link buffer (re-encrypt).  If any
        hop lacks sufficient key material the relay fails and **no** buffers
        are modified.

        Parameters
        ----------
        path     : Ordered list of node IDs (length ≥ 2).
        key_bits : Number of bits in the payload key to relay.

        Returns
        -------
        True on success, False if any link has insufficient key material.
        """
        if len(path) < 2:
            logger.warning("relay_key: path must have at least 2 nodes.")
            return False

        # Collect all link buffers along the path
        links = [self._get_buffer(path[i], path[i + 1]) for i in range(len(path) - 1)]
        if any(b is None for b in links):
            logger.error("relay_key: one or more links not found in data plane.")
            return False

        # Pre-flight check — ensure all links have enough bits
        insufficient = [
            (path[i], path[i + 1])
            for i, b in enumerate(links)
            if b.level < key_bits        # type: ignore[union-attr]
        ]
        if insufficient:
            logger.warning(
                "relay_key FAILED: insufficient key on links %s "
                "(need %.0f bits each).",
                insufficient, key_bits,
            )
            return False

        # Commit withdrawals
        for i, buf in enumerate(links):
            success = buf.withdraw(key_bits)   # type: ignore[union-attr]
            if not success:
                # Should not happen after pre-flight, but guard anyway
                logger.error(
                    "relay_key: unexpected withdrawal failure on link (%d↔%d).",
                    path[i], path[i + 1],
                )
                return False

        logger.debug(
            "OTP relay success: %.0f bits along path %s",
            key_bits, " → ".join(map(str, path)),
        )
        return True

    # -----------------------------------------------------------------------
    # Telemetry helpers (called by SDN controller)
    # -----------------------------------------------------------------------
    def get_k_curr(self, u: int, v: int) -> float:
        """Return current key buffer level for link (u, v) in bits."""
        buf = self._get_buffer(u, v)
        return buf.level if buf else 0.0

    def get_qber(self, u: int, v: int) -> float:
        """Return current QBER for link (u, v)."""
        buf = self._get_buffer(u, v)
        if buf is None:
            return 0.0
        with buf._lock:
            return buf.qber

    def set_qber(self, u: int, v: int, qber: float) -> None:
        """Externally set the QBER on a link (used by Eve and the test harness)."""
        buf = self._get_buffer(u, v)
        if buf:
            buf.set_qber(qber)
            logger.debug("QBER on (%d↔%d) forced to %.4f", u, v, qber)

    def all_link_states(self) -> List[Tuple[int, int, float, float]]:
        """
        Return a list of (u, v, K_curr, QBER) tuples for all links.
        Useful for bulk telemetry push to the SDN Controller.
        """
        states = []
        for (u, v), buf in self._buffers.items():
            with buf._lock:
                states.append((u, v, buf.bits, buf.qber))
        return states
