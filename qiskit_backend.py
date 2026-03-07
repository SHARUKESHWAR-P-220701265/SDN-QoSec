"""
qiskit_backend.py
=================
Qiskit-powered BB84 quantum key distribution backend for the SDQN.

Integrates IBM Qiskit's quantum circuit simulator to perform **real** qubit
generation, Hadamard-basis encoding, measurement, and sifted-key extraction
following the BB84 protocol.

Architecture
------------
    Alice (sender)
      │  Prepares qubits in random {Z, X} bases
      │  using H-gates for X-basis encoding
      ▼
    Quantum Channel  ←── Eve may intercept here (Intercept-Resend)
      │
      ▼
    Bob (receiver)
      │  Measures in random {Z, X} bases
      │  Sifts key by comparing bases over classical channel

The module exposes two public functions:

    ``bb84_generate_raw_key``
        Run a single BB84 round producing *n* raw key bits.

    ``estimate_qber_from_sample``
        Compare a sample of Alice/Bob bits to estimate the QBER.

Dependencies
------------
    qiskit          >= 1.0       (Terra)
    qiskit-aer      >= 0.13      (Aer simulator backend)
    numpy           >= 1.26
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Qiskit imports ──────────────────────────────────────────────────────────
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import HGate, XGate, IGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.visualization import plot_bloch_multivector

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
BB84_DEFAULT_N_QUBITS: int = 256      # photons per round
BB84_SIFTING_EFFICIENCY: float = 0.50  # ~50 % bases match on average
QBER_SAMPLE_FRACTION: float = 0.10    # fraction of sifted key used for QBER check
QBER_SECURITY_THRESHOLD: float = 0.11 # abort if QBER > 11 %

# Basis labels for readability
BASIS_Z = 0  # computational basis  {|0⟩, |1⟩}
BASIS_X = 1  # Hadamard basis       {|+⟩, |−⟩}


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class BB84Round:
    """Result of a single BB84 key-generation round."""
    n_photons: int                          # total photons sent by Alice
    alice_bits: List[int] = field(default_factory=list)
    alice_bases: List[int] = field(default_factory=list)   # 0=Z, 1=X
    bob_bases: List[int] = field(default_factory=list)
    bob_measurements: List[int] = field(default_factory=list)
    sifted_key_alice: List[int] = field(default_factory=list)
    sifted_key_bob: List[int] = field(default_factory=list)
    matching_indices: List[int] = field(default_factory=list)
    qber_estimate: float = 0.0
    secure: bool = True
    raw_circuit: Optional[QuantumCircuit] = None


@dataclass
class BB84ProtocolEntry:
    """Single-row entry for the frontend BB84 live-feed table."""
    index: int
    alice_bit: int
    alice_basis: str        # "Z" or "X"
    encoded_state: str      # "|0⟩", "|1⟩", "|+⟩", "|−⟩"
    bob_basis: str
    bob_measurement: int
    basis_match: bool
    in_sifted_key: bool


# ── BB84 circuit builder ───────────────────────────────────────────────────

def _build_bb84_circuit(
    alice_bits: List[int],
    alice_bases: List[int],
    bob_bases: List[int],
    eve_intercept: bool = False,
    eve_bases: Optional[List[int]] = None,
) -> QuantumCircuit:
    """
    Construct a Qiskit QuantumCircuit that encodes Alice's qubits,
    optionally applies Eve's intercept-resend, and measures in Bob's bases.

    Parameters
    ----------
    alice_bits   : Alice's random bit string.
    alice_bases  : Alice's random basis choices (0=Z, 1=X).
    bob_bases    : Bob's random basis choices.
    eve_intercept: If True, Eve measures and re-prepares each qubit.
    eve_bases    : Eve's measurement bases (random if not supplied).

    Returns
    -------
    QuantumCircuit ready to execute on AerSimulator.
    """
    n = len(alice_bits)
    qr = QuantumRegister(n, name="q")
    cr = ClassicalRegister(n, name="c")
    qc = QuantumCircuit(qr, cr, name="BB84")

    # ── Step 1: Alice encodes ───────────────────────────────────────────
    for i in range(n):
        if alice_bits[i] == 1:
            qc.x(qr[i])                       # encode bit value
        if alice_bases[i] == BASIS_X:
            qc.h(qr[i])                       # switch to X-basis

    qc.barrier(label="channel")

    # ── Step 2 (optional): Eve intercepts ───────────────────────────────
    if eve_intercept:
        _eve_bases = eve_bases if eve_bases is not None else \
            np.random.randint(0, 2, size=n).tolist()
        eve_cr = ClassicalRegister(n, name="eve")
        qc.add_register(eve_cr)

        for i in range(n):
            if _eve_bases[i] == BASIS_X:
                qc.h(qr[i])
            qc.measure(qr[i], eve_cr[i])
            # Eve re-prepares based on her measurement (simulated by reset)
            qc.reset(qr[i])
            if _eve_bases[i] == BASIS_X:
                qc.h(qr[i])

        qc.barrier(label="eve_resend")

    # ── Step 3: Bob measures ────────────────────────────────────────────
    for i in range(n):
        if bob_bases[i] == BASIS_X:
            qc.h(qr[i])                       # rotate to X-basis before measuring
        qc.measure(qr[i], cr[i])

    return qc


# ── Public API ─────────────────────────────────────────────────────────────

def bb84_generate_raw_key(
    n_photons: int = BB84_DEFAULT_N_QUBITS,
    eve_present: bool = False,
    backend_shots: int = 1,
) -> BB84Round:
    """
    Execute one full BB84 round on the Qiskit Aer simulator.

    Parameters
    ----------
    n_photons     : Number of qubits (photons) Alice sends.
    eve_present   : Whether Eve performs an Intercept-Resend attack.
    backend_shots : Aer simulator shots (1 = single-shot measurement).

    Returns
    -------
    BB84Round with sifted keys, QBER estimate, and the raw circuit.
    """
    rng = np.random.default_rng()

    # Random choices
    alice_bits = rng.integers(0, 2, size=n_photons).tolist()
    alice_bases = rng.integers(0, 2, size=n_photons).tolist()
    bob_bases = rng.integers(0, 2, size=n_photons).tolist()

    # Build and execute circuit
    qc = _build_bb84_circuit(alice_bits, alice_bases, bob_bases,
                              eve_intercept=eve_present)

    simulator = AerSimulator()
    result = simulator.run(qc, shots=backend_shots).result()
    counts = result.get_counts(qc)

    # Parse Bob's measurement results (Qiskit returns bit strings reversed)
    measurement_str = max(counts, key=counts.get)
    # Strip Eve's register bits if present
    bob_result_str = measurement_str.split()[-1] if eve_present else measurement_str
    bob_measurements = [int(b) for b in reversed(bob_result_str)]

    # ── Sifting: keep only positions where Alice & Bob chose same basis ──
    sifted_alice: List[int] = []
    sifted_bob: List[int] = []
    matching_indices: List[int] = []

    for i in range(n_photons):
        if alice_bases[i] == bob_bases[i]:
            sifted_alice.append(alice_bits[i])
            sifted_bob.append(bob_measurements[i] if i < len(bob_measurements) else 0)
            matching_indices.append(i)

    # ── QBER estimation from a public sample ────────────────────────────
    qber = estimate_qber_from_sample(sifted_alice, sifted_bob,
                                      sample_fraction=QBER_SAMPLE_FRACTION)

    secure = qber < QBER_SECURITY_THRESHOLD

    logger.info(
        "BB84 round: %d photons → %d sifted bits | QBER=%.4f | secure=%s%s",
        n_photons, len(sifted_alice), qber, secure,
        " [EVE PRESENT]" if eve_present else "",
    )

    return BB84Round(
        n_photons=n_photons,
        alice_bits=alice_bits,
        alice_bases=alice_bases,
        bob_bases=bob_bases,
        bob_measurements=bob_measurements,
        sifted_key_alice=sifted_alice,
        sifted_key_bob=sifted_bob,
        matching_indices=matching_indices,
        qber_estimate=qber,
        secure=secure,
        raw_circuit=qc,
    )


def estimate_qber_from_sample(
    alice_bits: List[int],
    bob_bits: List[int],
    sample_fraction: float = QBER_SAMPLE_FRACTION,
) -> float:
    """
    Estimate the Quantum Bit Error Rate by publicly comparing a sample
    of the sifted key bits.

    Parameters
    ----------
    alice_bits      : Alice's sifted key bits.
    bob_bits        : Bob's sifted key bits.
    sample_fraction : Fraction of sifted bits to sacrifice for QBER check.

    Returns
    -------
    float : Estimated QBER ∈ [0, 1].
    """
    if not alice_bits or not bob_bits:
        return 0.0

    n = min(len(alice_bits), len(bob_bits))
    sample_size = max(1, int(n * sample_fraction))

    errors = sum(
        a != b for a, b in zip(alice_bits[:sample_size], bob_bits[:sample_size])
    )
    return errors / sample_size


def format_bb84_feed(round_data: BB84Round, max_rows: int = 20) -> List[Dict]:
    """
    Convert a BB84Round into a list of serialisable dicts for the
    frontend BB84 live-feed table.

    Each row shows: Alice's bit, Alice's basis, encoded quantum state,
    Bob's basis, Bob's measurement, whether bases matched, and whether
    the bit made it into the sifted key.
    """
    _state_map = {
        (0, BASIS_Z): "|0⟩",
        (1, BASIS_Z): "|1⟩",
        (0, BASIS_X): "|+⟩",
        (1, BASIS_X): "|−⟩",
    }
    _basis_label = {BASIS_Z: "Z", BASIS_X: "X"}

    rows: List[Dict] = []
    n = min(round_data.n_photons, max_rows)
    matching_set = set(round_data.matching_indices)

    for i in range(n):
        rows.append({
            "index": i,
            "alice_bit": round_data.alice_bits[i],
            "alice_basis": _basis_label[round_data.alice_bases[i]],
            "encoded_state": _state_map.get(
                (round_data.alice_bits[i], round_data.alice_bases[i]), "?"
            ),
            "bob_basis": _basis_label[round_data.bob_bases[i]],
            "bob_measurement": round_data.bob_measurements[i]
                if i < len(round_data.bob_measurements) else "?",
            "basis_match": round_data.alice_bases[i] == round_data.bob_bases[i],
            "in_sifted_key": i in matching_set,
        })

    return rows


# ── Bloch sphere helper (for future frontend integration) ──────────────────

def get_bloch_sphere_data(bit: int, basis: int) -> Dict:
    """
    Compute the Bloch sphere coordinates for a single qubit state.

    Used by the frontend to render a 3D Bloch sphere showing the
    quantum state of a photon during BB84 encoding.

    Returns
    -------
    dict with keys: x, y, z (Bloch coordinates) and state_label.
    """
    # |0⟩ → (0, 0, 1),   |1⟩ → (0, 0, −1)
    # |+⟩ → (1, 0, 0),   |−⟩ → (−1, 0, 0)
    _coords = {
        (0, BASIS_Z): {"x": 0.0, "y": 0.0, "z":  1.0, "label": "|0⟩"},
        (1, BASIS_Z): {"x": 0.0, "y": 0.0, "z": -1.0, "label": "|1⟩"},
        (0, BASIS_X): {"x": 1.0, "y": 0.0, "z":  0.0, "label": "|+⟩"},
        (1, BASIS_X): {"x":-1.0, "y": 0.0, "z":  0.0, "label": "|−⟩"},
    }
    return _coords.get((bit, basis), {"x": 0, "y": 0, "z": 1, "label": "?"})


# ── Module self-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  BB84 QKD — Qiskit Backend Self-Test")
    print("=" * 60)

    # Normal round (no Eve)
    clean = bb84_generate_raw_key(n_photons=128, eve_present=False)
    print(f"\n  Clean round : {len(clean.sifted_key_alice)} sifted bits, "
          f"QBER = {clean.qber_estimate:.4f}")

    # Round with Eve
    attacked = bb84_generate_raw_key(n_photons=128, eve_present=True)
    print(f"  Eve round   : {len(attacked.sifted_key_alice)} sifted bits, "
          f"QBER = {attacked.qber_estimate:.4f}")

    # Feed table preview
    feed = format_bb84_feed(clean, max_rows=8)
    print(f"\n  BB84 Feed (first 8 photons):")
    for row in feed:
        match = "✓" if row["basis_match"] else "✗"
        print(f"    q[{row['index']:>3}]  Alice: {row['alice_bit']}|{row['alice_basis']}  "
              f"→ {row['encoded_state']:>3}  Bob: {row['bob_basis']}  "
              f"meas={row['bob_measurement']}  match={match}")

    print(f"\n{'=' * 60}")
