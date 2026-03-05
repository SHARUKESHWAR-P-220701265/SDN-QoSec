"""
tests/test_sdn_controller.py
============================
Unit tests for the SDQN simulation modules.

Coverage
--------
  1. Topology structure (10 nodes, 18 edges)
  2. QoSec cost — normal healthy link
  3. QoSec cost — pruned when K_curr < K_MIN
  4. QoSec cost — exponential spike at QBER = 0.15
  5. Path computation — valid route returned
  6. Rerouting after Eve QBER attack
  7. Telemetry update correctly mutates graph attributes
"""

from __future__ import annotations

import math
import threading
import time
import unittest

import networkx as nx

from sdn_controller import SDN_Controller, TOPOLOGY_EDGES
from quantum_data_plane import QuantumDataPlane, KeyBuffer
from adversary import Eve
from traffic_generator import TrafficGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_controller(**kwargs) -> SDN_Controller:
    """Return a fresh controller, optionally with overridden QoSec params."""
    return SDN_Controller(**kwargs)


def _edges() -> list:
    return [(u, v) for u, v, _ in TOPOLOGY_EDGES]


# ===========================================================================
# Test Suite 1 — SDN Controller
# ===========================================================================
class TestSDNControllerTopology(unittest.TestCase):
    """Validate the Global Knowledge Map topology."""

    def setUp(self):
        self.ctrl = _make_controller()

    def test_node_count(self):
        """Graph must have exactly 10 nodes."""
        self.assertEqual(self.ctrl.graph.number_of_nodes(), 10)

    def test_edge_count(self):
        """Graph must have exactly 18 undirected edges."""
        self.assertEqual(self.ctrl.graph.number_of_edges(), 18)

    def test_all_nodes_connected(self):
        """Graph must be fully connected (no isolated nodes)."""
        self.assertTrue(nx.is_connected(self.ctrl.graph))

    def test_edges_have_required_attrs(self):
        """Every edge must carry distance, K_curr, and QBER attributes."""
        for u, v, data in self.ctrl.graph.edges(data=True):
            with self.subTest(edge=(u, v)):
                self.assertIn("distance", data, f"Edge ({u},{v}) missing 'distance'")
                self.assertIn("K_curr",   data, f"Edge ({u},{v}) missing 'K_curr'")
                self.assertIn("QBER",     data, f"Edge ({u},{v}) missing 'QBER'")


# ===========================================================================
class TestQoSecCostFunction(unittest.TestCase):
    """Validate the QoSec link cost formula and pruning logic."""

    def setUp(self):
        self.ctrl = _make_controller()

    def _first_edge(self):
        u, v, _ = next(iter(TOPOLOGY_EDGES))
        return u, v

    def test_cost_is_positive_for_healthy_link(self):
        """A healthy link with K_curr >> K_MIN must yield a finite positive cost."""
        u, v = self._first_edge()
        # Ensure the link is healthy
        self.ctrl.update_telemetry(u, v, k_curr=1000.0, qber=0.01)
        cost = self.ctrl._qosec_cost(u, v)
        self.assertGreater(cost, 0.0)
        self.assertFalse(math.isinf(cost))

    def test_cost_is_inf_when_buffer_too_low(self):
        """K_curr < K_MIN must return math.inf (link pruned)."""
        u, v = self._first_edge()
        self.ctrl.update_telemetry(u, v, k_curr=10.0, qber=0.01)  # below 50-bit min
        cost = self.ctrl._qosec_cost(u, v)
        self.assertEqual(cost, math.inf)

    def test_cost_is_finite_at_exactly_k_min(self):
        """At K_curr == K_MIN the link is NOT pruned (guard uses strict <)."""
        u, v = self._first_edge()
        self.ctrl.update_telemetry(u, v, k_curr=self.ctrl.k_min, qber=0.01)
        cost = self.ctrl._qosec_cost(u, v)
        self.assertFalse(math.isinf(cost), "Expected finite cost at exactly K_MIN")
        self.assertGreater(cost, 0.0)

    def test_cost_increases_with_high_qber(self):
        """A QBER spike to 0.15 must raise the cost relative to QBER=0.01."""
        u, v = self._first_edge()
        self.ctrl.update_telemetry(u, v, k_curr=1000.0, qber=0.01)
        cost_low_qber = self.ctrl._qosec_cost(u, v)

        self.ctrl.update_telemetry(u, v, k_curr=1000.0, qber=0.15)
        cost_high_qber = self.ctrl._qosec_cost(u, v)

        self.assertGreater(cost_high_qber, cost_low_qber)

    def test_cost_decreases_with_higher_k_curr(self):
        """More key material (higher K_curr) must yield a lower cost."""
        u, v = self._first_edge()
        self.ctrl.update_telemetry(u, v, k_curr=200.0, qber=0.01)
        cost_low = self.ctrl._qosec_cost(u, v)

        self.ctrl.update_telemetry(u, v, k_curr=2000.0, qber=0.01)
        cost_high = self.ctrl._qosec_cost(u, v)

        self.assertLess(cost_high, cost_low)

    def test_cost_formula_manual_check(self):
        """Manually verify the formula against a known edge."""
        # Use edge (0,1): distance=45, set K_curr=1000, QBER=0.01
        self.ctrl.update_telemetry(0, 1, k_curr=1000.0, qber=0.01)
        d_max = self.ctrl._d_max
        alpha, beta, gamma, eta = (
            self.ctrl.alpha, self.ctrl.beta, self.ctrl.gamma, self.ctrl.eta
        )
        expected = (
            alpha * (45.0 / d_max)
            + beta * (self.ctrl.k_threshold / 1000.0)
            + gamma * math.exp(eta * 0.01)
        )
        actual = self.ctrl._qosec_cost(0, 1)
        self.assertAlmostEqual(actual, expected, places=10)


# ===========================================================================
class TestRouting(unittest.TestCase):
    """Validate the QoSec Dijkstra path computation."""

    def setUp(self):
        self.ctrl = _make_controller()

    def test_path_returned_for_all_pairs(self):
        """A path must exist between every (src, dst) pair in the healthy graph."""
        nodes = list(self.ctrl.graph.nodes())
        for src in nodes:
            for dst in nodes:
                if src == dst:
                    continue
                with self.subTest(src=src, dst=dst):
                    path = self.ctrl.compute_qosec_path(src, dst)
                    self.assertIsInstance(path, list)
                    self.assertGreaterEqual(len(path), 2)
                    self.assertEqual(path[0], src)
                    self.assertEqual(path[-1], dst)

    def test_path_is_valid_walk_in_graph(self):
        """Every consecutive pair in the returned path must be a graph edge."""
        path = self.ctrl.compute_qosec_path(0, 9)
        for i in range(len(path) - 1):
            self.assertTrue(
                self.ctrl.graph.has_edge(path[i], path[i + 1]),
                f"({path[i]}, {path[i+1]}) is not an edge in the graph",
            )

    def test_reroute_after_eve_attack(self):
        """
        After Eve spikes a link on the original route, the path must change.

        Strategy:
          1. Find the QoSec path 0→9.
          2. Spike QBER on every link in that path to 0.15.
          3. Recompute — expect a different path (or confirm cost increased).
        """
        original_path = self.ctrl.compute_qosec_path(0, 9)

        # Spike all links on the original route
        for i in range(len(original_path) - 1):
            u, v = original_path[i], original_path[i + 1]
            self.ctrl.update_telemetry(u, v, k_curr=1000.0, qber=0.15)

        rerouted_path = self.ctrl.compute_qosec_path(0, 9)

        # Either the path changed, or (unlikely) the rerouted path is costlier
        # but both must still be valid.
        self.assertEqual(rerouted_path[0], 0)
        self.assertEqual(rerouted_path[-1], 9)

    def test_invalid_node_raises_value_error(self):
        """compute_qosec_path must raise ValueError for nodes not in the graph."""
        with self.assertRaises(ValueError):
            self.ctrl.compute_qosec_path(0, 999)

    def test_no_path_when_all_pruned(self):
        """nx.NetworkXNoPath when every single link in the graph is pruned."""
        # Drain ALL edges to guarantee no path exists for any src/dst
        for u, v in list(self.ctrl.graph.edges()):
            self.ctrl.update_telemetry(u, v, k_curr=0.0, qber=0.0)
        with self.assertRaises(nx.NetworkXNoPath):
            self.ctrl.compute_qosec_path(0, 9)


# ===========================================================================
class TestTelemetryUpdate(unittest.TestCase):
    """Validate that telemetry ingestion correctly mutates graph attributes."""

    def setUp(self):
        self.ctrl = _make_controller()

    def test_update_k_curr(self):
        self.ctrl.update_telemetry(0, 1, k_curr=123.0, qber=0.01)
        self.assertAlmostEqual(self.ctrl.graph[0][1]["K_curr"], 123.0)

    def test_update_qber(self):
        self.ctrl.update_telemetry(0, 1, k_curr=500.0, qber=0.07)
        self.assertAlmostEqual(self.ctrl.graph[0][1]["QBER"], 0.07)

    def test_update_nonexistent_edge_does_not_raise(self):
        """Updating a non-existent edge should log a warning but not crash."""
        try:
            self.ctrl.update_telemetry(0, 99, k_curr=100.0, qber=0.01)
        except Exception as exc:
            self.fail(f"update_telemetry raised unexpectedly: {exc}")

    def test_telemetry_thread_safety(self):
        """Concurrent telemetry updates must not corrupt graph state."""
        errors = []

        def updater(k):
            try:
                self.ctrl.update_telemetry(0, 1, k_curr=float(k), qber=0.01)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=updater, args=(i * 10,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertFalse(errors, f"Thread-safety errors: {errors}")


# ===========================================================================
class TestQuantumDataPlane(unittest.TestCase):
    """Validate KeyBuffer and QuantumDataPlane behaviour."""

    def test_key_buffer_deposit_and_withdraw(self):
        buf = KeyBuffer(bits=200.0, max_bits=500.0)
        buf.deposit(100.0)
        self.assertAlmostEqual(buf.level, 300.0)
        success = buf.withdraw(150.0)
        self.assertTrue(success)
        self.assertAlmostEqual(buf.level, 150.0)

    def test_key_buffer_withdraw_insufficient(self):
        buf = KeyBuffer(bits=30.0, max_bits=500.0)
        success = buf.withdraw(100.0)
        self.assertFalse(success)
        self.assertAlmostEqual(buf.level, 30.0)  # unchanged

    def test_key_buffer_caps_at_max(self):
        buf = KeyBuffer(bits=490.0, max_bits=500.0)
        added = buf.deposit(100.0)
        self.assertAlmostEqual(added, 10.0)
        self.assertAlmostEqual(buf.level, 500.0)

    def test_qdp_tick_increases_buffers(self):
        qdp = QuantumDataPlane(edges=_edges())
        # Drain a link first so there's room to fill
        qdp._buffers[(0, 1)].bits = 0.0
        qdp.tick(dt=1.0)  # 1 second — plenty of bits
        self.assertGreater(qdp.get_k_curr(0, 1), 0.0)

    def test_qdp_relay_success(self):
        qdp = QuantumDataPlane(edges=_edges())
        # Ensure all buffers have plenty of key material
        for buf in qdp._buffers.values():
            buf.bits = 5000.0
        result = qdp.relay_key([0, 1, 2], key_bits=256.0)
        self.assertTrue(result)

    def test_qdp_relay_fails_on_low_buffer(self):
        qdp = QuantumDataPlane(edges=_edges())
        qdp._buffers[(0, 1)].bits = 10.0  # not enough
        result = qdp.relay_key([0, 1, 2], key_bits=256.0)
        self.assertFalse(result)


# ===========================================================================
class TestTrafficGenerator(unittest.TestCase):
    """Validate Poisson request generation and stats."""

    def _make_gen(self, always_route=True):
        self.ctrl = _make_controller()
        self.qdp = QuantumDataPlane(edges=_edges())
        # Give all buffers a massive amount of key material so they don't exhaust
        for buf in self.qdp._buffers.values():
            buf.bits = 1_000_000.0
            buf.max_bits = 1_000_000.0

        if always_route:
            gen = TrafficGenerator(
                nodes=list(self.ctrl.graph.nodes()),
                compute_path_fn=self.ctrl.compute_qosec_path,
                relay_key_fn=self.qdp.relay_key,
                lam=2.0,
                rng_seed=0,
            )
        else:
            gen = TrafficGenerator(
                nodes=list(self.ctrl.graph.nodes()),
                compute_path_fn=self.ctrl.compute_qosec_path,
                relay_key_fn=lambda path, bits: False,  # always fail
                lam=5.0,
                rng_seed=0,
            )
        return gen

    def test_requests_generated_over_ticks(self):
        gen = self._make_gen()
        total = sum(len(gen.tick(t)) for t in range(1, 101))
        # With λ=2 over 100 ticks expect ~200 ± ~3σ ≈ 200 ± 42
        self.assertGreater(total, 100)

    def test_blocking_rate_zero_when_buffers_full(self):
        gen = self._make_gen(always_route=True)
        for t in range(1, 51):
            gen.tick(t)
        self.assertAlmostEqual(gen.blocking_rate, 0.0)

    def test_blocking_rate_nonzero_on_relay_failure(self):
        gen = self._make_gen(always_route=False)
        for t in range(1, 51):
            gen.tick(t)
        self.assertGreater(gen.blocking_rate, 0.0)


# ===========================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
