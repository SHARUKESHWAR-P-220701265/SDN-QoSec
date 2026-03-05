"""
sdn_controller.py
=================
SDN Controller for a Software-Defined Quantum Network (SDQN).

Responsibilities
----------------
* Maintain a **Global Knowledge Map** of the network topology as a weighted
  NetworkX graph.
* Ingest per-link telemetry (current key volume K_curr and QBER).
* Compute the **QoSec cost** for every link using the formula:

      C(u,v) = α·(D_uv / D_max)
             + β·(K_threshold / K_curr(u,v))
             + γ·exp(η · QBER_uv)

  Links whose key buffer drops below K_MIN (50 bits) are treated as if they
  have infinite cost and are effectively removed from routing.
* Route key-distribution requests via a modified Dijkstra algorithm that uses
  the QoSec cost as the edge weight.
* React to telemetry updates (e.g. QBER spikes injected by Eve) by
  recomputing affected paths on demand.

10-node / 18-link mesh topology
--------------------------------
Nodes  : 0 – 9
Edges  : 18 undirected links (see TOPOLOGY_EDGES below)
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default QoSec tuning parameters
# (can be overridden via constructor kwargs or environment variables)
# ---------------------------------------------------------------------------
DEFAULT_ALPHA: float = 0.4    # Weight: normalised link distance
DEFAULT_BETA: float = 0.4     # Weight: inverse key-buffer level
DEFAULT_GAMMA: float = 0.2    # Weight: QBER exponential penalty
DEFAULT_ETA: float = 10.0     # Exponent multiplier for QBER term
DEFAULT_K_THRESHOLD: float = 200.0  # Target key reserve used in β (bits)
DEFAULT_K_MIN: float = 50.0   # Link pruned when K_curr < K_MIN (bits)

# ---------------------------------------------------------------------------
# Fixed mesh topology
# ---------------------------------------------------------------------------
#  10 nodes (0-9), 18 bidirectional links
TOPOLOGY_NODES: List[int] = list(range(10))

TOPOLOGY_EDGES: List[Tuple[int, int, Dict[str, Any]]] = [
    # (u, v, initial_attributes)
    # distances in km, initial K_curr in bits, base QBER
    (0, 1, {"distance": 45.0,  "K_curr": 1000.0, "QBER": 0.01}),
    (0, 2, {"distance": 30.0,  "K_curr": 1200.0, "QBER": 0.01}),
    (0, 3, {"distance": 60.0,  "K_curr": 900.0,  "QBER": 0.02}),
    (1, 2, {"distance": 25.0,  "K_curr": 1100.0, "QBER": 0.01}),
    (1, 4, {"distance": 55.0,  "K_curr": 800.0,  "QBER": 0.01}),
    (1, 5, {"distance": 70.0,  "K_curr": 750.0,  "QBER": 0.02}),
    (2, 3, {"distance": 40.0,  "K_curr": 950.0,  "QBER": 0.01}),
    (2, 6, {"distance": 80.0,  "K_curr": 700.0,  "QBER": 0.02}),
    (3, 7, {"distance": 50.0,  "K_curr": 860.0,  "QBER": 0.01}),
    (4, 5, {"distance": 35.0,  "K_curr": 1050.0, "QBER": 0.01}),
    (4, 8, {"distance": 65.0,  "K_curr": 780.0,  "QBER": 0.02}),
    (5, 6, {"distance": 45.0,  "K_curr": 920.0,  "QBER": 0.01}),
    (5, 9, {"distance": 75.0,  "K_curr": 680.0,  "QBER": 0.02}),
    (6, 7, {"distance": 30.0,  "K_curr": 1150.0, "QBER": 0.01}),
    (7, 8, {"distance": 55.0,  "K_curr": 830.0,  "QBER": 0.01}),
    (7, 9, {"distance": 40.0,  "K_curr": 990.0,  "QBER": 0.02}),
    (8, 9, {"distance": 50.0,  "K_curr": 870.0,  "QBER": 0.01}),
    (0, 9, {"distance": 90.0,  "K_curr": 620.0,  "QBER": 0.02}),
]


# ---------------------------------------------------------------------------
# Helper dataclass — snapshot of a single link's live state
# ---------------------------------------------------------------------------
@dataclass
class LinkState:
    """Current telemetry for a single (u, v) link."""
    u: int
    v: int
    distance: float       # km
    K_curr: float         # current key buffer (bits)
    QBER: float           # Quantum Bit Error Rate  [0, 1]
    qosec_cost: float     # last computed QoSec cost (inf = pruned)


# ---------------------------------------------------------------------------
# SDN Controller
# ---------------------------------------------------------------------------
class SDN_Controller:
    """
    Centralised Software-Defined Network Controller for an SDQN.

    Parameters
    ----------
    alpha, beta, gamma : float
        Weighting coefficients for the three QoSec cost terms (must sum ≤ 1).
    eta : float
        Exponent multiplier applied to QBER in the exponential penalty term.
    k_threshold : float
        Target key reserve (bits) used in the β term of the cost function.
    k_min : float
        Links with K_curr < k_min are pruned (cost = ∞).
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        eta: float = DEFAULT_ETA,
        k_threshold: float = DEFAULT_K_THRESHOLD,
        k_min: float = DEFAULT_K_MIN,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.k_threshold = k_threshold
        self.k_min = k_min

        # Thread safety: telemetry updates can come from multiple threads
        # (quantum data plane, Eve, etc.)
        self._lock = threading.RLock()

        # ── Build the Global Knowledge Map ──────────────────────────────────
        self.graph: nx.Graph = nx.Graph()
        self._build_topology()

        # Pre-compute D_max across all links (used to normalise distance term)
        self._d_max: float = max(
            data["distance"] for _, _, data in self.graph.edges(data=True)
        )

        logger.info(
            "SDN Controller initialised | nodes=%d edges=%d D_max=%.1f km",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            self._d_max,
        )

    # -----------------------------------------------------------------------
    # Topology construction
    # -----------------------------------------------------------------------
    def _build_topology(self) -> None:
        """Populate the NetworkX graph with the fixed 10-node/18-link mesh."""
        self.graph.add_nodes_from(TOPOLOGY_NODES)
        for u, v, attrs in TOPOLOGY_EDGES:
            self.graph.add_edge(u, v, **attrs)
        logger.debug("Topology built: %d nodes, %d edges",
                     self.graph.number_of_nodes(),
                     self.graph.number_of_edges())

    # -----------------------------------------------------------------------
    # Telemetry ingestion
    # -----------------------------------------------------------------------
    def update_telemetry(
        self,
        u: int,
        v: int,
        k_curr: float,
        qber: float,
    ) -> None:
        """
        Ingest a telemetry report from the quantum data plane for link (u, v).

        Both directions (u→v and v→u) are updated because NetworkX stores
        undirected edges.  The controller logs a WARNING when QBER exceeds
        the QBER alert threshold (10 %).

        Parameters
        ----------
        u, v    : int   — Node identifiers at each end of the link.
        k_curr  : float — Current key buffer level in bits.
        qber    : float — Measured Quantum Bit Error Rate [0, 1].
        """
        QBER_ALERT_THRESHOLD = 0.10

        if not self.graph.has_edge(u, v):
            logger.warning("update_telemetry: no edge (%d, %d) in graph", u, v)
            return

        with self._lock:
            self.graph[u][v]["K_curr"] = k_curr
            self.graph[u][v]["QBER"] = qber

        if qber >= QBER_ALERT_THRESHOLD:
            logger.warning(
                "[ALERT] HIGH QBER on link (%d<->%d): QBER=%.3f "
                "(threshold=%.2f) -- possible Intercept-Resend attack!",
                u, v, qber, QBER_ALERT_THRESHOLD,
            )

        logger.debug(
            "Telemetry updated (%d<->%d) K_curr=%.1f bits  QBER=%.4f",
            u, v, k_curr, qber,
        )

    # -----------------------------------------------------------------------
    # QoSec cost function  ← PRIMARY DELIVERABLE
    # -----------------------------------------------------------------------
    def _qosec_cost(self, u: int, v: int) -> float:
        """
        Compute the QoSec link cost for the edge connecting *u* and *v*.

        Formula
        -------
            C(u,v) = α · (D_uv / D_max)
                   + β · (K_threshold / K_curr(u,v))
                   + γ · exp(η · QBER_uv)

        Guard
        -----
        Returns ``math.inf`` when K_curr < k_min (50 bits by default),
        effectively removing the link from the routing graph.

        Returns
        -------
        float : QoSec cost  (0 < cost < ∞)  or  math.inf  (pruned).
        """
        data = self.graph[u][v]
        k_curr: float = data["K_curr"]
        qber: float = data["QBER"]
        distance: float = data["distance"]

        # ── Guard: prune links with critically low key buffers ──────────────
        if k_curr < self.k_min:
            logger.debug(
                "Link (%d<->%d) pruned: K_curr=%.1f < K_min=%.1f",
                u, v, k_curr, self.k_min,
            )
            return math.inf

        # ── Term 1: normalised geographic distance ──────────────────────────
        distance_term: float = self.alpha * (distance / self._d_max)

        # ── Term 2: inverse key-buffer level (higher cost = less keys) ──────
        key_term: float = self.beta * (self.k_threshold / k_curr)

        # ── Term 3: exponential QBER penalty ────────────────────────────────
        qber_term: float = self.gamma * math.exp(self.eta * qber)

        cost: float = distance_term + key_term + qber_term

        logger.debug(
            "QoSec cost (%d<->%d): dist=%.4f  key=%.4f  qber=%.4f  total=%.4f",
            u, v, distance_term, key_term, qber_term, cost,
        )
        return cost

    # -----------------------------------------------------------------------
    # QoSec-aware Dijkstra routing
    # -----------------------------------------------------------------------
    def compute_qosec_path(self, src: int, dst: int) -> List[int]:
        """
        Find the lowest-QoSec-cost path from *src* to *dst*.

        Uses NetworkX's ``dijkstra_path`` with a weight callable that
        evaluates ``_qosec_cost`` on-the-fly, so telemetry updates are
        always reflected without rebuilding the graph.

        Parameters
        ----------
        src : int — Source node.
        dst : int — Destination node.

        Returns
        -------
        list[int] — Ordered list of node IDs forming the path.

        Raises
        ------
        nx.NetworkXNoPath
            When no path exists (all routes pruned or graph is disconnected).
        ValueError
            When src or dst are not in the graph.
        """
        for node in (src, dst):
            if node not in self.graph:
                raise ValueError(f"Node {node} is not in the topology graph.")

        with self._lock:
            try:
                path = nx.dijkstra_path(
                    self.graph,
                    src,
                    dst,
                    weight=lambda u, v, _: self._qosec_cost(u, v),
                )
            except nx.NetworkXNoPath:
                logger.error(
                    "No feasible QoSec path from %d to %d "
                    "(all routes exhausted or pruned).",
                    src, dst,
                )
                raise

        total_cost = sum(
            self._qosec_cost(path[i], path[i + 1])
            for i in range(len(path) - 1)
        )

        # Guard: Dijkstra may return a path whose every hop has cost=inf
        # (NetworkX treats inf-weight edges as reachable). Raise explicitly.
        if math.isinf(total_cost):
            logger.error(
                "No feasible QoSec path from %d to %d — all hops pruned "
                "(K_curr below K_MIN on every link along best route).", src, dst,
            )
            raise nx.NetworkXNoPath(
                f"No feasible QoSec path from {src} to {dst}: "
                "all links pruned due to insufficient key material."
            )
        logger.info(
            "QoSec path %d->%d : %s  (total cost=%.4f)",
            src, dst, " -> ".join(map(str, path)), total_cost,
        )
        return path

    # -----------------------------------------------------------------------
    # Topology snapshot (for logging / debugging)
    # -----------------------------------------------------------------------
    def get_topology_snapshot(self) -> Dict[str, Any]:
        """
        Return a serialisable snapshot of the current network state.

        Returns
        -------
        dict with keys:
            ``nodes``     — list of node IDs
            ``links``     — list of LinkState objects
            ``d_max_km``  — maximum link distance (normalisation factor)
        """
        with self._lock:
            links: List[LinkState] = []
            for u, v, data in self.graph.edges(data=True):
                links.append(LinkState(
                    u=u,
                    v=v,
                    distance=data["distance"],
                    K_curr=data["K_curr"],
                    QBER=data["QBER"],
                    qosec_cost=self._qosec_cost(u, v),
                ))

        return {
            "nodes": list(self.graph.nodes()),
            "links": links,
            "d_max_km": self._d_max,
        }

    # -----------------------------------------------------------------------
    # Convenience display helpers
    # -----------------------------------------------------------------------
    def print_link_table(self) -> None:
        """Pretty-print the current state of all links."""
        snapshot = self.get_topology_snapshot()
        header = f"{'Link':<10} {'Dist(km)':<12} {'K_curr(bits)':<15} {'QBER':<10} {'QoSec Cost':<12}"
        print("\n" + "=" * len(header))
        print("  Global Knowledge Map — Link State Table")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for ls in snapshot["links"]:
            cost_str = f"{ls.qosec_cost:.4f}" if ls.qosec_cost != math.inf else "  inf (pruned)"
            print(
                f"({ls.u}<->{ls.v}){'':<4} {ls.distance:<12.1f} {ls.K_curr:<15.1f} "
                f"{ls.QBER:<10.4f} {cost_str}"
            )
        print("=" * len(header) + "\n")

    def __repr__(self) -> str:
        return (
            f"SDN_Controller(nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()}, "
            f"α={self.alpha}, β={self.beta}, γ={self.gamma}, η={self.eta})"
        )
