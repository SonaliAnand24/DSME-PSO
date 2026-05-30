"""
src/network/cluster_head.py
────────────────────────────
Cluster Head (CH) logic for distributed PSO-based DSME parameter tuning.

Each CH independently runs a local PSO instance (Algorithm 1 from the paper)
to find the optimal (BO, MO, SO) that minimises power consumption for its
local cluster. After each iteration, it reports its personal best (pBest)
to the PAN Coordinator, which aggregates results into a global best (gBest).

This file models the distributed execution described in Section IV-C:
  "This algorithm runs on the cluster head of each cluster in the network
   as it is beneficial because it balances computational power and
   communication overhead."
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.pso.pso_optimizer import PSOOptimizer, PSOConfig
from src.pso.fitness import make_fitness_fn
from src.network.topology import Node, NodeRole


@dataclass
class PBestReport:
    """
    A personal-best report sent from a CH to the PANC after each PSO iteration.
    The PANC collects these to update the global best.
    """
    cluster_id:    int
    ch_node_id:    int
    p_best_params: np.ndarray     # shape (3,) — [BO, MO, SO]
    p_best_fit:    float
    iteration:     int


class ClusterHead:
    """
    Models one Cluster Head executing local PSO and communicating with the PANC.

    Parameters
    ----------
    node       : Node — the CH node from the topology
    n_devices  : int  — number of end devices in this cluster
                        (affects packet size estimate and load)
    metric     : str  — fitness metric: "power" | "combined"
    pso_cfg    : PSOConfig — PSO hyperparameters
    seed       : int  — random seed for this CH's PSO instance
    """

    def __init__(
        self,
        node:       Node,
        n_devices:  int         = 5,
        metric:     str         = "power",
        pso_cfg:    PSOConfig   = PSOConfig(),
        seed:       int         = 0,
    ):
        assert node.role == NodeRole.CLUSTER_HEAD, \
            f"Node {node.node_id} is not a Cluster Head (role={node.role})"

        self.node       = node
        self.n_devices  = n_devices
        self.metric     = metric

        # Packet size scales slightly with cluster density
        packet_size = 200 + n_devices * 5

        fitness_fn  = make_fitness_fn(metric=metric, packet_size_bytes=packet_size)
        cfg         = PSOConfig(
            swarm_size=pso_cfg.swarm_size,
            max_iter=pso_cfg.max_iter,
            w=pso_cfg.w, c1=pso_cfg.c1, c2=pso_cfg.c2,
            seed=seed,
        )
        self.pso    = PSOOptimizer(fitness_fn, cfg=cfg)

        self.iteration:      int   = 0
        self.p_best_params:  Optional[np.ndarray] = None
        self.p_best_fit:     float = float("inf")
        self._history:       List[float] = []

    # ── PSO step ──────────────────────────────────────────────────────────────

    def run_iteration(self) -> PBestReport:
        """
        Execute one PSO iteration locally and return a pBest report for the PANC.

        In the distributed scheme (Algorithm 1), Steps 2–4 are executed here:
          - Evaluate fitness for each particle
          - Update pBest per particle
          - Return best local result to PANC
        """
        self.iteration += 1

        # Run one full PSO sweep (all particles update once)
        # We re-run the full PSO here for simplicity; in a real distributed
        # system each CH would run one velocity/position update per iteration.
        best_params, best_fit, history = self.pso.run()

        self.p_best_params = best_params.copy()
        self.p_best_fit    = best_fit
        self._history.extend(history["g_best_fitness"])

        return PBestReport(
            cluster_id=self.node.cluster,
            ch_node_id=self.node.node_id,
            p_best_params=best_params.copy(),
            p_best_fit=best_fit,
            iteration=self.iteration,
        )

    def receive_g_best(self, g_best_params: np.ndarray, g_best_fit: float) -> None:
        """
        Receive the global best broadcast from the PANC (Algorithm 1, Step 9)
        and inject it into the local PSO as the social attractor.
        """
        self.pso.inject_global_best(g_best_params, g_best_fit)

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def cluster_id(self) -> int:
        return self.node.cluster

    @property
    def convergence_history(self) -> List[float]:
        return list(self._history)

    def current_best_params(self) -> Tuple[int, int, int]:
        """Return current best (BO, MO, SO) as integer triple."""
        if self.p_best_params is None:
            return (6, 5, 3)   # DSME defaults
        return tuple(int(v) for v in self.p_best_params)

    def __repr__(self) -> str:
        bo, mo, so = self.current_best_params()
        return (
            f"ClusterHead(cluster={self.cluster_id}, "
            f"node={self.node.node_id}, "
            f"best_fit={self.p_best_fit:.4f}, "
            f"params=(BO={bo},MO={mo},SO={so}))"
        )
