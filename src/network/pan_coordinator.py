"""
src/network/pan_coordinator.py
────────────────────────────────
PAN Coordinator (PANC) logic for distributed PSO-based parameter tuning.

The PANC is the root of the cluster-tree. In the distributed PSO scheme
(Algorithm 1 from the paper), the PANC:
  1. Collects pBest reports from all Cluster Heads (Step 7)
  2. Computes and maintains the global best — gBest (Step 8)
  3. Broadcasts the updated gBest back to all CHs (Step 9)

The PANC also serves as the CH for its own cluster (Cluster 0),
so it participates in PSO optimisation directly.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.network.cluster_head import ClusterHead, PBestReport
from src.network.topology import ClusterTreeTopology, NodeRole
from src.pso.pso_optimizer import PSOConfig
from src.dsme.superframe import DSMEParams


@dataclass
class GBestRecord:
    """Snapshot of the global best at a given iteration."""
    iteration:    int
    g_best_params: np.ndarray   # [BO, MO, SO]
    g_best_fit:    float
    source_cluster: int         # which CH contributed this gBest


class PANCoordinator:
    """
    Models the PAN Coordinator's role in the distributed PSO algorithm.

    The PANC orchestrates one full round of distributed PSO per call to
    `run_round()`, which corresponds to one pass through Algorithm 1's
    outer loop (Steps 3–10).

    Parameters
    ----------
    topology   : ClusterTreeTopology
    pso_cfg    : PSOConfig — shared PSO hyperparameters for all CHs
    metric     : str — fitness metric: "power" | "combined"
    """

    def __init__(
        self,
        topology:  ClusterTreeTopology,
        pso_cfg:   PSOConfig  = PSOConfig(),
        metric:    str        = "power",
    ):
        self.topology   = topology
        self.pso_cfg    = pso_cfg
        self.metric     = metric

        self.g_best_params: Optional[np.ndarray] = None
        self.g_best_fit:    float                = float("inf")
        self.g_best_history: List[GBestRecord]   = []

        # Instantiate one ClusterHead object per cluster
        self.cluster_heads: Dict[int, ClusterHead] = {}
        self._init_cluster_heads()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_cluster_heads(self) -> None:
        """Create a ClusterHead instance for every CH node in the topology."""
        for ch_node in self.topology.get_cluster_heads():
            # Count end devices in this cluster for realistic load scaling
            cluster_nodes = self.topology.get_cluster_nodes(ch_node.cluster)
            n_devices = sum(
                1 for n in cluster_nodes if n.role == NodeRole.END_DEVICE
            )
            self.cluster_heads[ch_node.cluster] = ClusterHead(
                node=ch_node,
                n_devices=n_devices,
                metric=self.metric,
                pso_cfg=self.pso_cfg,
                seed=ch_node.cluster,   # deterministic per-cluster seed
            )

    # ── Algorithm 1: Distributed PSO round ───────────────────────────────────

    def run_round(self, iteration: int) -> Dict[str, object]:
        """
        Execute one full distributed PSO round (one outer loop iteration).

        Steps 3–9 of Algorithm 1:
          3. Each CH runs local PSO → produces pBest
          7. CHs report pBest to PANC
          8. PANC updates gBest from all pBest reports
          9. PANC broadcasts gBest to all CHs

        Returns
        -------
        dict with keys: g_best_params, g_best_fit, p_best_reports, converged
        """
        # Step 3–6: Each CH executes one PSO iteration locally
        p_best_reports: List[PBestReport] = []
        for ch in self.cluster_heads.values():
            report = ch.run_iteration()
            p_best_reports.append(report)

        # Step 7–8: PANC aggregates pBest reports → update gBest
        self._update_g_best(p_best_reports, iteration)

        # Step 9: Broadcast gBest to all CHs
        self._broadcast_g_best()

        converged = self._check_convergence()

        return {
            "g_best_params":   self.g_best_params.copy() if self.g_best_params is not None else None,
            "g_best_fit":      self.g_best_fit,
            "p_best_reports":  p_best_reports,
            "converged":       converged,
            "iteration":       iteration,
        }

    def _update_g_best(self, reports: List[PBestReport], iteration: int) -> None:
        """Step 8: Update gBest if any CH found a better solution."""
        for report in reports:
            if report.p_best_fit < self.g_best_fit:
                self.g_best_fit    = report.p_best_fit
                self.g_best_params = report.p_best_params.copy()
                self.g_best_history.append(GBestRecord(
                    iteration=iteration,
                    g_best_params=self.g_best_params.copy(),
                    g_best_fit=self.g_best_fit,
                    source_cluster=report.cluster_id,
                ))

    def _broadcast_g_best(self) -> None:
        """Step 9: Send gBest to every CH so they can update their social component."""
        if self.g_best_params is None:
            return
        for ch in self.cluster_heads.values():
            ch.receive_g_best(self.g_best_params, self.g_best_fit)

    def _check_convergence(self, window: int = 5, tol: float = 1e-4) -> bool:
        """Check if gBest has stagnated over the last `window` updates."""
        if len(self.g_best_history) < window:
            return False
        recent_fits = [r.g_best_fit for r in self.g_best_history[-window:]]
        return (max(recent_fits) - min(recent_fits)) < tol

    # ── Full optimisation run ─────────────────────────────────────────────────

    def optimise(self, max_rounds: int = 35) -> Tuple[DSMEParams, float, List[float]]:
        """
        Run the full distributed PSO until convergence or max_rounds.

        Returns
        -------
        optimal_params : DSMEParams — best (BO, MO, SO) found
        best_fitness   : float
        g_best_curve   : list[float] — gBest fitness at each round (for plotting)
        """
        g_best_curve: List[float] = []

        print(f"Starting distributed PSO ({self.topology.n_nodes} nodes, "
              f"{len(self.cluster_heads)} clusters) ...")

        for rnd in range(1, max_rounds + 1):
            result = self.run_round(rnd)
            g_best_curve.append(result["g_best_fit"])

            if rnd % 5 == 0:
                bo, mo, so = [int(round(v)) for v in result["g_best_params"]]
                print(f"  Round {rnd:3d} | gBest fit={result['g_best_fit']:.4f} "
                      f"| params=(BO={bo}, MO={mo}, SO={so})")

            if result["converged"]:
                print(f"  Converged at round {rnd}.")
                break

        # Build a validated DSMEParams from the best solution found
        bo, mo, so = [int(round(v)) for v in self.g_best_params]
        # Repair if needed: enforce SO ≤ MO ≤ BO
        vals = sorted([so, mo, bo])
        optimal_params = DSMEParams(bo=vals[2], mo=vals[1], so=vals[0])

        print(f"\nOptimal params: {optimal_params}")
        print(f"Optimal fitness: {self.g_best_fit:.6f}")

        return optimal_params, self.g_best_fit, g_best_curve

    # ── Reporting ─────────────────────────────────────────────────────────────

    def per_cluster_summary(self) -> List[dict]:
        """Return a summary dict for each cluster's best solution."""
        return [
            {
                "cluster":     cid,
                "ch_node":     ch.node.node_id,
                "n_devices":   ch.n_devices,
                "best_params": ch.current_best_params(),
                "best_fit":    ch.p_best_fit,
            }
            for cid, ch in sorted(self.cluster_heads.items())
        ]
