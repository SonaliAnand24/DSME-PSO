"""
src/network/topology.py
────────────────────────
Cluster-tree topology builder for DSME IoT network simulations.

Constructs the cluster-tree topology described in Section III-A
of the paper (Fig. 2): a PANC at the root, with multiple clusters
each containing a Cluster Head and several End Devices (sensors).

The topology provides:
  - Node list with roles (PANC, Cluster Head, End Device)
  - Parent-child relationships
  - Per-cluster node groupings for distributed PSO execution
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Node roles ────────────────────────────────────────────────────────────────

class NodeRole:
    PANC         = "PANC"
    CLUSTER_HEAD = "ClusterHead"
    END_DEVICE   = "EndDevice"


@dataclass
class Node:
    """A single node in the cluster-tree topology."""
    node_id:   int
    role:      str
    cluster:   int               # cluster index (0 = PANC's cluster)
    parent_id: Optional[int]     # None for PANC
    children:  List[int] = field(default_factory=list)
    x: float = 0.0              # 2D position for visualisation
    y: float = 0.0

    def is_coordinator(self) -> bool:
        return self.role in (NodeRole.PANC, NodeRole.CLUSTER_HEAD)


# ── Topology builder ──────────────────────────────────────────────────────────

class ClusterTreeTopology:
    """
    Builds a cluster-tree topology with configurable size.

    Layout
    ------
    - Node 0 : PANC (root)
    - Nodes 1 … n_clusters : Cluster Heads (one per cluster, direct children of PANC)
    - Remaining nodes : End Devices distributed across clusters

    Parameters
    ----------
    n_nodes     : total number of nodes (including PANC)
    n_clusters  : number of clusters (default: 5, matching Fig. 2 of the paper)
    seed        : random seed for reproducible layouts
    """

    def __init__(self, n_nodes: int = 40, n_clusters: int = 5, seed: int = 0):
        if n_nodes < n_clusters + 1:
            raise ValueError(
                f"Need at least n_clusters+1={n_clusters+1} nodes, got {n_nodes}"
            )
        self.n_nodes    = n_nodes
        self.n_clusters = n_clusters
        self.rng        = np.random.default_rng(seed)
        self.nodes: Dict[int, Node] = {}
        self._build()

    # ── Construction ──────────────────────────────────────────────────────────

    def _build(self) -> None:
        """Build the full cluster-tree."""
        # PANC at origin
        self.nodes[0] = Node(
            node_id=0, role=NodeRole.PANC, cluster=0,
            parent_id=None, x=0.0, y=0.0,
        )

        # Place Cluster Heads in a ring around the PANC
        ch_ids = []
        for c in range(self.n_clusters):
            ch_id = c + 1
            angle = 2 * np.pi * c / self.n_clusters
            x     = 50.0 * np.cos(angle)
            y     = 50.0 * np.sin(angle)
            self.nodes[ch_id] = Node(
                node_id=ch_id, role=NodeRole.CLUSTER_HEAD, cluster=c + 1,
                parent_id=0, x=x, y=y,
            )
            self.nodes[0].children.append(ch_id)
            ch_ids.append(ch_id)

        # Distribute End Devices across clusters
        n_end_devices = self.n_nodes - (1 + self.n_clusters)
        end_device_ids = list(range(1 + self.n_clusters, self.n_nodes))

        for i, ed_id in enumerate(end_device_ids):
            cluster_idx = i % self.n_clusters
            ch_id       = ch_ids[cluster_idx]
            ch_node     = self.nodes[ch_id]

            # Random offset from the CH
            angle  = self.rng.uniform(0, 2 * np.pi)
            radius = self.rng.uniform(10.0, 25.0)
            x      = ch_node.x + radius * np.cos(angle)
            y      = ch_node.y + radius * np.sin(angle)

            self.nodes[ed_id] = Node(
                node_id=ed_id, role=NodeRole.END_DEVICE,
                cluster=cluster_idx + 1, parent_id=ch_id, x=x, y=y,
            )
            ch_node.children.append(ed_id)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_cluster_nodes(self, cluster_idx: int) -> List[Node]:
        """Return all nodes (CH + End Devices) in a given cluster."""
        return [n for n in self.nodes.values() if n.cluster == cluster_idx]

    def get_cluster_heads(self) -> List[Node]:
        """Return all Cluster Head nodes."""
        return [n for n in self.nodes.values() if n.role == NodeRole.CLUSTER_HEAD]

    def get_end_devices(self) -> List[Node]:
        """Return all End Device nodes."""
        return [n for n in self.nodes.values() if n.role == NodeRole.END_DEVICE]

    def get_panc(self) -> Node:
        return self.nodes[0]

    def node_ids_by_role(self, role: str) -> List[int]:
        return [n.node_id for n in self.nodes.values() if n.role == role]

    # ── Cluster assignments ───────────────────────────────────────────────────

    @property
    def cluster_map(self) -> Dict[int, List[int]]:
        """
        Returns dict: cluster_idx → [node_ids in that cluster].
        Used by ClusterHead and PanCoordinator to route PSO results.
        """
        mapping: Dict[int, List[int]] = {}
        for node in self.nodes.values():
            if node.cluster not in mapping:
                mapping[node.cluster] = []
            mapping[node.cluster].append(node.node_id)
        return mapping

    def summary(self) -> str:
        n_ch = len(self.get_cluster_heads())
        n_ed = len(self.get_end_devices())
        return (
            f"ClusterTreeTopology | {self.n_nodes} nodes | "
            f"1 PANC + {n_ch} CHs + {n_ed} End Devices | "
            f"{self.n_clusters} clusters"
        )
