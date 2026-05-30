"""
src/dsme/mac_layer.py
──────────────────────
DSME MAC layer simulation.

Models the key MAC-layer behaviours used in the paper's simulation:
  - GTS (Guaranteed Time Slot) allocation and scheduling in the CFP
  - CSMA/CA contention in the CAP
  - Per-node packet queue management
  - Power state tracking (Tx, Rx, Idle) for Eq. 3

This is a discrete-time simulation operating at the granularity
of one multi-superframe per step.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.dsme.superframe import DSMEParams, MultiSuperframe


# ── Node state ────────────────────────────────────────────────────────────────

@dataclass
class NodeState:
    """Per-node MAC state tracked across multi-superframes."""
    node_id:         int
    queue_length:    int   = 0       # packets waiting for transmission
    packets_sent:    int   = 0
    packets_dropped: int   = 0
    energy_mJ:       float = 0.0
    total_delay_ms:  float = 0.0     # cumulative delay of all sent packets
    gts_allocated:   int   = 0       # number of GTS slots assigned to this node


# ── Hardware constants matching Table I ───────────────────────────────────────

@dataclass(frozen=True)
class RadioParams:
    Ptx_mW:   float = 255.0    # transmit power
    Prx_mW:   float = 135.0    # receive power
    Pidle_mW: float = 1.3      # idle/sleep power
    data_rate_bps: float = 650_000   # 650 kb/s
    ack_bytes: int = 11
    delta_s:  float = 1e-6     # propagation delay
    SIFS_s:   float = 160e-6


# ── CSMA/CA model ─────────────────────────────────────────────────────────────

def csma_ca_success_probability(n_contenders: int, backoff_slots: int = 8) -> float:
    """
    Approximate success probability for slotted CSMA/CA.
    Uses the standard 802.15.4 approximation:
        P_success ≈ (1 − 1/backoff_slots)^(n−1)
    where n is the number of contending nodes.
    """
    if n_contenders <= 0:
        return 1.0
    return (1.0 - 1.0 / backoff_slots) ** (n_contenders - 1)


# ── MAC simulator ─────────────────────────────────────────────────────────────

class DSMEMACSimulator:
    """
    Discrete-time DSME MAC simulator.

    One simulation step = one multi-superframe.

    The simulator tracks packet queues for all nodes and models:
      1. Packet arrivals (Poisson process, rate λ per node)
      2. CFP service: each node transmits up to gts_allocated packets
      3. CAP service: excess packets contend via CSMA/CA
      4. Energy accounting using the model from Eq. 3
      5. Delay accounting: time in queue until transmission

    Parameters
    ----------
    params      : DSMEParams — current BO/MO/SO configuration
    node_ids    : list of node IDs to simulate
    radio       : RadioParams — hardware constants
    rng_seed    : int — for reproducibility
    """

    def __init__(
        self,
        params:    DSMEParams,
        node_ids:  List[int],
        radio:     RadioParams = RadioParams(),
        rng_seed:  int = 42,
    ):
        self.params    = params
        self.radio     = radio
        self.rng        = np.random.default_rng(rng_seed)
        self.msf        = MultiSuperframe(params, cap_reduction=True)

        self.nodes: Dict[int, NodeState] = {
            nid: NodeState(node_id=nid) for nid in node_ids
        }
        self._gts_schedule: Dict[int, int] = {}   # node_id → gts slots allocated
        self._allocate_gts()

    # ── GTS Allocation ────────────────────────────────────────────────────────

    def _allocate_gts(self) -> None:
        """Distribute available CFP slots evenly across nodes."""
        total_gts = self.msf.total_cfp_slots
        n_nodes   = len(self.nodes)
        if n_nodes == 0:
            return
        base  = total_gts // n_nodes
        extra = total_gts % n_nodes
        for i, nid in enumerate(self.nodes):
            slots = base + (1 if i < extra else 0)
            self._gts_schedule[nid] = slots
            self.nodes[nid].gts_allocated = slots

    def update_params(self, new_params: DSMEParams) -> None:
        """Apply a new (BO, MO, SO) configuration and reallocate GTS."""
        self.params = new_params
        self.msf    = MultiSuperframe(new_params, cap_reduction=True)
        self._allocate_gts()

    # ── Per-step simulation ───────────────────────────────────────────────────

    def step(self, arrival_rates: Dict[int, float]) -> Dict[str, float]:
        """
        Simulate one multi-superframe.

        Parameters
        ----------
        arrival_rates : dict[node_id → λ in pkts/ms]
            Poisson arrival rate for each node this step.

        Returns
        -------
        metrics : dict with keys:
            avg_queue_length, avg_delay_ms, throughput_pct,
            total_energy_mJ, packets_sent, packets_dropped
        """
        msf_duration_ms = self.params.multi_superframe_duration_ms

        # 1. Packet arrivals
        for nid, lam in arrival_rates.items():
            arrivals = self.rng.poisson(lam * msf_duration_ms)
            self.nodes[nid].queue_length += arrivals

        # 2. CFP service (GTS — deterministic, one packet per slot)
        for nid, node in self.nodes.items():
            gts_slots = self._gts_schedule.get(nid, 0)
            served    = min(node.queue_length, gts_slots)
            node.queue_length  -= served
            node.packets_sent  += served

            # Delay for GTS-served packets ≈ half a multi-superframe (avg wait)
            node.total_delay_ms += served * (msf_duration_ms / 2)

            # Energy: Tx time per packet
            pkt_tx_time_ms = (256 * 8) / self.radio.data_rate_bps * 1e3
            node.energy_mJ += served * self.radio.Ptx_mW * pkt_tx_time_ms

        # 3. CAP service (CSMA/CA — probabilistic)
        cap_slots    = self.msf.total_cap_slots
        n_contenders = sum(1 for n in self.nodes.values() if n.queue_length > 0)

        if cap_slots > 0 and n_contenders > 0:
            p_success = csma_ca_success_probability(n_contenders)
            cap_capacity = int(cap_slots * p_success)   # effective CAP throughput

            # Serve nodes round-robin from CAP
            cap_remaining = cap_capacity
            for nid, node in self.nodes.items():
                if cap_remaining <= 0:
                    break
                if node.queue_length > 0:
                    served = min(node.queue_length, cap_remaining, 2)
                    node.queue_length  -= served
                    node.packets_sent  += served
                    cap_remaining      -= served

                    # CAP delay includes full multi-superframe wait + contention
                    contention_penalty = msf_duration_ms * (1 - p_success)
                    node.total_delay_ms += served * (msf_duration_ms + contention_penalty)

                    pkt_tx_time_ms = (256 * 8) / self.radio.data_rate_bps * 1e3
                    node.energy_mJ += served * self.radio.Ptx_mW * pkt_tx_time_ms

        # 4. Idle energy for all nodes during inactive portions
        idle_fraction = 0.3   # approximate duty-cycle idle fraction
        for node in self.nodes.values():
            node.energy_mJ += (
                self.radio.Pidle_mW * idle_fraction * msf_duration_ms
            )

        # 5. Drop packets that exceed a max queue depth (buffer overflow)
        max_queue = 50
        for node in self.nodes.values():
            overflow = max(0, node.queue_length - max_queue)
            node.queue_length   -= overflow
            node.packets_dropped += overflow

        return self._collect_metrics()

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _collect_metrics(self) -> Dict[str, float]:
        total_sent    = sum(n.packets_sent    for n in self.nodes.values())
        total_dropped = sum(n.packets_dropped for n in self.nodes.values())
        total_delay   = sum(n.total_delay_ms  for n in self.nodes.values())
        total_energy  = sum(n.energy_mJ       for n in self.nodes.values())
        avg_queue     = np.mean([n.queue_length for n in self.nodes.values()])

        avg_delay = (total_delay / total_sent) if total_sent > 0 else 0.0
        pdr       = total_sent / max(total_sent + total_dropped, 1)

        return {
            "avg_queue_length": float(avg_queue),
            "avg_delay_ms":     float(avg_delay),
            "throughput_pct":   float(pdr * 100),
            "total_energy_mJ":  float(total_energy),
            "packets_sent":     int(total_sent),
            "packets_dropped":  int(total_dropped),
        }

    def reset_counters(self) -> None:
        """Reset per-node accumulators (keep queue state)."""
        for node in self.nodes.values():
            node.packets_sent    = 0
            node.packets_dropped = 0
            node.energy_mJ       = 0.0
            node.total_delay_ms  = 0.0
