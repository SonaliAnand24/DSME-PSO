"""
src/pso/fitness.py
───────────────────
Fitness functions derived directly from the paper's equations.

Eq. 1  — DSME multi-superframe timing definitions
Eq. 2  — Beacon synchronisation time TBS
Eq. 3  — Average power consumption P  (primary fitness function)
Eq. 4  — Superframe duration SD
Eq. 5  — Average transmission delay
Eq. 6  — Multi-superframe total delay DMSF
"""

import numpy as np
from dataclasses import dataclass


# ── Hardware constants (Table I from paper) ───────────────────────────────────

@dataclass(frozen=True)
class HardwareParams:
    """IEEE 802.15.4 radio hardware parameters from Table I."""
    Ptx:       float = 255.0          # mW  — transmit power
    Prx:       float = 135.0          # mW  — receive power
    Pidle:     float = 1.3            # mW  — idle power
    data_rate: float = 650e3          # b/s — 650 kb/s
    ack_size:  int   = 11             # bytes
    delta:     float = 1e-6           # s   — propagation delay (1 µs)
    SIFS:      float = 160e-6         # s   — 160 µs
    symbol_duration: float = 0.016e-3 # s   — 0.016 ms per symbol (4 bits)


# ── Timing model ──────────────────────────────────────────────────────────────

class DSMETimingModel:
    """
    Computes DSME timing quantities from (BO, MO, SO) and hardware parameters.
    All times in seconds unless noted.
    """

    # aBaseSuperframeDuration = 960 symbols (Eq. 4: BSD × NSS = 60 × 16)
    BASE_SUPERFRAME_SYMBOLS: int = 960

    def __init__(self, hw: HardwareParams = HardwareParams()):
        self.hw = hw

    def superframe_duration(self, so: int) -> float:
        """SD = aBaseSuperframeDuration × 2^SO  (Eq. 1 / Eq. 4)."""
        return self.BASE_SUPERFRAME_SYMBOLS * (2 ** so) * self.hw.symbol_duration

    def multi_superframe_duration(self, mo: int) -> float:
        """MD = aBaseSuperframeDuration × 2^MO  (Eq. 1)."""
        return self.BASE_SUPERFRAME_SYMBOLS * (2 ** mo) * self.hw.symbol_duration

    def beacon_interval(self, bo: int) -> float:
        """BI = aBaseSuperframeDuration × 2^BO  (Eq. 1)."""
        return self.BASE_SUPERFRAME_SYMBOLS * (2 ** bo) * self.hw.symbol_duration

    def beacon_sync_time(self, bo: int, so: int) -> float:
        """TBS = SD × (2^BO + 1)  (Eq. 2)."""
        sd = self.superframe_duration(so)
        return sd * (2 ** bo + 1)

    def tx_time(self, packet_size_bytes: int) -> float:
        """Time to transmit one data packet."""
        bits = packet_size_bytes * 8
        return bits / self.hw.data_rate

    def rx_time(self, packet_size_bytes: int) -> float:
        """Time spent receiving one data packet + ACK."""
        rx_bits = (packet_size_bytes + self.hw.ack_size) * 8
        return rx_bits / self.hw.data_rate

    def idle_time(self, bo: int, so: int) -> float:
        """
        Tidle = network setup time + beacon synchronisation time.
        Approximated as TBS here.
        """
        return self.beacon_sync_time(bo, so)

    def ack_tx_time(self) -> float:
        """Time to transmit an ACK frame."""
        return (self.hw.ack_size * 8) / self.hw.data_rate


# ── Power Consumption (Eq. 3) ─────────────────────────────────────────────────

def power_consumption(
    bo: int,
    mo: int,
    so: int,
    packet_size_bytes: int = 250,
    hw: HardwareParams = HardwareParams(),
) -> float:
    """
    Average power consumption per multi-superframe  (Eq. 3).

    P = 2^(MO−SO) × { Ptx·Ttx + Prx·Trx + Pidle·Tidle } / TMD

    Parameters
    ----------
    bo, mo, so : int  — DSME multi-superframe order parameters
    packet_size_bytes : int  — data payload size in bytes

    Returns
    -------
    P : float  — average power in mW
    """
    timing = DSMETimingModel(hw)

    T_tx   = timing.tx_time(packet_size_bytes)
    T_rx   = timing.rx_time(packet_size_bytes)
    T_idle = timing.idle_time(bo, so)
    T_MD   = timing.multi_superframe_duration(mo)

    superframes_per_multisuperframe = 2 ** (mo - so)

    energy_per_superframe = (
        hw.Ptx   * T_tx  +
        hw.Prx   * T_rx  +
        hw.Pidle * T_idle
    )

    P = superframes_per_multisuperframe * energy_per_superframe / T_MD

    # Add a small noise term to simulate realistic measurement variation
    noise = np.random.normal(0, 0.5)
    return float(P + noise)


# ── Delay (Eq. 5 & 6) ────────────────────────────────────────────────────────

def transmission_delay(
    bo: int,
    mo: int,
    so: int,
    packet_size_bytes: int = 250,
    hw: HardwareParams = HardwareParams(),
) -> float:
    """
    Total delay for successful transmission in one multi-superframe  (Eq. 6).

    DMSF = 2^(MO−SO) × { TA + 2·TACK + 3δ + 3·SIFS + Tidle }

    Returns
    -------
    DMSF : float  — delay in milliseconds
    """
    timing = DSMETimingModel(hw)

    T_A    = timing.tx_time(packet_size_bytes)    # data frame transmission time
    T_ACK  = timing.ack_tx_time()
    T_idle = timing.idle_time(bo, so)
    delta  = hw.delta
    SIFS   = hw.SIFS

    single_sf_delay = T_A + 2 * T_ACK + 3 * delta + 3 * SIFS + T_idle
    DMSF = (2 ** (mo - so)) * single_sf_delay

    # Return in milliseconds for readability
    return float(DMSF * 1e3)


# ── Throughput ────────────────────────────────────────────────────────────────

def throughput(
    bo: int,
    mo: int,
    so: int,
    packet_size_bytes: int = 250,
    n_gts: int = 7,                   # GTS slots per superframe (CFP spans 7 slots)
    hw: HardwareParams = HardwareParams(),
) -> float:
    """
    Effective network throughput in bits per second.

    Approximated as: (data bits × GTS slots per multi-superframe) / MD

    Returns
    -------
    throughput : float  — bps
    """
    timing = DSMETimingModel(hw)
    T_MD = timing.multi_superframe_duration(mo)
    superframes = 2 ** (mo - so)
    total_gts = superframes * n_gts
    bits_delivered = total_gts * packet_size_bytes * 8

    return float(bits_delivered / T_MD)


# ── Combined fitness (for multi-objective experiments) ─────────────────────────

def combined_fitness(
    bo: int,
    mo: int,
    so: int,
    packet_size_bytes: int = 250,
    w_power: float = 0.6,
    w_delay: float = 0.4,
    hw: HardwareParams = HardwareParams(),
) -> float:
    """
    Weighted combination of normalised power and delay.
    Used for multi-objective optimisation experiments in Section V.

    Lower is better.
    """
    P = power_consumption(bo, mo, so, packet_size_bytes, hw)
    D = transmission_delay(bo, mo, so, packet_size_bytes, hw)

    # Rough normalisation to bring both to similar scales
    P_norm = P / 300.0     # typical range 100–300 mW
    D_norm = D / 10000.0   # typical range 1000–10000 ms

    return w_power * P_norm + w_delay * D_norm


# ── Fitness factory (returns a callable for PSOOptimizer) ─────────────────────

def make_fitness_fn(
    metric:            str   = "power",   # "power" | "delay" | "throughput" | "combined"
    packet_size_bytes: int   = 250,
    hw:                HardwareParams = HardwareParams(),
):
    """
    Return a fitness function compatible with PSOOptimizer.

    The returned function signature is:
        f(position: np.ndarray) -> float
    where position = [BO, MO, SO].
    """
    def _fn(position):
        bo = int(round(float(position[0])))
        mo = int(round(float(position[1])))
        so = int(round(float(position[2])))

        if metric == "power":
            return power_consumption(bo, mo, so, packet_size_bytes, hw)
        elif metric in ("delay", "latency"):
            return transmission_delay(bo, mo, so, packet_size_bytes, hw)
        elif metric == "throughput":
            # Negate: PSO minimises, but we want to maximise throughput
            return -throughput(bo, mo, so, packet_size_bytes, hw=hw)
        elif metric == "combined":
            return combined_fitness(bo, mo, so, packet_size_bytes, hw=hw)
        else:
            raise ValueError(f"Unknown metric: {metric!r}")

    return _fn
