"""
src/dsme/superframe.py
───────────────────────
DSME multi-superframe structure and parameter model.

Implements the timing definitions from Eq. 1 of the paper:
    SD = aBaseSuperframeDuration × 2^SO   (Superframe Duration)
    MD = aBaseSuperframeDuration × 2^MO   (Multi-superframe Duration)
    BI = aBaseSuperframeDuration × 2^BO   (Beacon Interval)

    Constraint: 0 ≤ SO ≤ MO ≤ BO ≤ 14
"""

from dataclasses import dataclass
from typing import List


# aBaseSuperframeDuration = 960 symbols (Eq. 4: BSD × NSS = 60 × 16)
BASE_SUPERFRAME_SYMBOLS: int = 960
SYMBOL_DURATION_MS: float    = 0.016   # each symbol = 0.016 ms (4 bits @ 250 kb/s)
SLOTS_PER_SUPERFRAME: int    = 16
CAP_SLOTS_DEFAULT: int       = 8       # slots 1–8
CFP_SLOTS_DEFAULT: int       = 7       # slots 9–15


@dataclass
class DSMEParams:
    """
    DSME multi-superframe parameter triple (BO, MO, SO).

    Enforces the DSME constraint: 0 ≤ SO ≤ MO ≤ BO ≤ 14
    """
    bo: int = 6   # Beacon Order
    mo: int = 5   # Multi-superframe Order
    so: int = 3   # Superframe Order

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if not (0 <= self.so <= self.mo <= self.bo <= 14):
            raise ValueError(
                f"DSME constraint violated: 0 ≤ SO ≤ MO ≤ BO ≤ 14, "
                f"got SO={self.so}, MO={self.mo}, BO={self.bo}"
            )

    @property
    def superframe_duration_ms(self) -> float:
        """SD = aBaseSuperframeDuration × 2^SO  (ms)"""
        return BASE_SUPERFRAME_SYMBOLS * (2 ** self.so) * SYMBOL_DURATION_MS

    @property
    def multi_superframe_duration_ms(self) -> float:
        """MD = aBaseSuperframeDuration × 2^MO  (ms)"""
        return BASE_SUPERFRAME_SYMBOLS * (2 ** self.mo) * SYMBOL_DURATION_MS

    @property
    def beacon_interval_ms(self) -> float:
        """BI = aBaseSuperframeDuration × 2^BO  (ms)"""
        return BASE_SUPERFRAME_SYMBOLS * (2 ** self.bo) * SYMBOL_DURATION_MS

    @property
    def superframes_per_multisuperframe(self) -> int:
        """Number of superframes in one multi-superframe = 2^(MO−SO)"""
        return 2 ** (self.mo - self.so)

    @property
    def beacon_sync_time_ms(self) -> float:
        """TBS = SD × (2^BO + 1)  (Eq. 2)"""
        return self.superframe_duration_ms * (2 ** self.bo + 1)

    def slot_duration_ms(self) -> float:
        """Duration of one timeslot within a superframe."""
        return self.superframe_duration_ms / SLOTS_PER_SUPERFRAME

    def __repr__(self) -> str:
        return (
            f"DSMEParams(BO={self.bo}, MO={self.mo}, SO={self.so} | "
            f"SD={self.superframe_duration_ms:.1f}ms, "
            f"MD={self.multi_superframe_duration_ms:.1f}ms, "
            f"SFs/MSF={self.superframes_per_multisuperframe})"
        )


class MultiSuperframe:
    """
    Represents one complete multi-superframe and its slot schedule.

    A multi-superframe contains 2^(MO−SO) superframes.
    Each superframe has SLOTS_PER_SUPERFRAME = 16 slots:
        Slot 0       : Beacon
        Slots 1–8    : CAP  (if this superframe has a CAP)
        Slots 9–15   : CFP  (GTS slots)

    Under CAP Reduction, only the first superframe contains a CAP;
    the remaining superframes have all non-beacon slots as CFP.
    """

    def __init__(self, params: DSMEParams, cap_reduction: bool = True):
        self.params        = params
        self.cap_reduction = cap_reduction
        self._build_schedule()

    def _build_schedule(self) -> None:
        """Build the slot schedule for this multi-superframe."""
        n_sf = self.params.superframes_per_multisuperframe
        self.superframes: List[dict] = []

        for i in range(n_sf):
            has_cap = (not self.cap_reduction) or (i == 0)
            cap_slots = CAP_SLOTS_DEFAULT if has_cap else 0
            cfp_slots = (SLOTS_PER_SUPERFRAME - 1) - cap_slots   # -1 for beacon

            self.superframes.append({
                "index":     i,
                "has_cap":   has_cap,
                "cap_slots": cap_slots,
                "cfp_slots": cfp_slots,
                "fcs":       cap_slots,   # Final CAP Slot field in beacon
            })

    @property
    def total_cap_slots(self) -> int:
        """Total CAP slots across the entire multi-superframe."""
        return sum(sf["cap_slots"] for sf in self.superframes)

    @property
    def total_cfp_slots(self) -> int:
        """Total CFP (GTS) slots across the entire multi-superframe."""
        return sum(sf["cfp_slots"] for sf in self.superframes)

    @property
    def cap_fraction(self) -> float:
        """Fraction of usable slots (non-beacon) allocated to CAP."""
        usable = self.params.superframes_per_multisuperframe * (SLOTS_PER_SUPERFRAME - 1)
        return self.total_cap_slots / usable if usable > 0 else 0.0

    def toggle_cap_reduction(self, enabled: bool) -> None:
        """Switch between CR and NCR mode and rebuild the schedule."""
        self.cap_reduction = enabled
        self._build_schedule()

    def summary(self) -> str:
        mode = "CR" if self.cap_reduction else "NCR"
        return (
            f"MultiSuperframe [{mode}] | "
            f"{self.params.superframes_per_multisuperframe} SFs | "
            f"CAP slots={self.total_cap_slots} ({self.cap_fraction:.0%}) | "
            f"CFP slots={self.total_cfp_slots}"
        )

