"""
src/pso/pso_optimizer.py
─────────────────────────
Core PSO engine implementing Equations 7 and 8 from the paper.

Each particle represents a candidate (BO, MO, SO) triple.
The swarm searches for the configuration that minimises the
fitness function (power consumption / delay) subject to:
    0 ≤ SO ≤ MO ≤ BO ≤ 14   (DSME constraint, Eq. 1)

Distributed execution: this optimizer runs independently on
each Cluster Head. The PANC aggregates pBest reports to
maintain and broadcast the global best.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


@dataclass
class Particle:
    """A single PSO particle representing one (BO, MO, SO) candidate."""
    position:  np.ndarray          # shape (3,) — [BO, MO, SO]
    velocity:  np.ndarray          # shape (3,)
    p_best_pos: np.ndarray = field(default=None)
    p_best_fit: float      = field(default=float("inf"))

    def __post_init__(self):
        if self.p_best_pos is None:
            self.p_best_pos = self.position.copy()


@dataclass
class PSOConfig:
    """Hyperparameters for the PSO algorithm."""
    swarm_size:   int   = 20
    max_iter:     int   = 35
    w:            float = 0.7     # inertia weight
    c1:           float = 1.5     # cognitive coefficient
    c2:           float = 1.5     # social coefficient
    bo_bounds:    Tuple[int, int] = (1, 14)
    mo_bounds:    Tuple[int, int] = (1, 14)
    so_bounds:    Tuple[int, int] = (0, 14)
    seed:         Optional[int]   = None


def _is_valid(position: np.ndarray) -> bool:
    """Check DSME constraint: 0 ≤ SO ≤ MO ≤ BO ≤ 14 (Eq. 1)."""
    bo, mo, so = int(round(position[0])), int(round(position[1])), int(round(position[2]))
    return 0 <= so <= mo <= bo <= 14


def _clip_to_bounds(position: np.ndarray, cfg: PSOConfig) -> np.ndarray:
    """Hard-clip each dimension to its allowed range."""
    clipped = position.copy()
    clipped[0] = np.clip(clipped[0], *cfg.bo_bounds)  # BO
    clipped[1] = np.clip(clipped[1], *cfg.mo_bounds)  # MO
    clipped[2] = np.clip(clipped[2], *cfg.so_bounds)  # SO
    return clipped


def _repair(position: np.ndarray) -> np.ndarray:
    """
    Repair a position that violates SO ≤ MO ≤ BO.
    Strategy: sort the three values ascending → assign SO ≤ MO ≤ BO.
    """
    vals = np.clip(np.round(position).astype(int), 0, 14)
    vals.sort()
    so, mo, bo = vals[0], vals[1], vals[2]
    return np.array([bo, mo, so], dtype=float)


class PSOOptimizer:
    """
    PSO optimizer for DSME multi-superframe parameter tuning.

    Usage (on a single Cluster Head)
    ---------------------------------
    fitness_fn = lambda pos: power_consumption(bo=pos[0], mo=pos[1], so=pos[2], ...)
    pso = PSOOptimizer(fitness_fn, cfg=PSOConfig())
    best_params, best_fitness, history = pso.run()

    Parameters
    ----------
    fitness_fn : callable
        Takes a (3,) array [BO, MO, SO] and returns a scalar fitness value.
        Lower is better (minimisation).
    cfg : PSOConfig
        Algorithm hyperparameters.
    g_best_pos : np.ndarray, optional
        Initial global best injected by the PANC (from a previous broadcast).
    g_best_fit : float, optional
        Fitness value of the injected global best.
    """

    def __init__(
        self,
        fitness_fn:   Callable[[np.ndarray], float],
        cfg:          PSOConfig = PSOConfig(),
        g_best_pos:   Optional[np.ndarray] = None,
        g_best_fit:   float = float("inf"),
    ):
        self.fitness_fn  = fitness_fn
        self.cfg         = cfg
        self.g_best_pos  = g_best_pos
        self.g_best_fit  = g_best_fit

        rng = np.random.default_rng(cfg.seed)
        self._rng = rng
        self.swarm: List[Particle] = []
        self._initialise_swarm()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _random_valid_position(self) -> np.ndarray:
        """Sample a random (BO, MO, SO) triple satisfying DSME constraints."""
        for _ in range(1000):
            bo = self._rng.integers(*self.cfg.bo_bounds, endpoint=True)
            mo = self._rng.integers(self.cfg.so_bounds[0], bo + 1)
            so = self._rng.integers(self.cfg.so_bounds[0], mo + 1)
            pos = np.array([bo, mo, so], dtype=float)
            if _is_valid(pos):
                return pos
        raise RuntimeError("Could not sample a valid initial position.")

    def _initialise_swarm(self) -> None:
        v_max = 3.0
        for _ in range(self.cfg.swarm_size):
            pos = self._random_valid_position()
            vel = self._rng.uniform(-v_max, v_max, size=3)
            particle = Particle(position=pos.copy(), velocity=vel)

            fit = self.fitness_fn(pos)
            particle.p_best_pos = pos.copy()
            particle.p_best_fit = fit

            if fit < self.g_best_fit:
                self.g_best_fit = fit
                self.g_best_pos = pos.copy()

            self.swarm.append(particle)

    # ── Core PSO Step ─────────────────────────────────────────────────────────

    def _update_particle(self, p: Particle) -> None:
        """Apply Eq. 7 (velocity) and Eq. 8 (position) from the paper."""
        r1 = self._rng.random(size=3)
        r2 = self._rng.random(size=3)

        # Eq. 7: velocity update
        cognitive = self.cfg.c1 * r1 * (p.p_best_pos - p.position)
        social    = self.cfg.c2 * r2 * (self.g_best_pos - p.position)
        p.velocity = self.cfg.w * p.velocity + cognitive + social

        # Clamp velocity to prevent explosion
        v_max = 4.0
        p.velocity = np.clip(p.velocity, -v_max, v_max)

        # Eq. 8: position update
        new_pos = p.position + p.velocity

        # Enforce bounds and repair constraint violations
        new_pos = _clip_to_bounds(new_pos, self.cfg)
        if not _is_valid(new_pos):
            new_pos = _repair(new_pos)

        p.position = new_pos

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def run(self) -> Tuple[np.ndarray, float, dict]:
        """
        Run the PSO optimisation loop (Algorithm 1 from the paper).

        Returns
        -------
        g_best_pos : np.ndarray — optimal (BO, MO, SO) as integers
        g_best_fit : float      — optimal fitness value
        history    : dict       — per-iteration log for convergence plots
        """
        history = {
            "g_best_fitness": [],
            "avg_fitness":    [],
            "g_best_params":  [],   # list of (BO, MO, SO) at each iteration
        }

        for iteration in range(1, self.cfg.max_iter + 1):
            iter_fitnesses = []

            for p in self.swarm:
                # Step 3 — evaluate fitness at new position
                self._update_particle(p)
                fit = self.fitness_fn(p.position)
                iter_fitnesses.append(fit)

                # Step 4 — update personal best
                if fit < p.p_best_fit:
                    p.p_best_fit = fit
                    p.p_best_pos = p.position.copy()

                # Step 5 — update global best
                if fit < self.g_best_fit:
                    self.g_best_fit = fit
                    self.g_best_pos = p.position.copy()

            # Log this iteration
            history["g_best_fitness"].append(self.g_best_fit)
            history["avg_fitness"].append(float(np.mean(iter_fitnesses)))
            bo, mo, so = [int(round(v)) for v in self.g_best_pos]
            history["g_best_params"].append((bo, mo, so))

            # Termination: constraint SO ≤ MO ≤ BO already enforced by repair;
            # secondary termination: stagnation over 10 iterations
            if iteration > 10:
                recent = history["g_best_fitness"][-10:]
                if max(recent) - min(recent) < 1e-6:
                    print(f"  [PSO] Converged at iteration {iteration} (stagnation).")
                    break

        best_params = np.array([int(round(v)) for v in self.g_best_pos])
        return best_params, self.g_best_fit, history

    # ── PANC Integration ──────────────────────────────────────────────────────

    def inject_global_best(self, g_best_pos: np.ndarray, g_best_fit: float) -> None:
        """
        Update the global best from an external broadcast (e.g. from PANC).
        Called at the start of each iteration in the distributed setting.
        """
        if g_best_fit < self.g_best_fit:
            self.g_best_fit = g_best_fit
            self.g_best_pos = g_best_pos.copy()
