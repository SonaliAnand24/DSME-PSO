"""
run_simulation.py
──────────────────
Main entry point. Reproduces all six plots from Figure 3 of the paper:

  (a) Power consumption vs PSO iterations       (fixed network size)
  (b) Power consumption vs network size         (at convergence)
  (c) Latency vs PSO iterations                 (fixed network size)
  (d) Latency vs network size                   (at convergence)
  (e) Throughput vs PSO iterations              (fixed network size)
  (f) Throughput vs network size                (at convergence)

Usage
-----
  # Full reproduction (all 6 plots)
  python run_simulation.py --mode full --save results/figures/

  # Single metric
  python run_simulation.py --mode iterations --metric power --nodes 40

  # Scalability sweep
  python run_simulation.py --mode scalability --metric latency
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.pso.pso_optimizer import PSOOptimizer, PSOConfig
from src.pso.fitness import (
    make_fitness_fn, power_consumption, transmission_delay,
    throughput, HardwareParams,
)


# ── Simulation helpers ────────────────────────────────────────────────────────

def simulate_standard_dsme(metric: str, bo: int = 6, mo: int = 5, so: int = 3,
                             packet_size: int = 250) -> float:
    """
    Baseline: standard DSME with fixed default parameters (BO=6, MO=5, SO=3).
    These are the reference values from the paper's simulation setup.
    """
    hw = HardwareParams()
    if metric == "power":
        return power_consumption(bo, mo, so, packet_size, hw)
    elif metric in ("latency", "delay"):
        return transmission_delay(bo, mo, so, packet_size, hw)
    elif metric == "throughput":
        return throughput(bo, mo, so, packet_size, hw=hw)


def run_pso_experiment(
    metric:       str,
    max_iter:     int = 35,
    packet_size:  int = 250,
    seed:         int = 42,
) -> dict:
    """
    Run PSO for a given metric and return the convergence history.
    Simulates one cluster head executing Algorithm 1.
    """
    hw       = HardwareParams()
    fit_fn   = make_fitness_fn(metric=metric, packet_size_bytes=packet_size, hw=hw)
    cfg      = PSOConfig(swarm_size=20, max_iter=max_iter, seed=seed)
    pso      = PSOOptimizer(fit_fn, cfg=cfg)
    best_params, best_fit, history = pso.run()

    # Convert throughput back to positive
    if metric == "throughput":
        history["g_best_fitness"] = [-v for v in history["g_best_fitness"]]

    return {
        "best_params": best_params.tolist(),
        "best_fit":    best_fit if metric != "throughput" else -best_fit,
        "history":     history,
    }


def run_scalability_experiment(
    metric:      str,
    network_sizes: list,
    packet_size: int = 250,
    n_trials:    int = 5,
) -> dict:
    """
    For each network size, run PSO n_trials times (different seeds) and
    average the converged metric. Returns mean ± std for both PSO and baseline.
    """
    hw = HardwareParams()
    pso_means, pso_stds   = [], []
    dsme_means            = []

    for n_nodes in tqdm(network_sizes, desc=f"Scalability [{metric}]"):
        # Slightly scale packet size with network density (realistic)
        pkt = packet_size + (n_nodes - 10) // 10 * 5

        # PSO: average over multiple seeds
        results = []
        for seed in range(n_trials):
            fit_fn = make_fitness_fn(metric=metric, packet_size_bytes=pkt, hw=hw)
            cfg    = PSOConfig(swarm_size=20, max_iter=35, seed=seed)
            pso    = PSOOptimizer(fit_fn, cfg=cfg)
            _, best_fit, _ = pso.run()
            val = -best_fit if metric == "throughput" else best_fit
            results.append(val)

        pso_means.append(float(np.mean(results)))
        pso_stds.append(float(np.std(results)))

        # Baseline: standard DSME with default params
        # Scale slightly with network size to reflect real congestion effects
        scale = 1.0 + (n_nodes - 10) * 0.005
        base  = simulate_standard_dsme(metric=metric, packet_size=pkt)
        dsme_means.append(float(base * scale))

    return {
        "network_sizes": network_sizes,
        "pso_means":     pso_means,
        "pso_stds":      pso_stds,
        "dsme_means":    dsme_means,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "power":      ("Avg. Cluster Power Consumption (mW)", "Power Consumption"),
    "latency":    ("Latency (ms)",                         "Latency"),
    "throughput": ("Throughput (bps)",                     "Throughput"),
}

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "grid.linestyle": "--",
})


def plot_vs_iterations(ax, history_pso: list, history_dsme_flat: float,
                        metric: str, title: str) -> None:
    iters = list(range(1, len(history_pso) + 1))
    dsme_line = [history_dsme_flat] * len(iters)

    ax.plot(iters, history_pso,  "b-o", ms=4, lw=1.5, label="Proposed PSO Scheme")
    ax.plot(iters, dsme_line,    "r--^", ms=4, lw=1.5, label="Standard DSME")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(METRIC_LABELS[metric][0])
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)


def plot_vs_network_size(ax, sizes, pso_means, pso_stds, dsme_means,
                          metric: str, title: str) -> None:
    ax.plot(sizes, pso_means,  "b-o", ms=5, lw=1.5, label="Proposed PSO Scheme")
    ax.fill_between(sizes,
                    [m - s for m, s in zip(pso_means, pso_stds)],
                    [m + s for m, s in zip(pso_means, pso_stds)],
                    alpha=0.15, color="blue")
    ax.plot(sizes, dsme_means, "r--^", ms=5, lw=1.5, label="Standard DSME")
    ax.set_xlabel("Network Size")
    ax.set_ylabel(METRIC_LABELS[metric][0])
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DSME-PSO simulation runner")
    parser.add_argument("--mode",    choices=["full", "iterations", "scalability"],
                        default="full")
    parser.add_argument("--metric",  choices=["power", "latency", "throughput"],
                        default="power")
    parser.add_argument("--nodes",   type=int, default=40)
    parser.add_argument("--iter",    type=int, default=35)
    parser.add_argument("--packet",  type=int, default=250,
                        help="Packet size in bytes (150–350)")
    parser.add_argument("--save",    type=str, default="results/figures/",
                        help="Directory to save plots")
    args = parser.parse_args()

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    network_sizes = [10, 20, 30, 40, 50, 60, 70, 80]

    if args.mode in ("full", "iterations"):
        metrics = ["power", "latency", "throughput"] if args.mode == "full" else [args.metric]
        print("\n── Iteration convergence experiments ──")

        iter_results = {}
        for m in metrics:
            print(f"  Running PSO [{m}] ...")
            res = run_pso_experiment(m, max_iter=args.iter, packet_size=args.packet)
            iter_results[m] = res
            print(f"    Best ({m}): {res['best_fit']:.4f}  at params BO={res['best_params'][0]}, "
                  f"MO={res['best_params'][1]}, SO={res['best_params'][2]}")

        # Save raw results
        with open(save_dir / "iteration_results.json", "w") as f:
            json.dump(iter_results, f, indent=2)

    if args.mode in ("full", "scalability"):
        metrics = ["power", "latency", "throughput"] if args.mode == "full" else [args.metric]
        print("\n── Scalability experiments (network size 10–80) ──")

        scale_results = {}
        for m in metrics:
            res = run_scalability_experiment(m, network_sizes, args.packet)
            scale_results[m] = res

        with open(save_dir / "scalability_results.json", "w") as f:
            json.dump(scale_results, f, indent=2)

    # ── Generate Figure 3 (all 6 subplots) ───────────────────────────────────
    if args.mode == "full":
        fig, axes = plt.subplots(3, 2, figsize=(12, 14))
        fig.suptitle(
            "PSO-DSME vs Standard DSME: QoS Comparison\n"
            "(Reproducing Figure 3 from Anand et al.)",
            fontsize=13, fontweight="bold", y=0.98
        )

        metric_subplot_titles = {
            "power":      ("(a) Power vs iterations", "(b) Power vs network size"),
            "latency":    ("(c) Latency vs iterations", "(d) Latency vs network size"),
            "throughput": ("(e) Throughput vs iterations", "(f) Throughput vs network size"),
        }

        for row, metric in enumerate(["power", "latency", "throughput"]):
            # Left: vs iterations
            history = iter_results[metric]["history"]["g_best_fitness"]
            baseline = simulate_standard_dsme(metric, packet_size=args.packet)
            plot_vs_iterations(
                axes[row, 0], history, baseline, metric,
                metric_subplot_titles[metric][0],
            )

            # Right: vs network size
            sr = scale_results[metric]
            plot_vs_network_size(
                axes[row, 1],
                sr["network_sizes"], sr["pso_means"], sr["pso_stds"], sr["dsme_means"],
                metric, metric_subplot_titles[metric][1],
            )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = save_dir / "figure3_reproduction.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n  ✓  Figure saved → {out}")
        plt.close()

    elif args.mode == "iterations":
        fig, ax = plt.subplots(figsize=(7, 4))
        history  = iter_results[args.metric]["history"]["g_best_fitness"]
        baseline = simulate_standard_dsme(args.metric, packet_size=args.packet)
        plot_vs_iterations(ax, history, baseline, args.metric,
                           f"{METRIC_LABELS[args.metric][1]} vs PSO Iterations")
        plt.tight_layout()
        out = save_dir / f"{args.metric}_vs_iterations.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n  ✓  Figure saved → {out}")
        plt.close()

    elif args.mode == "scalability":
        fig, ax = plt.subplots(figsize=(7, 4))
        sr = scale_results[args.metric]
        plot_vs_network_size(ax, sr["network_sizes"], sr["pso_means"],
                             sr["pso_stds"], sr["dsme_means"], args.metric,
                             f"{METRIC_LABELS[args.metric][1]} vs Network Size")
        plt.tight_layout()
        out = save_dir / f"{args.metric}_vs_network_size.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n  ✓  Figure saved → {out}")
        plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
