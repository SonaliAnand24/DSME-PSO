# PSO-Optimized IEEE 802.15.4-DSME for IoT Networks

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Published-brightgreen?style=flat-square)
![Venue](https://img.shields.io/badge/Venue-IEEE-00629B?style=flat-square&logo=ieee&logoColor=white)

**Simulation code for the paper:**

*"Improving Network Efficiency in Clustered Tree Topology through PSO Optimization in IEEE 802.15.4-DSME based IoT Networks"*

**Sonali Anand**, Nikumani Choudhury, Tamoghna Ojha, Anakhi Hazarika, Jay Dave

BITS Pilani, Hyderabad Campus · IIT (ISM) Dhanbad

*Supported by DST-SERB Startup Research Grant SRG/2023/002016*

</div>

---

## Abstract

IEEE 802.15.4 DSME (Deterministic and Synchronous Multi-channel Extension) is a key MAC protocol for low-rate IoT networks. However, under dynamic network conditions, static multi-superframe parameter settings (BO, MO, SO) lead to excessive power consumption, high latency, and degraded throughput.

We propose a **PSO-based adaptive parameter tuning mechanism** that dynamically optimises BO, SO, and MO to minimise power consumption while satisfying QoS constraints. Simulations on a cluster-tree topology (10–80 nodes) demonstrate significant improvements over standard DSME across all three metrics.

---

## Key Results

| Metric | Standard DSME | PSO-DSME (proposed) | Improvement |
|--------|--------------|---------------------|-------------|
| Avg. Cluster Power (mW) | ~215 mW | ~188 mW | **~12.5% reduction** |
| Latency (ms) | ~7,500 ms | ~2,200 ms | **~71% reduction** |
| Throughput (bps) | ~4 bps | ~9 bps | **~125% increase** |
| Convergence | — | 12–20 iterations | — |

> Results at network size = 40 nodes. See `results/` for full plots.

---

## Problem Formulation

### Multi-Superframe Structure

A DSME multi-superframe is governed by three parameters:

```
SD = aBaseSuperframeDuration × 2^SO   (Superframe Duration)
MD = aBaseSuperframeDuration × 2^MO   (Multi-superframe Duration)  
BI = aBaseSuperframeDuration × 2^BO   (Beacon Interval)

Constraint: 0 ≤ SO ≤ MO ≤ BO ≤ 14
```

### Fitness Function — Power Consumption

The average power consumed by a single multi-superframe:

```
P = 2^(MO−SO) × {(Ptx × Ttx) + (Prx × Trx) + (Pidle × Tidle)} / TMD
```

where `Ptx = 255 mW`, `Prx = 135 mW`, `Pidle = 1.3 mW` (per IEEE 802.15.4 spec).

### Delay Model

```
DMSF = 2^(MO−SO) × {TA + 2·TACK + 3δ + 3·SIFS + Tidle}
```

PSO simultaneously optimises both `P` and `DMSF` by finding the optimal `(BO, MO, SO)` triple for the current network state.

---

## System Architecture

```
Cluster-Tree Topology
│
├── PAN Coordinator (PANC) — root; maintains global best (gBest)
│   ├── Cluster 1
│   │   ├── Cluster Head (CH) — runs PSO locally; reports pBest to PANC
│   │   ├── End Device 1
│   │   └── End Device 2
│   ├── Cluster 2
│   │   ├── Cluster Head
│   │   └── End Devices ...
│   └── ...
│
PSO Loop (per Cluster Head):
  1. Initialise swarm with random (BO, MO, SO) positions
  2. Evaluate fitness P and DMSF for each particle
  3. Update pBest per particle
  4. Report to PANC → PANC updates gBest
  5. Broadcast gBest to all Cluster Heads
  6. Update velocity: v(t+1) = w·v(t) + c1·r1·(pBest−x) + c2·r2·(gBest−x)
  7. Update position: x(t+1) = x(t) + v(t+1)
  8. Terminate when SO ≤ MO ≤ BO violated OR max iterations reached
```

---

## Repository Structure

```
dsme-pso/
│
├── 📂 src/
│   ├── pso/
│   │   ├── pso_optimizer.py        # Core PSO engine (velocity, position update)
│   │   └── fitness.py              # Power, delay, throughput fitness functions
│   ├── dsme/
│   │   ├── superframe.py           # Multi-superframe structure & parameter model
│   │   └── mac_layer.py            # DSME MAC simulation (CAP, CFP, GTS)
│   ├── network/
│   │   ├── topology.py             # Cluster-tree topology builder
│   │   ├── cluster_head.py         # CH logic: local PSO + pBest reporting
│   │   └── pan_coordinator.py      # PANC: gBest aggregation & broadcast
│   └── metrics/
│       └── evaluator.py            # Power, latency, throughput measurement
│
├── 📂 configs/
│   ├── default.yaml                # Default simulation parameters
│   └── scalability.yaml            # Config for 10–80 node scalability tests
│
├── 📂 results/
│   ├── figures/                    # Reproduced plots (Fig. 3a–3f from paper)
│   └── logs/                       # Raw simulation output (JSON)
│
├── 📂 notebooks/
│   ├── 01_pso_convergence.ipynb    # Visualise PSO convergence over iterations
│   ├── 02_scalability_analysis.ipynb  # Power/latency/throughput vs network size
│   └── 03_parameter_sensitivity.ipynb # Impact of BO, MO, SO individually
│
├── 📂 docs/
│   └── DSME_PRIMER.md              # Background on DSME superframe structure
│
├── run_simulation.py               # Main entry point
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/dsme-pso-iot.git
cd dsme-pso-iot
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, NumPy, Matplotlib, PyYAML, tqdm

---

## Running Simulations

### Reproduce paper results (Fig. 3a–3f)

```bash
# Power consumption vs PSO iterations (Fig. 3a)
python run_simulation.py --mode iterations --metric power --nodes 40

# Power consumption vs network size (Fig. 3b)
python run_simulation.py --mode scalability --metric power

# Latency vs iterations (Fig. 3c)
python run_simulation.py --mode iterations --metric latency --nodes 40

# All six plots at once
python run_simulation.py --mode full --save results/figures/
```

### Custom configuration

```bash
python run_simulation.py \
  --config configs/default.yaml \
  --nodes 50 \
  --iterations 30 \
  --swarm_size 20 \
  --packet_size 250
```

### Parameter sweep

```bash
python run_simulation.py --mode sweep \
  --bo_range 4 8 \
  --mo_range 3 7 \
  --so_range 1 4
```

---

## Simulation Parameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Tx power | Ptx | 255 mW |
| Rx power | Prx | 135 mW |
| Idle power | Pidle | 1.3 mW |
| Data rate | — | 650 kb/s |
| ACK size | — | 11 B |
| Propagation delay | δ | 1 µs |
| SIFS | — | 160 µs |
| PSO inertia | w | 0.7 |
| Cognitive coeff. | c1 | 1.5 |
| Social coeff. | c2 | 1.5 |
| Swarm size | — | 20 particles |
| Max iterations | — | 35 |
| Packet size range | — | 150–350 bytes |
| Network sizes | — | 10, 20, …, 80 nodes |

---

## How PSO Works Here

The key insight is that `(BO, MO, SO)` is a 3-dimensional integer search space. Each PSO particle represents one candidate configuration. The swarm collectively searches for the configuration that minimises power consumption subject to the constraint `SO ≤ MO ≤ BO ≤ 14`.

```python
# Velocity update (Eq. 7 from paper)
v[t+1] = w * v[t] \
        + c1 * r1 * (pBest - x[t]) \   # cognitive: pull toward personal best
        + c2 * r2 * (gBest - x[t])     # social: pull toward swarm's best

# Position update (Eq. 8)
x[t+1] = x[t] + v[t+1]

# Constraint enforcement
if not (SO <= MO <= BO <= 14):
    terminate and return last valid gBest
```

The distributed execution on Cluster Heads (rather than the PANC) reduces communication overhead while still achieving global optimality through the gBest broadcast mechanism.

---

## Citation

If you build upon this work, please cite the paper:

```bibtex
@inproceedings{anand2024dsme,
  author    = {Anand, Sonali and Choudhury, Nikumani and Ojha, Tamoghna
               and Hazarika, Anakhi and Dave, Jay},
  title     = {Improving Network Efficiency in Clustered Tree Topology
               through {PSO} Optimization in {IEEE} 802.15.4-{DSME}
               based {IoT} Networks},
  booktitle = {[IEEE ANTS]},
  year      = {2024},
  note      = {Supported by DST-SERB Grant SRG/2023/002016}
}
```

---

## Authors

### Lead Author & Maintainer

**Sonali Anand**
MTech, AI

[![Email](https://img.shields.io/badge/Email-sonalianand2406%40gmail.com-EA4335?style=flat-square&logo=gmail)](mailto:sonalianand2406@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-YOUR__USERNAME-181717?style=flat-square&logo=github)](https://github.com/YOUR_USERNAME)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)

### Co-Authors

**Nikumani Choudhury** · BITS Pilani, Hyderabad Campus *(Supervisor)*

**Tamoghna Ojha** · IIT (ISM) Dhanbad

**Anakhi Hazarika** · BITS Pilani, Hyderabad Campus

**Jay Dave** · BITS Pilani, Hyderabad Campus

---

## Acknowledgement

This work is supported by the **Science and Engineering Research Board, Department of Science and Technology, Government of India** through the Startup Research Grant under Grant **SRG/2023/002016**.

---

<div align="center">
<sub>IEEE 802.15.4 · DSME · IoT · PSO · MAC Optimisation · Wireless Sensor Networks</sub>
</div>
