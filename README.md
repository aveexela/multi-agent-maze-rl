# multi-agent-maze-rl

Multi-agent reinforcement learning in procedurally generated mazes with custom reward shaping and DAgger training.

# Predator–Prey Coordination in Maze Environments

A reinforcement learning project exploring multi-agent pursuit strategies in partially observable, maze-like environments. Built for academic exploration of agent coordination, reward shaping, and imitation learning techniques.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7b4169c1-6daf-4a80-bfe5-4eca4fd13265" width="30%">
  <img src="https://github.com/user-attachments/assets/29efe66c-a98f-453c-8f7a-8be9ca11357d" width="30%">
  <img src="https://github.com/user-attachments/assets/3eebc884-b55c-49ed-9d88-43dccd8d38d5" width="30%">
  <br/>
  <img src="https://github.com/user-attachments/assets/dadd562c-8250-4811-9806-ffa0d9735222" width="30%">
  <img src="https://github.com/user-attachments/assets/cca4ec25-df85-4b43-9bc8-b84cf36e379d" width="30%">
  <img src="https://github.com/user-attachments/assets/ccb3d38a-e2b4-4fc0-8127-042a2618a1a1" width="30%">
</p>

<p align="center"><i>Blue = trained model, Red = scripted “ClosestTarget” agent.</i></p>

---

## Table of Contents

* [Goals](#goals)
* [Techniques](#techniques)
* [Environment](#environment)
* [Results](#results)
* [Quickstart](#quickstart)
* [Configuration](#configuration)
* [Logging & Visualization](#logging--visualization)
* [Repository Structure](#repository-structure)
* [Technical Notes](#technical-notes)
* [FAQ](#faq)
* [Citation](#citation)
* [License](#license)

---

## Goals

* Train a team of **5 predator agents** to efficiently **capture mobile prey** in procedurally generated **40×40** mazes.
* Implement and benchmark **DAgger-style** imitation learning with curriculum-based exploration.
* Introduce **behavioral shaping** rewards to reduce stalling, overlap, and inefficient wandering.

## Techniques

* **DAgger** imitation learning with dynamically scheduled teacher mixing.
* **Shared-parameter** MLP policy (PyTorch), trained via behavior cloning and reward-driven refinement.
* **Dynamic role assignment** & **repulsion** penalties to reduce congestion in narrow corridors.
* **Toroidal** map patches and compact, maze-size–invariant feature vectors.
* **Custom reward shaping**: distance-based potential, exploration incentives, anti flip-flop regularization.
* **Multi-metric checkpointing**: captures, first-capture step, cluster spread, idle time.

## Environment

* GridWorld simulation with **dynamic prey**.
* **5 actions** per agent: stay, up, down, left, right.
* **Partial observability** and **toroidal** boundary conditions.
* Episodes terminate after a step budget or when all prey are captured.

---

## Results

### Key Highlights

* **>45% win rate** in PvP against the **ClosestTarget** scripted bot.
* Agents show **coordinated dispersal**, efficient pathing, and **adaptive pursuit**.
* Modular training pipeline with logging, visualization, and eval scripts.

### Summary Table

| Phase         | Avg Captures | Avg Episode Length | Avg Uncaught Prey |
| ------------- | ------------ | ------------------ | ----------------- |
| BC-DAgger     | 741.5        | 175.2              | 0.19              |
| BC-DAgger-PvP | 638.6        | 97.1               | 0.00              |

> Notes: “Avg Captures”/“Uncaught Prey” aggregate across evaluation seeds/configs; episode length is in steps.

---

## Quickstart

> Requires **Python 3.9+**. GPU is optional but recommended for training.

```bash
# 1) Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 2) Install dependencies
pip install -U pip
pip install -r requirements.txt  # if provided
# Minimal stack (example):
# pip install torch numpy matplotlib tqdm

# 3) Train & evaluate (example entry points – adjust to your repo layout)
# DAgger / BC training
python -m training.run_dagger --map-size 40 --team-size 5 --steps 300 --log-dir ./logs

# PvP evaluation
python -m training.eval_pvp --model ./logs/best.pt --episodes 50

# Visualizations
python -m training.render --trace ./logs/episodes --out ./logs/figs
```

---

## Configuration

Common flags / settings (defaults shown as an example):

* `--team-size` (default: `5`) — number of predators.
* `--map-size` (default: `40`) — square maze size.
* `--prey` (e.g., `mobile|static`) — prey dynamics.
* `--steps` (default: `300`) — episode step budget.
* `--lr` (default: `1e-4`) — learning rate.
* `--seed` — random seed for reproducibility.
* `--log-dir` — base folder for checkpoints and artifacts.

**Reward shaping knobs** (typical):

* Exploration bonus (lower to avoid wandering)
* Stand-still penalty
* Revisit penalty
* Repulsion between teammates (same cell / adjacent / radius-2)
* Anti-oscillation (flip-flop) & anti lockstep
* Potential-based shaping (distance-to-target)

---

## Logging & Visualization

Artifacts under `logs/` typically include:

* **Checkpoints** — model weights & training state.
* **Per-step CSV** (optional) — phase, episode, step; positions; executed vs teacher action; prey counters; team score; reward decomposition; diagnostics (idle count, dispersion); nearest-prey coordinates; newly caught prey coordinates.
* **Frames / GIFs** — episodic renders and heatmaps (visited cells per agent and team).
* **Maps** — generated mazes (raw and preprocessed).

Utilities:

* `make_color_gif` — build animated GIFs from recorded frames.
* `visualize_visited_map` — single-agent heatmap with path & prey overlays.
* `visualize_team_map` — team-level heatmap with multiple tracks and markers.

> Toroidal geometry is respected across movement, features, and visualization (wrap-around at borders).

---

## Repository Structure

```text
maze_rl_project/
├── world/         # Environment and map logic
├── agents/        # Agent models (shared MLP, policy heads, final weights)
├── training/      # DAgger training pipeline
├── logs/          # Checkpoints, heatmaps, metrics
└── Report.pdf     # Project report
README.md
```

---

## Technical Notes

**Feature builders**

* Per-agent fixed-length feature vectors independent of maze size.
* Local passability patch (toroidal wrap), K-nearest prey & teammates with normalized deltas and distances.
* BFS-derived shortest paths for prey distances on a torus.

**Policies**

* Compact MLP policy with value head; action masking enforces passability.
* Shared-weights agent controls all K predators; supports greedy and stochastic sampling.

**Scripted baselines**

* `ClosestTarget` and an assigned variant with hysteresis (hold window + switch margin) to reduce thrashing and crowding.

**Geometry**

* Consistent `[y, x]` coordinate order; toroidal Manhattan distance; utilities for BFS grids and torus-aware patches.

---

## Citation

If this repository is useful, you may cite:

```bibtex
@misc{multiagent_maze_rl,
  title        = {Multi-agent Maze RL: Predator-Prey Coordination with DAgger and Reward Shaping},
  author       = {Valentina Alekseeva},
  year         = {2025},
  howpublished = {\url{https://github.com/aveexela/multi-agent-maze-rl}},
  note         = {Version 1.0}
}
```
