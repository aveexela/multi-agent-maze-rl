# multi-agent-maze-rl
Multi-agent reinforcement learning in procedurally generated mazes with custom reward shaping and DAgger training.
# Predator-Prey Coordination in Maze Environments

A reinforcement learning project exploring multi-agent pursuit strategies in partially observable, maze-like environments. Built for academic exploration of agent coordination, reward shaping, and imitation learning techniques.

## Project Goals

- Train a team of **5 predator agents** to efficiently **capture mobile prey** in procedurally generated 40x40 mazes.  
- Implement and benchmark **DAgger-style imitation learning** with curriculum-based exploration.  
- Introduce **behavioral shaping** rewards to reduce agent stalling, overlap, and inefficient wandering.

## Techniques Used

- **DAgger** imitation learning with dynamically scheduled teacher mixing  
- **Shared-parameter MLP policy** (PyTorch), trained via behavioral cloning and reward-driven refinement  
- **Dynamic role assignment & repulsion penalties** to reduce congestion in narrow maze corridors  
- **Toroidal map patches** and compact feature vectors invariant to maze size  
- **Custom reward shaping**: distance-based potential, exploration incentives, anti-flip-flop regularization  
- **Multi-metric checkpointing**: captures, first-capture step, cluster spread, idle time

## Environment Setup

- GridWorld simulation with static/dynamic prey  
- 5 available agent actions: stay, move up/down/left/right  
- Episodic termination after fixed steps or all prey captured  
- Toroidal boundary conditions with partial observability

## Key Results

- Achieved **>45% win rate** in PvP against ClosestTarget scripted bot  
- Agents demonstrate **coordinated dispersal**, efficient pathing, and **adaptive pursuit** behavior  
- Modular and scalable training system with logging, visualization, and evaluation scripts

## Structure
maze_rl_project/
├── world/ # Environment and map logic
├── agents/ # Agent models (shared MLP, policy heads, final weights)
├── training/ # DAgger training pipeline
├── logs/ # Checkpoints, heatmaps, metrics
├── Report.pdf # Project report
└── README.md

