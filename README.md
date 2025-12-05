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

<p align="center">
  <img src="https://github.com/user-attachments/assets/7b4169c1-6daf-4a80-bfe5-4eca4fd13265" width="30%">
  <img src="https://github.com/user-attachments/assets/29efe66c-a98f-453c-8f7a-8be9ca11357d" width="30%">
  <img src="https://github.com/user-attachments/assets/3eebc884-b55c-49ed-9d88-43dccd8d38d5" width="30%">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/dadd562c-8250-4811-9806-ffa0d9735222" width="30%">
  <img src="https://github.com/user-attachments/assets/cca4ec25-df85-4b43-9bc8-b84cf36e379d" width="30%">
  <img src="https://github.com/user-attachments/assets/ccb3d38a-e2b4-4fc0-8127-042a2618a1a1" width="30%">
</p>

<p align="center"><i>Blue is the trained model, red is the scripted “ClosestTarget” agent.</i></p>

## Results Summary

### Performance Metrics

| Phase           | Avg Captures | Avg Episode Length | Avg Uncaught Preys |
|----------------|--------------|---------------------|---------------------|
| BC-DAgger       | 741.5        | 175.2               | 0.19                |
| BC-DAgger-PvP   | 638.6        | 97.1                | 0.00                |

- **High capture rate:** Agents consistently capture nearly all targets across episodes.
- **Faster convergence:** PvP-trained models complete episodes 45% faster on average.
- **Winrate consistently near 100%**, with minimal leftover targets even on large maps.

## Structure
```text
maze_rl_project/
├── world/         # Environment and map logic
├── agents/        # Agent models (shared MLP, policy heads, final weights)
├── training/      # DAgger training pipeline
├── logs/          # Checkpoints, heatmaps, metrics
└── Report.pdf     # Project report
README.md      
```

