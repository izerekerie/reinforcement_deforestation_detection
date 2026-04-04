# UAV Deforestation Detection - Reinforcement Learning Agent

Reinforcement learning agent for autonomous UAV-based deforestation monitoring in Rwandan forests.
A drone navigates a 15x15 forest grid to detect illegal logging, deploy countermeasures, and maintain forest health above 70%.

## Project Structure

```
project_root/
├── environment/
│   ├── custom_env.py          -> forest_env.py (Gymnasium environment)
│   ├── forest_env.py          # Custom 15x15 grid environment
│   └── rendering.py           # Pygame 2D visualization
├── training/
│   ├── dqn_training.py        # DQN with 10 hyperparameter configs
│   └── pg_training.py         # PPO, A2C, REINFORCE (10 configs each)
├── agents/
│   ├── dqn_agent.py           # DQN single-run training
│   ├── ppo_agent.py           # PPO single-run training
│   ├── a2c_agent.py           # A2C single-run training
│   └── reinforce_agent.py     # Custom REINFORCE (PyTorch)
├── models/
│   ├── dqn/                   # Saved DQN models (10 configs)
│   └── pg/                    # Saved PPO, A2C, REINFORCE models
├── analysis/
│   └── compare.py             # Evaluation plots and tables
├── api/
│   └── server.py              # FastAPI serving trained agent
├── dashboard/
│   └── index.html             # Three.js live 3D web dashboard
├── visualization/
│   └── ursina_vis.py          # Ursina (Panda3D) 3D visualization
├── main.py                    # Entry point for best model
├── train.py                   # Quick training (all 4 algorithms)
├── random_agent.py            # Random agent demo (no training)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Random Agent Demo (no training needed)
```bash
python random_agent.py                # Terminal output
python random_agent.py --visual       # Pygame 2D visualization
```

### 2. Train All Algorithms (with hyperparameter tuning)
```bash
python training/dqn_training.py       # 10 DQN configs
python training/pg_training.py        # 10 PPO + 10 A2C + 10 REINFORCE configs
```

Or quick training (one config each):
```bash
python train.py
```

### 3. Run Best Trained Agent
```bash
python main.py --mode terminal        # Terminal output
python main.py --mode visual          # Pygame 2D
python main.py --mode 3d              # Ursina 3D
python main.py --mode dashboard       # Web dashboard (http://localhost:8000)
```

### 4. Generate Comparison Analysis
```bash
python analysis/compare.py
```

## Environment Details

- **Grid**: 15x15 tiles (Healthy, At-Risk, Deforesting, Destroyed)
- **Agent**: UAV drone with 200 fuel budget
- **Actions**: Move (4 dirs), Scan, Deploy Retardant, Call Rangers, Return to Base
- **Observation**: 5x5 local view + position, fuel, health, wind, truck location
- **Win**: Forest health >= 70% at episode end
- **Dynamics**: Deforestation spreads, logging truck moves, wind affects spread, rain slows it

## Algorithms

| Algorithm | Type | Library |
|-----------|------|---------|
| DQN | Value-Based | Stable Baselines3 |
| PPO | Policy Gradient | Stable Baselines3 |
| A2C | Actor-Critic | Stable Baselines3 |
| REINFORCE | Policy Gradient | Custom PyTorch |

## API Integration

The FastAPI server serializes the trained agent and environment state as JSON:
```bash
python main.py --mode dashboard
```
- `POST /reset` - Reset environment
- `POST /step` - Agent takes one action
- `GET /state` - Current environment state as JSON
- `POST /run_episode` - Run full episode, returns all states

This enables integration with any frontend or ranger monitoring dashboard.
