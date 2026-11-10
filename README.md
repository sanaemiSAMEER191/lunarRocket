# 🚀 lunarRocket
### Reinforcement Learning for Autonomous Lunar Landing using PPO

An experimental Reinforcement Learning project that trains an autonomous agent to perform lunar landings using **Proximal Policy Optimization (PPO)** and the Gymnasium LunarLander-v3 environment.

The project explores policy learning, reward optimization, hyperparameter experimentation, and agent evaluation.

---

## Overview

Autonomous lunar landing is a control problem in which an agent must learn to navigate a lander towards a designated landing zone while controlling its position, velocity, and orientation.

This project uses Reinforcement Learning to train an agent that learns landing behaviour through repeated interactions with a simulated environment.

**Key objectives:**
- Understand and implement PPO-based Reinforcement Learning.
- Train an agent for autonomous lunar landing.
- Investigate the effect of learning rate on model performance.
- Analyse training behaviour using TensorBoard.
- Explore reward shaping and its influence on agent behaviour.

## Environment

Built using Gymnasium's `LunarLander-v3`.

**Observation space:** 8-dimensional state vector.

| Observation | Description |
|---|---|
| x | Horizontal position |
| y | Vertical position |
| vx | Horizontal velocity |
| vy | Vertical velocity |
| angle | Lander's orientation |
| angular velocity | Rotational velocity |
| left leg contact | Left landing leg contact indicator |
| right leg contact | Right landing leg contact indicator |

**Action space:** Four discrete actions.

| Action | Description |
|---|---|
| 0 | Do nothing |
| 1 | Fire left orientation engine |
| 2 | Fire main engine |
| 3 | Fire right orientation engine |

## Algorithm: Proximal Policy Optimization (PPO)

The project uses PPO, an on-policy policy-gradient algorithm.

The agent learns a policy that maps environment observations to actions.

The PPO implementation uses:
- Actor-Critic architecture.
- Clipped policy objective.
- Generalized Advantage Estimation (GAE).
- Separate policy and value network layers.

**Network architecture:**

- Policy network: 64 → 64 neurons
- Value network: 64 → 64 neurons

## Project Structure

```text
lunarRocket/
├── configs/
├── data/
├── docs/
│   ├── design-decisions/
│   ├── figures/
│   └── experiment-analysis.md
├── experiments/
│   ├── baseline/
│   ├── ppo_lr_1e4/
│   └── comparison_results.json
├── models/
│   ├── checkpoints/
│   ├── ppo_lr_1e4/
│   └── ppo_lunar_lander_baseline.zip
├── notebooks/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── compare_experiments.py
│   └── train_lr_experiment.py
├── src/
│   └── lunar_lander/
│       ├── agents/
│       ├── environment/
│       ├── evaluation/
│       ├── rewards/
│       ├── training/
│       └── utils/
├── tests/
└── pyproject.toml
```

## Installation

**Requirements:** Python 3.10 or newer.

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd lunarRocket
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -e ".[dev]"
```

## Training

Train the baseline PPO agent:

```bash
python scripts/train.py
```

The baseline configuration uses:

| Hyperparameter | Value |
|---|---:|
| Algorithm | PPO |
| Learning rate | 3e-4 |
| Timesteps | 3,000,000 |
| Batch size | 64 |
| Rollout steps | 2048 |
| Epochs | 10 |
| Discount factor | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.2 |
| Random seed | 42 |

## Experiments

Two independently trained PPO configurations were evaluated.

| Configuration | Learning rate | Training steps |
|---|---:|---:|
| Baseline | 3e-4 | 3M |
| Experiment 2 | 1e-4 | 3M |

All other PPO hyperparameters were kept unchanged.

### Evaluation Results

Both models were evaluated over 100 episodes using seeds 100–199.

| Metric | Baseline | Lower LR |
|---|---:|---:|
| Mean reward | 172.086 | 110.515 |
| Reward standard deviation | 109.523 | 121.153 |
| Mean episode length | 308.910 | 293.180 |
| Success rate | **75%** | 56% |
| Successful landings | 75 | 56 |

### Learning Curves

#### Episode Reward

![PPO episode reward learning curves](docs/figures/learning_curves.png)

#### Episode Length

![PPO episode length learning curves](docs/figures/episode_lengths.png)

### Experimental Conclusion

Under the tested conditions, the baseline configuration with a learning rate of 3e-4 outperformed the lower learning rate configuration.

It achieved a 19 percentage-point improvement in successful landings.

This experiment demonstrates that hyperparameter changes can significantly influence PPO training and evaluation performance.

## Reward Engineering

The project also includes a configurable reward wrapper that introduces a per-step penalty.

This allows investigation of how modifying the reward objective can influence agent behaviour.

The wrapper is implemented separately from the baseline environment. Its effect has not yet been validated through a completed, comparable training and evaluation experiment.

## Evaluation

The evaluation module measures:

- Mean episode reward.
- Standard deviation of episode rewards.
- Mean episode length.
- Successful landing rate.
- Total successful landings.

## TensorBoard Monitoring

Launch TensorBoard:

```bash
tensorboard --logdir experiments
```

Open:

```text
http://localhost:6006
```

Training metrics can be monitored through TensorBoard's scalar visualizations.

## Testing

Run the test suite:

```bash
pytest
```

## Tech Stack

- Python
- Gymnasium
- Stable-Baselines3
- PyTorch
- NumPy
- TensorBoard
- Pytest

## Future Improvements

- Implement and experimentally validate potential-based reward shaping.
- Investigate PPO entropy regularization.
- Compare additional PPO hyperparameters.
- Improve evaluation across multiple training seeds.
- Analyse agent behaviour through rendered landing trajectories.

---

**Project:** lunarRocket  
**Domain:** Reinforcement Learning | Deep Learning | Autonomous Control