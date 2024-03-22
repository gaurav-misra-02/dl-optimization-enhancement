# Autonomous Optimizer

This directory contains a reinforcement learning-based optimizer that learns to optimize objective functions by training on a distribution of optimization problems.

## Overview

The Autonomous Optimizer treats optimization as a sequential decision-making problem. An RL agent learns a policy that maps from the history of objective values and gradients to parameter updates. This approach can potentially adapt to different problem structures and learn optimization strategies beyond hand-designed update rules.

### Key Idea

Instead of using fixed update rules (like Adam or SGD), the optimizer:
1. Observes recent history of objective values and gradients
2. Uses a learned policy to decide parameter updates
3. Is trained via PPO on a distribution of optimization problems

## Directory Structure

```
autonomous_optimizer/
├── optimizer.py              # Main autonomous optimizer class
├── environment.py            # Gym environment for training
├── benchmarks/              # Benchmark problems and evaluation
│   ├── problems.py          # Problem definitions
│   └── evaluation.py        # Comparison utilities
├── experiments/             # Training and evaluation scripts
│   ├── train_policy.py      # Train policy on problems
│   └── run_benchmarks.py    # Compare with standard optimizers
└── rl_quadratic/            # Specialized quadratic optimizer
    ├── environment.py       # Quadratic-specific environment
    ├── actor.py             # Actor network
    ├── ppo.py               # PPO trainer
    ├── utils.py             # Utility functions
    └── train.py             # Training script
```

## Autonomous Optimizer (General)

The general autonomous optimizer can be trained on various problem types and used as a drop-in replacement for standard optimizers.

### Training a Policy

Train on convex quadratic problems:

```bash
cd experiments
python train_policy.py --problem-type quadratic --num-problems 90 --num-passes 20
```

Train on logistic regression:

```bash
python train_policy.py --problem-type logistic --num-problems 90 --num-passes 20
```

Available problem types:
- `quadratic` - Convex quadratic functions
- `logistic` - Logistic regression
- `robust_linear` - Robust linear regression with Geman-McClure loss
- `mlp` - 2-layer neural network classification

### Running Benchmarks

Compare trained policy against standard optimizers:

```bash
python run_benchmarks.py --policy-path policy_quadratic.zip --benchmark all --save-plots
```

### Using the Optimizer

```python
from stable_baselines3 import PPO
from autonomous_optimizer.optimizer import AutonomousOptimizer

# Load trained policy
policy = PPO.load("policy_quadratic.zip")

# Use as optimizer
optimizer = AutonomousOptimizer(model.parameters(), policy=policy)

# Optimization loop
def closure():
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    return loss

for step in range(num_steps):
    optimizer.step(closure)
```

## RL Quadratic Optimizer

A specialized implementation for quadratic optimization using PPO directly (without stable-baselines3).

### Training

```bash
cd rl_quadratic
python train.py --n-episodes 90 --lr 1e-3 --save-model quadratic_model.pth
```

Arguments:
- `--n-episodes` - Number of training episodes (default: 90)
- `--lr` - Learning rate (default: 1e-3)
- `--clip` - PPO clipping parameter (default: 0.2)
- `--hidden-dim` - Actor network hidden dimension (default: 64)
- `--save-model` - Path to save trained model

### Components

**Environment** (`environment.py`):
- Optimizes f(x) = 0.5 * x^T @ Q @ x + c^T @ x
- State: Current parameters + objective value
- Action: Parameter update
- Reward: Reduction in objective value

**Actor Network** (`actor.py`):
- Outputs mean and variance for Gaussian policy
- Two-layer MLP with shared feature extraction

**PPO Trainer** (`ppo.py`):
- Proximal Policy Optimization implementation
- Clipped surrogate objective
- Early stopping based on KL divergence

## Benchmark Problems

The benchmarks module provides standard test problems:

### Convex Quadratic
```python
from benchmarks import convex_quadratic

problem = convex_quadratic()
# Returns: model0, obj_function, optimal_x, optimal_val, A, b
```

### Rosenbrock Function
```python
from benchmarks import rosenbrock

problem = rosenbrock()
# Returns: model0, obj_function, optimal_x, optimal_val
```

### Logistic Regression
```python
from benchmarks import logistic_regression

problem = logistic_regression()
# Returns: model0, obj_function, data
```

### Robust Linear Regression
```python
from benchmarks import robust_linear_regression

problem = robust_linear_regression()
# Returns: model0, obj_function
```

### Multi-Layer Perceptron
```python
from benchmarks import mlp

problem = mlp()
# Returns: model0, obj_function, dataset
```

## Evaluation

The evaluation module provides utilities for comparing optimizers:

```python
from benchmarks import run_all_optimizers, plot_trajectories

tune_dict = {
    "sgd": {"hyperparams": {"lr": 0.05}},
    "adam": {"hyperparams": {"lr": 0.1}},
    # ... other optimizers
}

results = run_all_optimizers(problem, iterations=40, tune_dict=tune_dict, policy=policy)

# Plot trajectories for 2D problems
plot_trajectories(
    {name: traj for name, (vals, traj) in results.items()},
    problem,
    get_weights=lambda m: (m.x[0].item(), m.x[1].item()),
    set_weights=lambda m, w1, w2: setattr(m.x, 'data', torch.tensor([w1, w2]))
)
```

## Requirements

- PyTorch >= 1.7.0
- stable-baselines3 >= 0.10.0
- gymnasium
- gym
- numpy
- scipy
- matplotlib

## Results Summary

Based on benchmark experiments:

**Convex Quadratic**: Competitive with standard optimizers, though not consistently better
**Logistic Regression**: Achieves best accuracy (86% vs 83% for Adam/Momentum)
**Robust Linear Regression**: Competitive performance (0.456 vs 0.452 for Adam)
**MLP Classification**: Struggles compared to Adam/L-BFGS (75% vs 84%)

## Key Observations

1. The learned optimizer shows promise on specific problem types
2. Best performance on logistic regression tasks
3. Struggles with more complex non-convex problems
4. Hyperparameter tuning of baseline optimizers matters significantly
5. Policy generalization across problem types is challenging

## Future Directions

- Train on broader distribution of problems
- Explore curriculum learning strategies
- Investigate meta-learning approaches
- Test on larger-scale problems
- Incorporate problem-specific features in observations

