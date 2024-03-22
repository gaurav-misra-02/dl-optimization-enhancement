"""
RL-based optimizer for quadratic optimization problems.

This module implements a reinforcement learning approach specifically
designed for optimizing convex quadratic functions.
"""

from .environment import QuadraticEnv
from .actor import ActorNetwork
from .ppo import PPOTrainer
from .utils import create_quadratic_problem, rollout, compute_rewards, calculate_gaes

__all__ = [
    'QuadraticEnv',
    'ActorNetwork',
    'PPOTrainer',
    'create_quadratic_problem',
    'rollout',
    'compute_rewards',
    'calculate_gaes'
]

