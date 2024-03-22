"""Benchmark problems for optimization."""

from .problems import (
    convex_quadratic,
    rosenbrock,
    logistic_regression,
    robust_linear_regression,
    mlp
)

from .evaluation import (
    run_optimizer,
    run_all_optimizers,
    accuracy,
    plot_trajectories
)

__all__ = [
    'convex_quadratic',
    'rosenbrock',
    'logistic_regression',
    'robust_linear_regression',
    'mlp',
    'run_optimizer',
    'run_all_optimizers',
    'accuracy',
    'plot_trajectories'
]

