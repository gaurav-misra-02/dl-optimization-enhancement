"""
Evaluation utilities for comparing optimizers on benchmark problems.
"""

import copy
import numpy as np
import torch
from matplotlib import pyplot as plt


def run_optimizer(make_optimizer, problem, iterations, hyperparams):
    """
    Run an optimizer on a problem for a specified number of iterations.
    
    Args:
        make_optimizer: Function that creates optimizer (e.g., torch.optim.Adam)
        problem: Problem dictionary with 'model0' and 'obj_function'
        iterations: Number of optimization steps
        hyperparams: Dictionary of hyperparameters for the optimizer
    
    Returns:
        values: Array of objective values at each iteration
        trajectory: List of model snapshots at each iteration
    """
    # Initialize with problem's starting point
    model = copy.deepcopy(problem["model0"])
    obj_function = problem["obj_function"]
    
    # Create optimizer
    optimizer = make_optimizer(model.parameters(), **hyperparams)
    
    # Track objective values and model trajectory
    values = []
    trajectory = []
    
    def closure():
        """Closure for computing objective and gradients."""
        trajectory.append(copy.deepcopy(model))
        optimizer.zero_grad()
        
        obj_value = obj_function(model)
        obj_value.backward()
        
        values.append(obj_value.item())
        return obj_value
    
    # Run optimization
    for i in range(iterations):
        optimizer.step(closure)
        
        # Stop if we encounter NaN or Inf
        if np.isnan(values[-1]) or np.isinf(values[-1]):
            print(f"Warning: Encountered {'NaN' if np.isnan(values[-1]) else 'Inf'} "
                  f"at iteration {i}")
            break
    
    # Replace NaN/Inf with large value for plotting
    values = np.nan_to_num(values, nan=1e6, posinf=1e6, neginf=-1e6)
    
    return values, trajectory


def accuracy(model, x, y):
    """
    Calculate classification accuracy for binary classification.
    
    Args:
        model: PyTorch model that outputs probabilities
        x: Input features
        y: True labels (0 or 1)
    
    Returns:
        Accuracy as a fraction in [0, 1]
    """
    with torch.no_grad():
        predictions = (model(x).view(-1) > 0.5).float()
        correct = (predictions == y).float().mean()
    return correct.item()


def run_all_optimizers(problem, iterations, tune_dict, policy):
    """
    Run multiple optimizers on a problem and compare their performance.
    
    Args:
        problem: Problem dictionary
        iterations: Number of iterations for each optimizer
        tune_dict: Dictionary mapping optimizer names to their hyperparameters
        policy: Trained RL policy for autonomous optimizer
    
    Returns:
        Dictionary mapping optimizer names to (values, trajectory) tuples
    """
    results = {}
    
    # SGD
    if "sgd" in tune_dict:
        sgd_vals, sgd_traj = run_optimizer(
            torch.optim.SGD, problem, iterations, tune_dict["sgd"]["hyperparams"]
        )
        print(f"SGD best loss: {sgd_vals.min():.6f}")
        results["sgd"] = (sgd_vals, sgd_traj)
    
    # Momentum (SGD with Nesterov momentum)
    if "momentum" in tune_dict:
        momentum_vals, momentum_traj = run_optimizer(
            torch.optim.SGD, problem, iterations, tune_dict["momentum"]["hyperparams"]
        )
        print(f"Momentum best loss: {momentum_vals.min():.6f}")
        results["momentum"] = (momentum_vals, momentum_traj)
    
    # Adam
    if "adam" in tune_dict:
        adam_vals, adam_traj = run_optimizer(
            torch.optim.Adam, problem, iterations, tune_dict["adam"]["hyperparams"]
        )
        print(f"Adam best loss: {adam_vals.min():.6f}")
        results["adam"] = (adam_vals, adam_traj)
    
    # LBFGS
    if "lbfgs" in tune_dict:
        lbfgs_vals, lbfgs_traj = run_optimizer(
            torch.optim.LBFGS, problem, iterations, tune_dict["lbfgs"]["hyperparams"]
        )
        print(f"LBFGS best loss: {lbfgs_vals.min():.6f}")
        results["lbfgs"] = (lbfgs_vals, lbfgs_traj)
    
    # Autonomous optimizer
    if policy is not None:
        from ..optimizer import AutonomousOptimizer
        
        ao_vals, ao_traj = run_optimizer(
            AutonomousOptimizer,
            problem,
            iterations,
            {"policy": policy},
        )
        print(f"Autonomous Optimizer best loss: {ao_vals.min():.6f}")
        results["ao"] = (ao_vals, ao_traj)
    
    return results


def plot_trajectories(trajectories, problem, get_weights, set_weights, save_path=None):
    """
    Plot optimization trajectories on a 2D contour plot of the objective.
    
    Args:
        trajectories: Dictionary mapping optimizer names to list of models
        problem: Problem dictionary with 'model0' and 'obj_function'
        get_weights: Function that extracts (w1, w2) from a model
        set_weights: Function that sets (w1, w2) in a model
        save_path: Optional path to save the figure
    """
    # Extract trajectory coordinates
    data = {}
    for name, traj in trajectories.items():
        data[name] = np.array([get_weights(model) for model in traj])
    
    # Determine plot bounds
    all_coords = np.concatenate(list(data.values()))
    xmin, xmax = all_coords[:, 0].min(), all_coords[:, 0].max()
    ymin, ymax = all_coords[:, 1].min(), all_coords[:, 1].max()
    
    # Add padding
    x_padding = (xmax - xmin) * 0.2
    y_padding = (ymax - ymin) * 0.2
    
    X = np.linspace(xmin - x_padding, xmax + x_padding, 100)
    Y = np.linspace(ymin - y_padding, ymax + y_padding, 100)
    
    # Compute objective function on grid
    model = copy.deepcopy(problem["model0"])
    Z = np.empty((len(Y), len(X)))
    
    for i in range(len(X)):
        for j in range(len(Y)):
            set_weights(model, X[i], Y[j])
            Z[j, i] = problem["obj_function"](model).item()
    
    # Create plot
    plt.figure(figsize=(10, 8), dpi=150)
    plt.contourf(X, Y, Z, levels=30, cmap="RdGy")
    plt.colorbar(label="Objective Value")
    
    # Plot trajectories
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan']
    for idx, (name, traj) in enumerate(data.items()):
        color = colors[idx % len(colors)]
        plt.plot(traj[:, 0], traj[:, 1], '-o', label=name, 
                color=color, linewidth=2, markersize=4, alpha=0.7)
    
    # Plot starting point
    start_x, start_y = get_weights(problem["model0"])
    plt.plot(start_x, start_y, 'k*', markersize=15, label='Start', zorder=10)
    
    plt.xlabel('Parameter 1', fontsize=12)
    plt.ylabel('Parameter 2', fontsize=12)
    plt.title('Optimization Trajectories', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")
    else:
        plt.show()
    
    plt.close()

