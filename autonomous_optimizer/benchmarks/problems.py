"""
Benchmark optimization problems for testing optimizers.

This module provides various test problems including convex quadratics,
non-convex functions, and machine learning tasks.
"""

import numpy as np
import scipy.linalg
import scipy.stats
import torch
from torch import nn
from torch.nn import functional as F


class Variable(nn.Module):
    """A wrapper to turn a tensor of parameters into a module for optimization."""
    
    def __init__(self, data: torch.Tensor):
        """Create Variable holding `data` tensor."""
        super().__init__()
        self.x = nn.Parameter(data)


def convex_quadratic():
    """
    Generate a convex quadratic optimization problem.
    
    Objective: f(x) = 0.5 * x^T @ A @ x + b^T @ x
    where A is symmetric positive definite with eigenvalues in [1, 30].
    
    Returns:
        Dictionary containing:
        - model0: Initial model (Variable)
        - obj_function: Objective function
        - optimal_x: Optimal solution
        - optimal_val: Optimal objective value
        - A, b: Problem matrices
    """
    num_vars = 2
    
    # Generate orthogonal matrix of eigenvectors
    eig_vecs = torch.tensor(
        scipy.stats.ortho_group.rvs(dim=(num_vars)), dtype=torch.float
    )
    # Generate eigenvalues uniformly in [1, 30]
    eig_vals = torch.rand(num_vars) * 29 + 1
    
    # Construct positive definite matrix A
    A = eig_vecs @ torch.diag(eig_vals) @ eig_vecs.T
    b = torch.normal(0, 1 / np.sqrt(num_vars), size=(num_vars,))
    
    # Initial point
    x0 = torch.normal(0, 0.5 / np.sqrt(num_vars), size=(num_vars,))
    
    def quadratic(var):
        x = var.x
        return 0.5 * x.T @ A @ x + b.T @ x
    
    # Compute optimal solution
    optimal_x = scipy.linalg.solve(A.numpy(), -b.numpy(), assume_a="pos")
    optimal_val = quadratic(Variable(torch.tensor(optimal_x))).item()
    
    return {
        "model0": Variable(x0),
        "obj_function": quadratic,
        "optimal_x": optimal_x,
        "optimal_val": optimal_val,
        "A": A.numpy(),
        "b": b.numpy(),
    }


def rosenbrock():
    """
    Generate a Rosenbrock function optimization problem.
    
    Objective: f(x) = sum(100 * (x[i+1] - x[i]^2)^2 + (1 - x[i])^2)
    
    This is a classic non-convex optimization problem with a narrow valley.
    
    Returns:
        Dictionary containing:
        - model0: Initial model (Variable)
        - obj_function: Rosenbrock objective function
        - optimal_x: Global optimum (all ones)
        - optimal_val: Optimal value (0)
    """
    num_vars = 2
    
    # Standard initialization: alternating -2 and +2
    x0 = torch.tensor([-1.5 if i % 2 == 0 else 1.5 for i in range(num_vars)])
    
    def rosen(var):
        x = var.x
        return torch.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
    
    # Global optimum at all x_i = 1, giving f(x) = 0
    optimal_x = np.ones(num_vars)
    optimal_val = 0
    
    return {
        "model0": Variable(x0),
        "obj_function": rosen,
        "optimal_x": optimal_x,
        "optimal_val": optimal_val,
    }


def logistic_regression():
    """
    Generate a logistic regression problem with synthetic data.
    
    Creates two Gaussian distributions with different means and fits
    a logistic regression model with L2 regularization.
    
    Returns:
        Dictionary containing:
        - model0: Initial model (linear layer + sigmoid)
        - obj_function: Binary cross-entropy loss with regularization
        - data: Tuple of (inputs, labels)
    """
    num_vars = 3
    
    # Generate two Gaussian distributions for binary classification
    g0 = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=torch.randn(num_vars),
        scale_tril=torch.tril(torch.randn((num_vars, num_vars))),
    )
    g1 = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=torch.randn(num_vars),
        scale_tril=torch.tril(torch.randn((num_vars, num_vars))),
    )
    
    # Sample data points
    x = torch.cat([g0.sample((50,)), g1.sample((50,))])
    y = torch.cat([torch.zeros((50,)), torch.ones((50,))])
    
    # Shuffle dataset
    perm = torch.randperm(len(x))
    x = x[perm]
    y = y[perm]
    
    # Create model
    model0 = nn.Sequential(nn.Linear(num_vars, 1), nn.Sigmoid())
    
    def obj_function(model):
        y_hat = model(x).view(-1)
        weight_norm = model[0].weight.norm()
        return F.binary_cross_entropy(y_hat, y) + 5e-4 / 2 * weight_norm
    
    return {"model0": model0, "obj_function": obj_function, "data": (x, y)}


def robust_linear_regression():
    """
    Generate a robust linear regression problem using Geman-McClure loss.
    
    Creates data from multiple Gaussian distributions with different
    linear relationships, suitable for testing robust regression.
    
    Returns:
        Dictionary containing:
        - model0: Initial model (linear layer)
        - obj_function: Geman-McClure robust loss function
    """
    num_vars = 3
    
    # Create four Gaussian distributions with random parameters
    x = []
    y = []
    
    for _ in range(4):
        # Random Gaussian distribution for features
        gaussian = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.randn(num_vars),
            scale_tril=torch.tril(torch.randn((num_vars, num_vars))),
        )
        new_points = gaussian.sample((25,))
        
        # Generate labels: y = true_vector @ x + true_bias + noise
        true_vector = torch.randn(num_vars)
        true_bias = torch.randn(1)
        new_labels = new_points @ true_vector + true_bias + torch.randn(25)
        
        x.append(new_points)
        y.append(new_labels)
    
    x = torch.cat(x)
    y = torch.cat(y)
    
    # Shuffle dataset
    perm = torch.randperm(len(x))
    x = x[perm]
    y = y[perm]
    
    # Create model
    model0 = nn.Linear(num_vars, 1)
    
    def geman_mcclure(model):
        """Geman-McClure robust loss function."""
        y_hat = model(x).view(-1)
        squared_errors = (y - y_hat) ** 2
        return (squared_errors / (1 + squared_errors)).mean()
    
    return {"model0": model0, "obj_function": geman_mcclure}


def mlp():
    """
    Generate a multi-layer perceptron classification problem.
    
    Creates a synthetic binary classification problem with data from
    four Gaussian distributions, requiring a non-linear decision boundary.
    
    Returns:
        Dictionary containing:
        - model0: Initial model (2-layer MLP)
        - obj_function: Binary cross-entropy loss with regularization
        - dataset: Tuple of (inputs, labels)
    """
    num_vars = 2
    
    # Create four Gaussian distributions
    gaussians = [
        torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.randn(num_vars),
            scale_tril=torch.tril(torch.randn((num_vars, num_vars))),
        )
        for _ in range(4)
    ]
    
    # Randomly assign binary labels to each Gaussian
    # Ensure not all labels are the same
    gaussian_labels = np.zeros((4,))
    while (gaussian_labels == 0).all() or (gaussian_labels == 1).all():
        gaussian_labels = torch.randint(0, 2, size=(4,))
    
    # Generate dataset: 25 points from each Gaussian
    x = torch.cat([g.sample((25,)) for g in gaussians])
    y = torch.cat([torch.full((25,), float(label)) for label in gaussian_labels])
    
    # Shuffle dataset
    perm = torch.randperm(len(x))
    x = x[perm]
    y = y[perm]
    
    # Create 2-layer MLP
    model0 = nn.Sequential(
        nn.Linear(num_vars, 2), nn.ReLU(), nn.Linear(2, 1), nn.Sigmoid()
    )
    
    def obj_function(model):
        y_hat = model(x).view(-1)
        # Regularize both layers
        weight_norm = model[0].weight.norm() + model[2].weight.norm()
        return F.binary_cross_entropy(y_hat, y) + 5e-4 / 2 * weight_norm
    
    return {"model0": model0, "obj_function": obj_function, "dataset": (x, y)}

