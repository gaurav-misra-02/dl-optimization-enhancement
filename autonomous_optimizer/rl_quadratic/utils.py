"""
Utility functions for the RL quadratic optimizer.
"""

import numpy as np
import scipy.stats
import scipy.linalg
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


DIM = 3  # Problem dimensionality


def create_quadratic_problem(dim=DIM, device=None):
    """
    Create a random convex quadratic optimization problem.
    
    Generates f(x) = 0.5 * x^T @ Q @ x + c^T @ x where Q is positive definite
    with eigenvalues in [1, 30].
    
    Args:
        dim: Problem dimensionality
        device: torch device
    
    Returns:
        Dictionary containing Q, c, optimal_x, and optimal_val
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate random orthogonal eigenvectors
    eig_vecs = torch.tensor(
        scipy.stats.ortho_group.rvs(dim=dim), dtype=torch.float
    )
    
    # Generate eigenvalues uniformly in [1, 30]
    eig_vals = torch.rand(dim) * 29 + 1
    
    # Construct positive definite matrix Q
    Q = eig_vecs @ torch.diag(eig_vals) @ eig_vecs.T
    
    # Generate random linear term c
    c = torch.normal(0, 1 / np.sqrt(dim), size=(dim,))
    
    # Solve for optimal solution: Q @ x* = -c
    optimal_x = torch.tensor(scipy.linalg.solve(Q.numpy(), -c.numpy(), assume_a="pos"))
    optimal_val = 0.5 * optimal_x.T @ Q @ optimal_x + c.T @ optimal_x
    
    return {
        "Q": Q,
        "c": c,
        "optimal_x": optimal_x,
        "optimal_val": optimal_val
    }


def compute_rewards(rewards, gamma=1.0):
    """
    Compute discounted cumulative rewards.
    
    Args:
        rewards: List of rewards
        gamma: Discount factor
    
    Returns:
        Array of discounted cumulative rewards
    """
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards) - 1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])


def calculate_gaes(rewards, values, gamma=1.0, decay=0.97):
    """
    Calculate Generalized Advantage Estimates (GAE).
    
    Args:
        rewards: Array of rewards
        values: Array of value estimates
        gamma: Discount factor
        decay: GAE decay parameter (lambda)
    
    Returns:
        Array of advantage estimates
    """
    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val 
              for rew, val, next_val in zip(rewards, values, next_values)]
    
    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas) - 1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])
    
    return np.array(gaes[::-1])


def rollout(model, env, max_steps=100, eps=1e-6, device=None):
    """
    Perform a rollout with the current policy.
    
    Args:
        model: Actor network
        env: Environment
        max_steps: Maximum steps per episode
        eps: Small constant for numerical stability
        device: torch device
    
    Returns:
        train_data: Tuple of (observations, actions, rewards, GAEs, log_probs)
        ep_reward: Total episode reward
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    obs = env.reset()
    ep_reward = 0.0
    
    # Storage for episode data
    train_data = [[], [], [], [], []]  # obs, actions, rewards, values, log_probs
    
    for _ in range(max_steps):
        # Prepare observation vector
        obs_vector = torch.cat(
            (torch.tensor([obs['objective']], device=device), obs['current_value']),
            dim=0
        ).clamp(-100., 100.)
        
        # Get action from policy
        mean, variance = model.forward(
            torch.tensor(obs_vector.view(size=(1, -1)), device=device)
        )
        mean = mean.squeeze()
        variance = variance.squeeze()
        
        # Sample action
        gaussian_sampler = MultivariateNormal(
            loc=mean,
            covariance_matrix=torch.diag(variance) + eps * torch.eye(
                mean.size()[0],
                dtype=torch.float32,
                device=device
            )
        )
        
        action = gaussian_sampler.sample().clamp(-10., 10.)
        action_log_prob = gaussian_sampler.log_prob(action).item()
        
        # Value estimate (negative objective, clamped)
        value = -obs["objective"].clamp(-100, 100).item()
        
        # Take step in environment
        new_obs, reward, terminated, _ = env.step(action.cpu().numpy())
        
        # Store data
        train_data[0].append(obs_vector.cpu())
        train_data[1].append(action.cpu())
        train_data[2].append(reward.cpu() if torch.is_tensor(reward) else reward)
        train_data[3].append(value)
        train_data[4].append(action_log_prob)
        
        obs = new_obs
        ep_reward += reward.item() if torch.is_tensor(reward) else reward
        
        if terminated:
            break
    
    # Convert to numpy arrays
    train_data = [np.asarray(d) for d in train_data]
    
    # Compute GAEs
    train_data[3] = calculate_gaes(train_data[2], train_data[3])
    
    return train_data, ep_reward

