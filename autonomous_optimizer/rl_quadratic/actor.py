"""
Actor network for the RL quadratic optimizer.

The actor network outputs both mean and variance for a Gaussian policy
that determines parameter updates.
"""

import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    """
    Actor network for quadratic optimization.
    
    Takes observation (objective + parameters) and outputs mean and variance
    for a Gaussian distribution over parameter updates.
    
    Args:
        obs_state_space: Dimension of observation space
        action_state_space: Dimension of action space
        hidden_dim: Hidden layer dimension (default: 64)
    """
    
    def __init__(self, obs_state_space, action_state_space, hidden_dim=64):
        super(ActorNetwork, self).__init__()
        
        self.action_state_space = action_state_space
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(obs_state_space, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean head
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_state_space)
        )
        
        # Variance head (outputs log std for numerical stability)
        self.log_std_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_state_space)
        )
    
    def forward(self, obs):
        """
        Forward pass through the network.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_state_space)
        
        Returns:
            mean: Mean of action distribution (batch_size, action_state_space)
            variance: Variance of action distribution (batch_size, action_state_space)
        """
        # Shared features
        features = self.shared(obs)
        
        # Mean
        mean = self.mean_head(features)
        
        # Variance (convert log_std to variance)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Stability
        variance = torch.exp(2 * log_std)
        
        return mean, variance


if __name__ == "__main__":
    # Test the network
    obs_dim = 4  # e.g., 3 parameters + 1 objective
    action_dim = 3
    
    actor = ActorNetwork(obs_dim, action_dim, hidden_dim=64)
    
    # Random observation
    obs = torch.randn(32, obs_dim)  # Batch of 32
    
    mean, variance = actor(obs)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Mean shape: {mean.shape}")
    print(f"Variance shape: {variance.shape}")
    print(f"Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"Variance range: [{variance.min():.3f}, {variance.max():.3f}]")

