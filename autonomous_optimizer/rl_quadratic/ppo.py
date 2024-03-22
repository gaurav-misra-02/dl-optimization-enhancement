"""
Proximal Policy Optimization (PPO) trainer for the quadratic optimizer.
"""

import torch
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal


def create_batched_covariance_matrix(variance):
    """
    Create diagonal covariance matrices from variance vectors.
    
    Args:
        variance: Tensor of shape (batch_size, dim) containing variances
    
    Returns:
        Tensor of shape (batch_size, dim, dim) containing diagonal covariance matrices
    """
    matrices = []
    for v in variance:
        matrices.append(torch.diag(v))
    return torch.stack(matrices, dim=0)


class PPOTrainer:
    """
    PPO trainer for optimizing the actor policy.
    
    Args:
        actor: Actor network
        policy_lr: Learning rate for policy updates (default: 1e-3)
        clip_factor: PPO clipping parameter (default: 0.2)
        policy_kldiv_bound: KL divergence bound for early stopping (default: 0.01)
        max_policy_train_steps: Maximum policy training steps per update (default: 50)
        device: torch device
    """
    
    def __init__(
        self,
        actor,
        policy_lr=1e-3,
        clip_factor=0.2,
        policy_kldiv_bound=0.01,
        max_policy_train_steps=50,
        device=None
    ):
        self.actor = actor
        self.lr = policy_lr
        self.clip = clip_factor
        self.kldiv_bound = policy_kldiv_bound
        self.max_policy_train_steps = max_policy_train_steps
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
    
    def train_policy(self, obs, acts, gaes, old_log_probs, eps=1e-6):
        """
        Perform PPO policy update.
        
        Args:
            obs: Observations (batch_size, obs_dim)
            acts: Actions taken (batch_size, action_dim)
            gaes: Generalized Advantage Estimates (batch_size,)
            old_log_probs: Log probabilities of actions under old policy (batch_size,)
            eps: Small constant for numerical stability
        """
        for step in range(self.max_policy_train_steps):
            self.optimizer.zero_grad()
            
            # Get current policy distribution
            mean, variance = self.actor.forward(obs)
            mean = mean.squeeze()
            variance = variance.squeeze()
            
            # Create covariance matrices
            variance_matrix = create_batched_covariance_matrix(variance)
            
            # Add small constant for numerical stability
            variance_matrix = variance_matrix + eps * torch.eye(
                mean.size()[1],
                dtype=torch.float32,
                device=self.device
            )
            
            # Create Gaussian distribution
            gaussian_sampler = MultivariateNormal(
                loc=mean,
                covariance_matrix=variance_matrix
            )
            
            # Compute new log probabilities
            new_log_probs = gaussian_sampler.log_prob(acts)
            
            # Compute probability ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute clipped surrogate objective
            clipped_ratio = ratio.clamp(1 - self.clip, 1 + self.clip)
            policy_loss = -torch.min(ratio * gaes, clipped_ratio * gaes).mean()
            
            # Backpropagation
            policy_loss.backward()
            self.optimizer.step()
            
            # Early stopping based on KL divergence
            kl_div = (old_log_probs - new_log_probs).mean()
            if kl_div >= self.kldiv_bound:
                break
        
        return step + 1  # Return number of steps taken

