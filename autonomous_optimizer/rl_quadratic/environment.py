"""
Gym environment for quadratic optimization problems.

The agent learns to minimize f(x) = 0.5 * x^T @ Q @ x + c^T @ x
where Q is a positive definite matrix.
"""

import torch
import numpy as np
import gymnasium as gym
from gym import spaces


# Problem dimensionality
DIM = 3


class QuadraticEnv(gym.Env):
    """
    Environment for learning to optimize convex quadratic functions.
    
    The objective is: f(x) = 0.5 * x^T @ Q @ x + c^T @ x
    
    Args:
        Q_mat: Positive definite matrix (DIM x DIM)
        c_mat: Linear term vector (DIM,)
        device: torch device for computation
    
    State:
        Current parameter values (DIM,) concatenated with objective value
    
    Action:
        Parameter update to apply (DIM,)
    
    Reward:
        Reduction in objective value (old_obj - new_obj)
    
    Termination:
        Episode terminates when objective changes are below threshold for
        multiple consecutive steps.
    """
    
    def __init__(self, Q_mat, c_mat, device=None):
        super().__init__()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.Q = Q_mat.to(device=self.device)
        self.c = c_mat.to(device=self.device)
        
        # Termination criteria
        self.threshold = 0.1
        self.max_consecutive_below_threshold = 3
        self._consecutive_below_threshold = 0
        self._old_objective = None
        
        # Gym spaces
        self.action_space = spaces.Box(
            low=-10., high=10., shape=(DIM,), dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(
                low=-100., high=100., shape=(DIM + 1,), dtype=np.float32
            )
        })
    
    def _compute_objective(self, x):
        """Compute quadratic objective f(x) = 0.5 * x^T @ Q @ x + c^T @ x"""
        return 0.5 * x.T @ self.Q @ x + self.c.T @ x
    
    def _get_obs(self):
        """Get current observation (objective value + current position)."""
        obj = self._compute_objective(self._agent_location)
        return {
            "objective": obj,
            "current_value": self._agent_location
        }
    
    def _check_termination(self, obj):
        """
        Check if episode should terminate based on objective changes.
        
        Terminates if objective changes are below threshold for
        max_consecutive_below_threshold steps.
        """
        if abs(self._old_objective - obj) <= self.threshold:
            self._consecutive_below_threshold += 1
        else:
            self._consecutive_below_threshold = 0
        
        self._old_objective = obj
        
        return self._consecutive_below_threshold >= self.max_consecutive_below_threshold
    
    def reset(self, seed=None):
        """
        Reset environment with random initialization.
        
        Returns:
            Initial observation
        """
        super().reset(seed=seed)
        
        # Random initialization from standard normal
        mean = np.zeros(DIM)
        cov_mat = np.eye(DIM)
        self._agent_location = torch.tensor(
            self.np_random.multivariate_normal(mean=mean, cov=cov_mat),
            dtype=torch.float32,
            device=self.device
        )
        
        # Initialize termination tracking
        obs = self._get_obs()
        self._old_objective = obs["objective"]
        self._consecutive_below_threshold = 0
        
        return obs
    
    def step(self, action):
        """
        Apply action (parameter update) and return new observation.
        
        Args:
            action: Parameter update to apply (DIM,)
        
        Returns:
            observation: New observation after update
            reward: Reduction in objective (old_obj - new_obj)
            terminated: Whether episode is done
            info: Additional info (empty dict)
        """
        # Convert action to torch tensor
        action = torch.tensor(action, dtype=torch.float32, device=self.device)
        
        # Apply parameter update
        self._agent_location = self._agent_location + action
        
        # Get new observation
        obs = self._get_obs()
        
        # Reward is the improvement in objective
        reward = self._old_objective - obs["objective"]
        
        # Check termination
        terminated = self._check_termination(obs["objective"])
        
        info = {}
        
        return obs, reward, terminated, info

