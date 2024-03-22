"""
Autonomous Optimizer - A learned optimization algorithm using reinforcement learning.

This optimizer uses a trained RL policy to determine parameter updates based on
the history of objective values and gradients.
"""

import numpy as np
import torch
from torch import optim


class AutonomousOptimizer(optim.Optimizer):
    """
    An optimizer that uses a trained RL policy to determine parameter updates.
    
    Instead of using a fixed update rule like Adam or SGD, this optimizer
    leverages a learned policy that takes as input the recent history of
    objective values and gradients, and outputs parameter updates.
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        policy: Trained RL policy (e.g., from stable-baselines3) that takes
            observation of shape (history_len, num_parameters + 1) and returns
            actions of shape (num_parameters,)
        history_len: Number of previous iterations to keep in history (default: 25)
    
    Example:
        >>> from stable_baselines3 import PPO
        >>> policy = PPO.load("trained_policy")
        >>> optimizer = AutonomousOptimizer(model.parameters(), policy=policy)
        >>> 
        >>> def closure():
        >>>     optimizer.zero_grad()
        >>>     loss = criterion(model(x), y)
        >>>     loss.backward()
        >>>     return loss
        >>> 
        >>> optimizer.step(closure)
    """
    
    def __init__(self, params, policy, history_len=25):
        super().__init__(params, {})
        
        self.policy = policy
        self.history_len = history_len
        
        # Calculate total number of parameters
        self.num_params = sum(
            p.numel() for group in self.param_groups for p in group["params"]
        )
        
        # History of objective values and gradients
        self.obj_values = []
        self.gradients = []
    
    def _make_observation(self, obj_value):
        """
        Create observation vector for the policy.
        
        The observation is a matrix where each row represents one time step in
        the history. Each row contains:
        - Difference between current objective and historical objective
        - Historical gradient (flattened)
        
        Args:
            obj_value: Current objective value
        
        Returns:
            Observation array of shape (history_len, 1 + num_params)
        """
        observation = np.zeros((self.history_len, 1 + self.num_params), dtype="float32")
        
        # Fill in objective value differences
        observation[: len(self.obj_values), 0] = (
            obj_value - torch.tensor(self.obj_values).detach().numpy()
        )
        
        # Fill in gradient history
        for i, grad in enumerate(self.gradients):
            observation[i, 1:] = grad.detach().numpy()
        
        # Normalize and clip to [-1, 1] for stable RL policy input
        observation /= 50
        return observation.clip(-1, 1)
    
    @torch.no_grad()
    def step(self, closure):
        """
        Perform a single optimization step using the learned policy.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
                This closure should call zero_grad(), compute the loss, and
                call backward() on it.
        
        Returns:
            The objective value returned by the closure
        """
        # Evaluate the objective function to get current value and gradients
        with torch.enable_grad():
            obj_value = closure()
        
        # Collect and flatten current gradient
        current_grad = torch.cat(
            [p.grad.flatten() for group in self.param_groups for p in group["params"]]
        ).flatten()
        
        # Update history with current objective value and gradient
        if len(self.obj_values) >= self.history_len:
            self.obj_values.pop(-1)
            self.gradients.pop(-1)
        self.obj_values.insert(0, obj_value)
        self.gradients.insert(0, current_grad)
        
        # Get action from policy based on current observation
        observation = self._make_observation(obj_value.item())
        action, _states = self.policy.predict(observation, deterministic=True)
        
        # Apply the policy's action as parameter updates
        action = torch.from_numpy(action)
        param_counter = 0
        
        for group in self.param_groups:
            for p in group["params"]:
                # Extract the update for this parameter
                delta_p = action[param_counter : param_counter + p.numel()]
                # Apply update (action directly specifies the change)
                p.add_(delta_p.reshape(p.shape))
                param_counter += p.numel()
        
        return obj_value

