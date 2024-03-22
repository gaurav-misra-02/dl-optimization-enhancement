"""
Gym Environment for training the autonomous optimizer policy.

This environment treats optimization as a reinforcement learning problem,
where the agent learns to output parameter updates that minimize objective functions.
"""

import copy
import numpy as np
import torch
import gym
from gym import spaces


def make_observation(obj_value, obj_values, gradients, num_params, history_len):
    """
    Create an observation from optimization history.
    
    Args:
        obj_value: Current objective value
        obj_values: List of historical objective values
        gradients: List of historical gradients (flattened)
        num_params: Total number of parameters
        history_len: Length of history to maintain
    
    Returns:
        Observation array of shape (history_len, num_params + 1)
    """
    observation = np.zeros((history_len, 1 + num_params), dtype="float32")
    
    # Fill objective value differences
    observation[: len(obj_values), 0] = (
        obj_value - torch.tensor(obj_values).detach().numpy()
    )
    
    # Fill gradient history
    for i, grad in enumerate(gradients):
        observation[i, 1:] = grad.detach().numpy()
    
    # Normalize and clip observation space to [-1, 1]
    observation /= 50
    return observation.clip(-1, 1)


class OptimizationEnvironment(gym.Env):
    """
    Gym environment for learning to optimize.
    
    The environment samples optimization problems from a dataset and presents
    them to an RL agent. The agent's action is a parameter update, and the
    reward is based on the reduction in objective value.
    
    Args:
        dataset: List of problem dictionaries, each containing:
            - 'model0': Initial model
            - 'obj_function': Objective function that takes a model and returns loss
        num_steps: Maximum number of optimization steps per episode
        history_len: Number of previous steps to include in observation
    
    Observation Space:
        Box of shape (history_len, num_params + 1) containing:
        - Column 0: Objective value differences
        - Columns 1+: Historical gradients
    
    Action Space:
        Box of shape (num_params,) containing parameter updates
    
    Reward:
        Negative objective value (higher is better)
    """
    
    def __init__(self, dataset, num_steps, history_len):
        super().__init__()
        
        self.dataset = dataset
        self.num_steps = num_steps
        self.history_len = history_len
        
        # Initialize with a problem to determine dimensions
        self._setup_episode()
        self.num_params = sum(p.numel() for p in self.model.parameters())
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.num_params,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.history_len, 1 + self.num_params),
            dtype=np.float32,
        )
    
    def _setup_episode(self):
        """Initialize a new optimization episode with a random problem."""
        problem = np.random.choice(self.dataset)
        self.model = copy.deepcopy(problem["model0"])
        self.obj_function = problem["obj_function"]
        
        self.obj_values = []
        self.gradients = []
        self.current_step = 0
    
    def reset(self):
        """
        Reset the environment to start a new episode.
        
        Returns:
            Initial observation (empty history)
        """
        self._setup_episode()
        return make_observation(
            None, self.obj_values, self.gradients, self.num_params, self.history_len
        )
    
    @torch.no_grad()
    def step(self, action):
        """
        Take an optimization step with the given action.
        
        Args:
            action: Parameter updates to apply
        
        Returns:
            observation: New observation after applying action
            reward: Negative objective value
            done: Whether episode is complete
            info: Additional information (empty dict)
        """
        # Apply the action (parameter updates)
        action = torch.from_numpy(action)
        param_counter = 0
        
        for p in self.model.parameters():
            delta_p = action[param_counter : param_counter + p.numel()]
            p.add_(delta_p.reshape(p.shape))
            param_counter += p.numel()
        
        # Calculate new objective value and gradient
        with torch.enable_grad():
            self.model.zero_grad()
            obj_value = self.obj_function(self.model)
            obj_value.backward()
        
        # Collect current gradient
        current_grad = torch.cat(
            [p.grad.flatten() for p in self.model.parameters()]
        ).flatten()
        
        # Update history
        if len(self.obj_values) >= self.history_len:
            self.obj_values.pop(-1)
            self.gradients.pop(-1)
        self.obj_values.insert(0, obj_value)
        self.gradients.insert(0, current_grad)
        
        # Create observation
        observation = make_observation(
            obj_value.item(),
            self.obj_values,
            self.gradients,
            self.num_params,
            self.history_len,
        )
        
        # Reward is negative objective (we want to minimize objective)
        reward = -obj_value.item()
        
        # Check if episode is done
        self.current_step += 1
        done = self.current_step >= self.num_steps
        
        info = {}
        
        return observation, reward, done, info

