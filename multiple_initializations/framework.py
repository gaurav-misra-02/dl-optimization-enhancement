"""
Multiple Initializations Framework

This module implements a training framework that starts with multiple randomly
initialized models and progressively eliminates poorly performing ones based on
validation loss. This approach can help find better local minima and improve
final model performance.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Callable, Optional
import copy


class MultipleInitializationsTrainer:
    """
    Trainer that manages multiple model initializations and progressively
    eliminates poor performers.
    
    The framework trains multiple models with different initializations for
    a small number of steps, evaluates them, and keeps only the top performers.
    This process continues until only one model remains, which is then trained
    to completion.
    
    Args:
        model_fn: Function that returns a new model instance
        optimizer_fn: Function that takes model parameters and returns an optimizer
        num_models: Initial number of models to train (default: 10)
        num_elim_steps: Number of training steps before elimination (default: 100)
        reduce_factor: Fraction of models to keep after each elimination (default: 0.5)
        device: Device to train on (default: 'cuda' if available else 'cpu')
    """
    
    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        optimizer_fn: Callable[[any], torch.optim.Optimizer],
        num_models: int = 10,
        num_elim_steps: int = 100,
        reduce_factor: float = 0.5,
        device: Optional[str] = None
    ):
        self.model_fn = model_fn
        self.optimizer_fn = optimizer_fn
        self.num_elim_steps = num_elim_steps
        self.reduce_factor = reduce_factor
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize multiple model instances
        self.instances = []
        for _ in range(num_models):
            model = model_fn()
            optimizer = optimizer_fn(model.parameters())
            self.instances.append({
                'model': model,
                'optimizer': optimizer,
                'val_loss': float('inf')
            })
        
        print(f"Initialized {len(self.instances)} models for training")
    
    def train_step(
        self,
        instance: dict,
        data_batch: Tuple[torch.Tensor, torch.Tensor],
        criterion: nn.Module
    ) -> float:
        """
        Perform a single training step for one model instance.
        
        Args:
            instance: Dictionary containing model and optimizer
            data_batch: Tuple of (inputs, targets)
            criterion: Loss function
        
        Returns:
            Loss value for this step
        """
        model = instance['model']
        optimizer = instance['optimizer']
        
        model.train()
        inputs, targets = data_batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def validate(
        self,
        instance: dict,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> float:
        """
        Evaluate a model instance on validation data.
        
        Args:
            instance: Dictionary containing model
            val_loader: Validation data loader
            criterion: Loss function
        
        Returns:
            Average validation loss
        """
        model = instance['model']
        model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def eliminate_poor_performers(self):
        """
        Sort instances by validation loss and keep only the top performers.
        """
        # Sort by validation loss (ascending)
        self.instances.sort(key=lambda x: x['val_loss'])
        
        # Keep only the top fraction
        num_to_keep = max(1, int(len(self.instances) * self.reduce_factor))
        self.instances = self.instances[:num_to_keep]
        
        print(f"Eliminated poor performers. Remaining models: {len(self.instances)}")
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        num_epochs: int,
        verbose: bool = True
    ) -> nn.Module:
        """
        Main training loop with progressive elimination.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            num_epochs: Total number of epochs to train
            verbose: Whether to print progress
        
        Returns:
            The best trained model
        """
        total_steps = 0
        target_steps = num_epochs * len(train_loader)
        
        while total_steps < target_steps:
            # Determine steps for this round
            if len(self.instances) > 1:
                steps_this_round = min(self.num_elim_steps, target_steps - total_steps)
            else:
                steps_this_round = target_steps - total_steps
            
            if verbose:
                print(f"\nTraining {len(self.instances)} model(s) for {steps_this_round} steps")
            
            # Train each model
            for idx, instance in enumerate(self.instances):
                instance['model'].to(self.device)
                
                train_loss = 0.0
                for step in range(steps_this_round):
                    # Get next batch
                    try:
                        data_batch = next(train_iter)
                    except:
                        train_iter = iter(train_loader)
                        data_batch = next(train_iter)
                    
                    loss = self.train_step(instance, data_batch, criterion)
                    train_loss += loss
                
                # Validate
                val_loss = self.validate(instance, val_loader, criterion)
                instance['val_loss'] = val_loss
                
                if verbose:
                    avg_train_loss = train_loss / steps_this_round
                    print(f"  Model {idx + 1}: Train Loss = {avg_train_loss:.4f}, "
                          f"Val Loss = {val_loss:.4f}")
                
                # Move model back to CPU if multiple models to save GPU memory
                if len(self.instances) > 1:
                    instance['model'].to('cpu')
            
            total_steps += steps_this_round
            
            # Eliminate poor performers if multiple models remain
            if len(self.instances) > 1:
                self.eliminate_poor_performers()
        
        # Return the best model
        best_instance = self.instances[0]
        best_model = best_instance['model'].to(self.device)
        
        if verbose:
            print(f"\nTraining complete! Best validation loss: {best_instance['val_loss']:.4f}")
        
        return best_model
    
    def get_best_model(self) -> nn.Module:
        """
        Get the current best model based on validation loss.
        
        Returns:
            Best model instance
        """
        best_instance = min(self.instances, key=lambda x: x['val_loss'])
        return best_instance['model']

