# Multiple Initializations Framework

This directory contains a training framework that leverages multiple random initializations to potentially find better local minima and improve final model performance.

## Overview

The Multiple Initializations approach starts training with several randomly initialized instances of the same model architecture. After a small number of training steps, models are evaluated on a validation set, and the poorest performers are eliminated. This process continues iteratively until only one model remains, which is then trained to completion.

## Key Concepts

### Progressive Elimination

1. Start with N models with different random initializations
2. Train all models for K steps
3. Evaluate on validation set
4. Keep top X% of models (e.g., 50%)
5. Repeat until one model remains
6. Complete training with the surviving model

### Benefits

- Explores multiple regions of the parameter space simultaneously
- Naturally selects initializations that lead to better local minima
- Can improve final model performance compared to single initialization
- Particularly useful for non-convex optimization landscapes

### Trade-offs

- Requires more computational resources initially
- Most beneficial when good initialization is critical
- Reduction factor and elimination frequency are hyperparameters to tune

## Files

- `framework.py` - Core implementation of the Multiple Initializations trainer
- `train_mnist.py` - Example usage on MNIST classification

## Usage

### Basic Example

```python
from framework import MultipleInitializationsTrainer
import torch.nn as nn

# Define model factory
def model_fn():
    return YourModel()

# Define optimizer factory
def optimizer_fn(params):
    return torch.optim.Adam(params, lr=0.001)

# Create trainer
trainer = MultipleInitializationsTrainer(
    model_fn=model_fn,
    optimizer_fn=optimizer_fn,
    num_models=10,           # Start with 10 models
    num_elim_steps=100,      # Train for 100 steps before elimination
    reduce_factor=0.5        # Keep top 50% after each round
)

# Train
best_model = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.CrossEntropyLoss(),
    num_epochs=20
)
```

### MNIST Example

Train on MNIST with default settings:

```bash
python train_mnist.py
```

Custom configuration:

```bash
python train_mnist.py \
    --num-models 20 \
    --num-elim-steps 200 \
    --reduce-factor 0.5 \
    --epochs 15 \
    --lr 0.001
```

### Command-line Arguments

- `--num-models` - Initial number of models to train (default: 10)
- `--num-elim-steps` - Training steps before elimination round (default: 100)
- `--reduce-factor` - Fraction of models to keep each round (default: 0.5)
- `--batch-size` - Batch size for training (default: 64)
- `--epochs` - Total number of epochs (default: 10)
- `--lr` - Learning rate (default: 0.001)
- `--seed` - Random seed (default: 42)
- `--data-dir` - Directory for dataset (default: ./data)

## Hyperparameter Guidelines

### Number of Models

- **Small (5-10)**: Quick experiments, modest improvements
- **Medium (10-20)**: Good balance of exploration and computation
- **Large (20+)**: Maximum exploration, significant computational cost

### Elimination Steps

- **Small (50-100)**: Quick elimination, less training per round
- **Medium (100-500)**: Balanced approach
- **Large (500+)**: More training before elimination, slower convergence to single model

### Reduction Factor

- **Aggressive (0.3-0.5)**: Rapid elimination, reaches single model quickly
- **Conservative (0.5-0.8)**: Slower elimination, more exploration time
- **Very Conservative (0.8-0.9)**: Very gradual elimination

## Algorithm Details

### Elimination Strategy

Models are ranked by validation loss (ascending order), and only the top fraction is retained:

```python
num_to_keep = max(1, int(num_models * reduce_factor))
```

The `max(1, ...)` ensures at least one model always survives.

### Training Phases

1. **Multi-model phase**: Multiple models train for `num_elim_steps`, then elimination occurs
2. **Single-model phase**: Once one model remains, standard training continues until completion

### Memory Management

To manage GPU memory, models are moved to CPU when not actively training if multiple models exist.

## Requirements

- PyTorch >= 1.7.0
- torchvision (for MNIST example)
- numpy

## Example Output

```
Initialized 10 models for training

Training 10 model(s) for 100 steps
  Model 1: Train Loss = 0.3245, Val Loss = 0.2891
  Model 2: Train Loss = 0.3567, Val Loss = 0.3124
  ...
Eliminated poor performers. Remaining models: 5

Training 5 model(s) for 100 steps
  ...
Eliminated poor performers. Remaining models: 2

Training 2 model(s) for 100 steps
  ...
Eliminated poor performers. Remaining models: 1

Training 1 model(s) for 7300 steps
  Model 1: Train Loss = 0.0234, Val Loss = 0.0456

Training complete! Best validation loss: 0.0456
```

## Notes

- The framework is model-agnostic and can be used with any PyTorch model
- Validation set should be separate from test set for unbiased evaluation
- Consider computational resources when setting the number of initial models
- GPU memory usage scales with the number of concurrent models

