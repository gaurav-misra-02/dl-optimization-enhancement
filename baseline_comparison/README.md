# Baseline Optimizer Comparison

This directory contains scripts for comparing various optimization algorithms on MNIST classification tasks. These provide baseline performance metrics for evaluating novel optimization approaches.

## Overview

The scripts train neural networks on MNIST using different optimizers to establish baseline performance. This allows for fair comparison when evaluating new optimization methods like the Alternate Adam or Autonomous Optimizer.

## Available Scripts

### Feedforward Network

Simple feedforward neural network (784 -> 64 -> 10):

```bash
python train_mnist_feedforward.py --optimizer adam --lr 1e-3 --epochs 15
```

### Convolutional Neural Network

CNN with two convolutional layers:

```bash
python train_mnist_cnn.py --optimizer adam --lr 1e-3 --epochs 15
```

## Supported Optimizers

- `adam` - Adam optimizer
- `sgd` - Stochastic Gradient Descent
- `adagrad` - AdaGrad (adaptive learning rates)
- `rmsprop` - RMSprop
- `adamw` - AdamW (Adam with weight decay)
- `sgd_momentum` - SGD with momentum (0.9)

## Command-Line Arguments

Common arguments for both scripts:

- `--optimizer` - Optimizer choice (default: adam)
- `--batch-size` - Training batch size (default: 32 for feedforward, 64 for CNN)
- `--epochs` - Maximum number of epochs (default: 15)
- `--lr` - Learning rate (default: 1e-3)
- `--early-stop-patience` - Early stopping patience (default: 3)
- `--seed` - Random seed for reproducibility (default: 42)
- `--data-dir` - MNIST data directory (default: ./data)
- `--save-model` - Path to save best model (optional)

## Example Usage

### Compare All Optimizers (Feedforward)

```bash
for opt in adam sgd adagrad rmsprop adamw sgd_momentum; do
    echo "Training with $opt"
    python train_mnist_feedforward.py --optimizer $opt --lr 1e-3
done
```

### Compare All Optimizers (CNN)

```bash
for opt in adam sgd adagrad rmsprop adamw sgd_momentum; do
    echo "Training with $opt"
    python train_mnist_cnn.py --optimizer $opt --lr 1e-3
done
```

### Hyperparameter Search

```bash
# Try different learning rates
for lr in 1e-4 1e-3 1e-2; do
    python train_mnist_cnn.py --optimizer adam --lr $lr
done
```

## Features

- Early stopping based on validation loss
- Train/validation/test split
- Progress tracking with per-epoch metrics
- Reproducible results with seed setting
- Optional model saving

## Expected Results

Typical test accuracies on MNIST:

**Feedforward Network:**
- Adam: ~97-98%
- SGD: ~95-96%
- SGD+Momentum: ~96-97%
- AdaGrad: ~96-97%
- RMSprop: ~97-98%
- AdamW: ~97-98%

**CNN:**
- Adam: ~98-99%
- SGD: ~97-98%
- SGD+Momentum: ~98-99%
- AdaGrad: ~97-98%
- RMSprop: ~98-99%
- AdamW: ~98-99%

Note: Results may vary based on random initialization and hyperparameters.

## Output Format

Each training run prints:
```
Device: cuda
Optimizer: adam
Learning rate: 0.001

==================================================
TRAINING
==================================================

Epoch 1/15
  Train: Loss=0.3245, Acc=90.52%
  Val:   Loss=0.1823, Acc=94.68%
  * Best model!

...

==================================================
Results:
  Best Val Loss: 0.0856
  Test Loss: 0.0891
  Test Accuracy: 97.34%
==================================================
```

## Integration with Other Experiments

Use these baselines to compare:

1. **Alternate Adam** - Compare against standard Adam baseline
2. **Multiple Initializations** - Compare final performance
3. **Autonomous Optimizer** - Compare learned vs. hand-designed optimizers

## Requirements

- PyTorch >= 1.7.0
- torchvision
- torchmetrics (for feedforward script)
- numpy

## Notes

- All experiments use early stopping with patience of 3 epochs
- Test set is split 50/50 from original MNIST test set for validation/testing
- Models are evaluated on the held-out test set using the best validation checkpoint
- Random seeds ensure reproducible comparisons

