# Alternate Adam Optimizer

This directory contains an implementation of a modified Adam optimizer with an adjusted variance term computation.

## Overview

The Alternate Adam optimizer modifies the standard Adam algorithm by adjusting how the second moment estimate (variance term) is computed. The modification applies a scaling factor of 1.1 to the variance update, potentially leading to different convergence behavior.

### Key Modification

**Standard Adam:**
```python
exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
```

**Alternate Adam:**
```python
exp_avg_sq.mul_(beta2).add_(grad.pow(2), alpha=(1 - beta2) * 1.1)
```

## Files

- `optimizer.py` - Implementation of the Alternate Adam optimizer
- `train_mnist.py` - Training script for MNIST comparison experiments
- `visualize_results.py` - Script to visualize and compare results

## Usage

### Training

Train a model using Alternate Adam:

```bash
python train_mnist.py --optimizer alternate_adam --epochs 10 --lr 0.001
```

Train a model using standard Adam for comparison:

```bash
python train_mnist.py --optimizer adam --epochs 10 --lr 0.001
```

### Command-line Arguments

- `--optimizer` - Choose between 'adam' or 'alternate_adam' (default: alternate_adam)
- `--batch-size` - Input batch size for training (default: 64)
- `--epochs` - Number of epochs to train (default: 10)
- `--lr` - Learning rate (default: 0.001)
- `--seed` - Random seed (default: 42)
- `--data-dir` - Directory for dataset (default: ./data)
- `--log-dir` - Directory for logs (default: ./logs)
- `--save-model` - Flag to save the trained model

### Visualization

After training both optimizers, visualize the comparison:

```bash
python visualize_results.py --log-dir ./logs --output-dir ./results
```

This will generate three comparison plots:
- Training loss comparison
- Validation loss comparison
- Validation accuracy comparison

## Results

Training logs are saved in the `logs/` directory:
- `train_loss_{optimizer}.txt` - Training loss at each batch
- `val_loss_{optimizer}.txt` - Validation loss per epoch
- `val_acc_{optimizer}.txt` - Validation accuracy per epoch

Visualization outputs are saved in the `results/` directory as PNG files.

## Example Workflow

```bash
# Train with standard Adam
python train_mnist.py --optimizer adam --epochs 10 --save-model

# Train with Alternate Adam
python train_mnist.py --optimizer alternate_adam --epochs 10 --save-model

# Generate comparison plots
python visualize_results.py

# View summary statistics
cat logs/val_acc_*.txt
```

## Requirements

- PyTorch >= 1.7.0
- torchvision
- matplotlib
- numpy

## Notes

The variance modification factor (1.1) can be adjusted in the `optimizer.py` file. Different values may yield different convergence characteristics depending on the problem domain.

