"""
MNIST classification with feedforward network for optimizer comparison.

This script trains a simple feedforward neural network on MNIST using
various optimizers (Adam, SGD, AdaGrad, RMSprop, AdamW, SGD+Momentum).
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchmetrics import Accuracy


class FeedForwardNet(nn.Module):
    """Simple feedforward neural network for MNIST."""
    
    def __init__(self, input_size=784, hidden_size=64, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.model(x)


def get_optimizer(model, optimizer_name, lr):
    """
    Get optimizer by name.
    
    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer
        lr: Learning rate
    
    Returns:
        Optimizer instance
    """
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'adagrad':
        return optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd_momentum':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    val_loss /= len(val_loader)
    accuracy = 100. * correct / total
    return val_loss, accuracy


def main():
    parser = argparse.ArgumentParser(
        description='Train MNIST with various optimizers'
    )
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'adagrad', 'rmsprop', 'adamw', 'sgd_momentum'],
                        help='Optimizer to use (default: adam)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of epochs (default: 15)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--early-stop-patience', type=int, default=3,
                        help='Early stopping patience (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Data directory (default: ./data)')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Optimizer: {args.optimizer}')
    print(f'Learning rate: {args.lr}\n')
    
    # Load data
    train_dataset = MNIST(
        root=args.data_dir, train=True, download=True, transform=ToTensor()
    )
    test_dataset = MNIST(
        root=args.data_dir, train=False, download=True, transform=ToTensor()
    )
    
    # Split test into val and test
    val_dataset, test_dataset = random_split(
        test_dataset, [0.5, 0.5], torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = FeedForwardNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args.optimizer, args.lr)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print("="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print('  * New best model!')
        else:
            patience_counter += 1
            print(f'  Patience: {patience_counter}/{args.early_stop_patience}')
        
        if patience_counter >= args.early_stop_patience:
            print(f'\n  Early stopping at epoch {epoch + 1}')
            break
        
        print()
    
    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70 + "\n")
    
    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    print(f'Final Results:')
    print(f'  Best Val Loss: {best_val_loss:.4f}')
    print(f'  Test Loss: {test_loss:.4f}')
    print(f'  Test Accuracy: {test_acc:.2f}%')


if __name__ == '__main__':
    main()

