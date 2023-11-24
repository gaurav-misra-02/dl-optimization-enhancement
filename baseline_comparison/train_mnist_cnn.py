"""
MNIST classification with CNN for optimizer comparison.

This script trains a convolutional neural network on MNIST using
various optimizers for baseline comparison.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class CNN(nn.Module):
    """Convolutional Neural Network for MNIST."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def get_optimizer(model, optimizer_name, lr):
    """Get optimizer by name."""
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
    
    for data, target in train_loader:
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
    
    return total_loss / len(train_loader), 100. * correct / total


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
    
    return val_loss / len(val_loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='Train MNIST CNN with various optimizers')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'adagrad', 'rmsprop', 'adamw', 'sgd_momentum'],
                        help='Optimizer (default: adam)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=15, help='Epochs (default: 15)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--early-stop-patience', type=int, default=3,
                        help='Early stopping patience (default: 3)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Data directory (default: ./data)')
    parser.add_argument('--save-model', type=str, default=None,
                        help='Path to save best model')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'Device: {device}')
    print(f'Optimizer: {args.optimizer}')
    print(f'Learning rate: {args.lr}\n')
    
    # Data
    train_dataset = MNIST(args.data_dir, train=True, download=True, transform=ToTensor())
    test_dataset = MNIST(args.data_dir, train=False, download=True, transform=ToTensor())
    val_dataset, test_dataset = random_split(
        test_dataset, [0.5, 0.5], torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model, criterion, optimizer
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args.optimizer, args.lr)
    
    # Training with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print("="*70)
    print("TRAINING")
    print("="*70 + "\n")
    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f'  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%')
        print(f'  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print('  * Best model!')
        else:
            patience_counter += 1
            print(f'  Patience: {patience_counter}/{args.early_stop_patience}')
        
        if patience_counter >= args.early_stop_patience:
            print(f'\nEarly stopping at epoch {epoch + 1}')
            break
        print()
    
    # Test evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    print("="*70)
    print(f'Results:')
    print(f'  Best Val Loss: {best_val_loss:.4f}')
    print(f'  Test Loss: {test_loss:.4f}')
    print(f'  Test Accuracy: {test_acc:.2f}%')
    print("="*70)
    
    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f'\nModel saved to {args.save_model}')


if __name__ == '__main__':
    main()

