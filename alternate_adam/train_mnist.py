"""
Training script for comparing Alternate Adam with standard Adam on MNIST.

This script trains a simple neural network on the MNIST dataset using both
standard Adam and Alternate Adam optimizers for comparison.
"""

import argparse
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from optimizer import AlternateAdam


class SimpleNN(nn.Module):
    """Simple feedforward neural network for MNIST classification."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_epoch(model, train_loader, optimizer, criterion, device, log_file=None, log_interval=100):
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
        
        if batch_idx % log_interval == 0:
            print(f'  Batch {batch_idx}/{len(train_loader)} | '
                  f'Loss: {loss.item():.6f} | '
                  f'Acc: {100. * correct / total:.2f}%')
            
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f'{loss.item()}\n')
    
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
    parser = argparse.ArgumentParser(description='Train MNIST with Alternate Adam')
    parser.add_argument('--optimizer', type=str, default='alternate_adam',
                        choices=['adam', 'alternate_adam'],
                        help='Optimizer to use (default: alternate_adam)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory for dataset (default: ./data)')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Directory for logs (default: ./logs)')
    parser.add_argument('--save-model', action='store_true',
                        help='Save the trained model')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(args.data_dir, train=True, download=True, 
                                    transform=transform)
    val_dataset = datasets.MNIST(args.data_dir, train=False, download=True,
                                  transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    # Model
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print('Using standard Adam optimizer')
    else:
        optimizer = AlternateAdam(model.parameters(), lr=args.lr)
        print('Using Alternate Adam optimizer')
    
    # Log files
    train_log = os.path.join(args.log_dir, f'train_loss_{args.optimizer}.txt')
    val_loss_log = os.path.join(args.log_dir, f'val_loss_{args.optimizer}.txt')
    val_acc_log = os.path.join(args.log_dir, f'val_acc_{args.optimizer}.txt')
    
    # Clear previous logs
    for log_file in [train_log, val_loss_log, val_acc_log]:
        if os.path.exists(log_file):
            os.remove(log_file)
    
    # Training loop
    print(f'\nTraining for {args.epochs} epochs...\n')
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            log_file=train_log, log_interval=100
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Log validation metrics
        with open(val_loss_log, 'a') as f:
            f.write(f'{val_loss}\n')
        with open(val_acc_log, 'a') as f:
            f.write(f'{val_acc}\n')
        
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%\n')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            if args.save_model:
                model_path = os.path.join(args.log_dir, f'best_model_{args.optimizer}.pth')
                torch.save(model.state_dict(), model_path)
                print(f'  Saved best model with accuracy: {best_acc:.2f}%\n')
    
    print(f'Training complete! Best validation accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()

