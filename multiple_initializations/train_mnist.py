"""
Example training script using the Multiple Initializations framework on MNIST.

This demonstrates how to use the framework with a simple CNN on MNIST classification.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from framework import MultipleInitializationsTrainer


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification."""
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
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


def evaluate_accuracy(model, data_loader, device):
    """Calculate accuracy on a dataset."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100. * correct / total


def main():
    parser = argparse.ArgumentParser(
        description='Train MNIST with Multiple Initializations'
    )
    parser.add_argument('--num-models', type=int, default=10,
                        help='Initial number of models (default: 10)')
    parser.add_argument('--num-elim-steps', type=int, default=100,
                        help='Steps before elimination (default: 100)')
    parser.add_argument('--reduce-factor', type=float, default=0.5,
                        help='Fraction of models to keep (default: 0.5)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Data directory (default: ./data)')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        args.data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        args.data_dir, train=False, download=True, transform=transform
    )
    
    # Split test into validation and test
    val_size = len(test_dataset) // 2
    test_size = len(test_dataset) - val_size
    val_dataset, test_dataset = torch.utils.data.random_split(
        test_dataset, [val_size, test_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    # Define model and optimizer factories
    def model_fn():
        return SimpleCNN()
    
    def optimizer_fn(params):
        return torch.optim.Adam(params, lr=args.lr)
    
    # Create trainer
    trainer = MultipleInitializationsTrainer(
        model_fn=model_fn,
        optimizer_fn=optimizer_fn,
        num_models=args.num_models,
        num_elim_steps=args.num_elim_steps,
        reduce_factor=args.reduce_factor,
        device=device
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Initial models: {args.num_models}")
    print(f"Elimination steps: {args.num_elim_steps}")
    print(f"Reduce factor: {args.reduce_factor}\n")
    print("=" * 70)
    
    best_model = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        num_epochs=args.epochs,
        verbose=True
    )
    
    print("=" * 70)
    
    # Final evaluation
    print("\nEvaluating best model...")
    train_acc = evaluate_accuracy(best_model, train_loader, device)
    val_acc = evaluate_accuracy(best_model, val_loader, device)
    test_acc = evaluate_accuracy(best_model, test_loader, device)
    
    print(f"\nFinal Results:")
    print(f"  Training Accuracy:   {train_acc:.2f}%")
    print(f"  Validation Accuracy: {val_acc:.2f}%")
    print(f"  Test Accuracy:       {test_acc:.2f}%")


if __name__ == '__main__':
    main()

