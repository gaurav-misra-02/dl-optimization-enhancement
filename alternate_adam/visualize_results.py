"""
Visualization script for comparing optimizer performance.

This script reads training logs and creates comparison plots between
standard Adam and Alternate Adam optimizers.
"""

import argparse
import os
import matplotlib.pyplot as plt


def load_log_file(filepath):
    """Load numeric data from a log file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Log file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]
    return data


def plot_comparison(data1, data2, label1, label2, xlabel, ylabel, title, save_path=None):
    """Create a comparison plot between two datasets."""
    plt.figure(figsize=(10, 6))
    plt.plot(data1, label=label1, linewidth=2)
    plt.plot(data2, label=label2, linewidth=2)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize optimizer comparison results'
    )
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Directory containing log files (default: ./logs)')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory to save plots (default: ./results)')
    parser.add_argument('--show', action='store_true',
                        help='Display plots instead of saving')
    args = parser.parse_args()
    
    # Create output directory
    if not args.show:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # File paths for Adam
    adam_train = os.path.join(args.log_dir, 'train_loss_adam.txt')
    adam_val_loss = os.path.join(args.log_dir, 'val_loss_adam.txt')
    adam_val_acc = os.path.join(args.log_dir, 'val_acc_adam.txt')
    
    # File paths for Alternate Adam
    alt_adam_train = os.path.join(args.log_dir, 'train_loss_alternate_adam.txt')
    alt_adam_val_loss = os.path.join(args.log_dir, 'val_loss_alternate_adam.txt')
    alt_adam_val_acc = os.path.join(args.log_dir, 'val_acc_alternate_adam.txt')
    
    # Check if all files exist
    required_files = [
        adam_train, adam_val_loss, adam_val_acc,
        alt_adam_train, alt_adam_val_loss, alt_adam_val_acc
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Error: Missing log files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease run training for both optimizers first:")
        print("  python train_mnist.py --optimizer adam")
        print("  python train_mnist.py --optimizer alternate_adam")
        return
    
    # Load data
    print("Loading log files...")
    adam_train_data = load_log_file(adam_train)
    adam_val_loss_data = load_log_file(adam_val_loss)
    adam_val_acc_data = load_log_file(adam_val_acc)
    
    alt_adam_train_data = load_log_file(alt_adam_train)
    alt_adam_val_loss_data = load_log_file(alt_adam_val_loss)
    alt_adam_val_acc_data = load_log_file(alt_adam_val_acc)
    
    print("Creating plots...")
    
    # Plot 1: Training Loss
    save_path = None if args.show else os.path.join(args.output_dir, 'train_loss_comparison.png')
    plot_comparison(
        adam_train_data,
        alt_adam_train_data,
        'Standard Adam',
        'Alternate Adam',
        'Training Steps',
        'Loss',
        'Training Loss Comparison',
        save_path
    )
    
    # Plot 2: Validation Loss
    save_path = None if args.show else os.path.join(args.output_dir, 'val_loss_comparison.png')
    plot_comparison(
        adam_val_loss_data,
        alt_adam_val_loss_data,
        'Standard Adam',
        'Alternate Adam',
        'Epoch',
        'Validation Loss',
        'Validation Loss Comparison',
        save_path
    )
    
    # Plot 3: Validation Accuracy
    save_path = None if args.show else os.path.join(args.output_dir, 'val_accuracy_comparison.png')
    plot_comparison(
        adam_val_acc_data,
        alt_adam_val_acc_data,
        'Standard Adam',
        'Alternate Adam',
        'Epoch',
        'Accuracy (%)',
        'Validation Accuracy Comparison',
        save_path
    )
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 60)
    print(f"{'Metric':<30} {'Adam':<15} {'Alternate Adam':<15}")
    print("-" * 60)
    print(f"{'Final Train Loss':<30} {adam_train_data[-1]:<15.6f} {alt_adam_train_data[-1]:<15.6f}")
    print(f"{'Final Val Loss':<30} {adam_val_loss_data[-1]:<15.4f} {alt_adam_val_loss_data[-1]:<15.4f}")
    print(f"{'Best Val Accuracy (%)':<30} {max(adam_val_acc_data):<15.2f} {max(alt_adam_val_acc_data):<15.2f}")
    print(f"{'Final Val Accuracy (%)':<30} {adam_val_acc_data[-1]:<15.2f} {alt_adam_val_acc_data[-1]:<15.2f}")
    print("-" * 60)
    
    if not args.show:
        print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == '__main__':
    main()

