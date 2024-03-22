"""
Run benchmark comparisons between autonomous optimizer and standard optimizers.

This script loads a trained policy and compares its performance against
SGD, Momentum, Adam, and L-BFGS on various test problems.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

import sys
sys.path.append('..')

from benchmarks import (
    convex_quadratic, logistic_regression, robust_linear_regression, mlp,
    run_all_optimizers, accuracy
)


def plot_loss_comparison(results, problem_name, optimal_val=None, save_path=None):
    """Plot loss curves for all optimizers."""
    plt.figure(figsize=(10, 6))
    
    for name, (values, _) in results.items():
        if optimal_val is not None and name != 'ao':
            values = values - optimal_val
        plt.plot(values, label=name.upper(), linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Objective Value' if optimal_val is None else 'Distance from Optimal', 
               fontsize=12)
    plt.title(f'{problem_name} - Loss Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_accuracy_comparison(accuracies, problem_name, save_path=None):
    """Plot accuracy curves for classification problems."""
    plt.figure(figsize=(10, 6))
    
    for name, acc_values in accuracies.items():
        plt.plot(acc_values, label=name.upper(), linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'{problem_name} - Accuracy Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def benchmark_quadratic(policy, iterations=40, save_plots=False):
    """Benchmark on convex quadratic problem."""
    print("\n" + "="*70)
    print("CONVEX QUADRATIC BENCHMARK")
    print("="*70 + "\n")
    
    problem = convex_quadratic()
    print(f"Optimal objective value: {problem['optimal_val']:.6f}\n")
    
    tune_dict = {
        "sgd": {"hyperparams": {"lr": 5e-2}},
        "momentum": {"hyperparams": {"lr": 1e-2, "momentum": 0.7, "nesterov": True}},
        "adam": {"hyperparams": {"lr": 1e-1}},
        "lbfgs": {"hyperparams": {"lr": 1, "max_iter": 1}}
    }
    
    results = run_all_optimizers(problem, iterations, tune_dict, policy)
    
    if save_plots:
        plot_loss_comparison(results, "Convex Quadratic", 
                           problem['optimal_val'], 
                           save_path='quadratic_loss.png')
    
    return results


def benchmark_logistic(policy, iterations=40, save_plots=False):
    """Benchmark on logistic regression problem."""
    print("\n" + "="*70)
    print("LOGISTIC REGRESSION BENCHMARK")
    print("="*70 + "\n")
    
    problem = logistic_regression()
    x, y = problem['data']
    
    tune_dict = {
        "sgd": {"hyperparams": {"lr": 1e-1}},
        "momentum": {"hyperparams": {"lr": 5e-1, "momentum": 0.85, "nesterov": True}},
        "adam": {"hyperparams": {"lr": 5e-1}},
        "lbfgs": {"hyperparams": {"lr": 1, "max_iter": 1}}
    }
    
    results = run_all_optimizers(problem, iterations, tune_dict, policy)
    
    # Compute accuracies
    accuracies = {}
    for name, (_, trajectory) in results.items():
        acc_values = np.array([accuracy(model, x, y) for model in trajectory])
        accuracies[name] = acc_values
        print(f"{name.upper()} best accuracy: {acc_values.max():.4f}")
    
    if save_plots:
        plot_loss_comparison(results, "Logistic Regression", 
                           save_path='logistic_loss.png')
        plot_accuracy_comparison(accuracies, "Logistic Regression",
                               save_path='logistic_accuracy.png')
    
    return results, accuracies


def benchmark_robust_linear(policy, iterations=40, save_plots=False):
    """Benchmark on robust linear regression problem."""
    print("\n" + "="*70)
    print("ROBUST LINEAR REGRESSION BENCHMARK")
    print("="*70 + "\n")
    
    problem = robust_linear_regression()
    
    tune_dict = {
        "sgd": {"hyperparams": {"lr": 5e-1}},
        "momentum": {"hyperparams": {"lr": 5e-1, "momentum": 0.9, "nesterov": True}},
        "adam": {"hyperparams": {"lr": 5e-1}},
        "lbfgs": {"hyperparams": {"lr": 0.1, "max_iter": 1}}
    }
    
    results = run_all_optimizers(problem, iterations, tune_dict, policy)
    
    if save_plots:
        plot_loss_comparison(results, "Robust Linear Regression",
                           save_path='robust_linear_loss.png')
    
    return results


def benchmark_mlp(policy, iterations=40, save_plots=False):
    """Benchmark on MLP classification problem."""
    print("\n" + "="*70)
    print("MLP (2-LAYER NEURAL NETWORK) BENCHMARK")
    print("="*70 + "\n")
    
    problem = mlp()
    x, y = problem['dataset']
    
    tune_dict = {
        "sgd": {"hyperparams": {"lr": 3e-1}},
        "momentum": {"hyperparams": {"lr": 3e-1, "momentum": 0.9, "nesterov": True}},
        "adam": {"hyperparams": {"lr": 5e-1}},
        "lbfgs": {"hyperparams": {"lr": 0.1, "max_iter": 1}}
    }
    
    results = run_all_optimizers(problem, iterations, tune_dict, policy)
    
    # Compute accuracies
    accuracies = {}
    for name, (_, trajectory) in results.items():
        acc_values = np.array([accuracy(model, x, y) for model in trajectory])
        accuracies[name] = acc_values
        print(f"{name.upper()} best accuracy: {acc_values.max():.4f}")
    
    if save_plots:
        plot_loss_comparison(results, "MLP Classification",
                           save_path='mlp_loss.png')
        plot_accuracy_comparison(accuracies, "MLP Classification",
                               save_path='mlp_accuracy.png')
    
    return results, accuracies


def main():
    parser = argparse.ArgumentParser(
        description='Run benchmarks comparing optimizers'
    )
    parser.add_argument('--policy-path', type=str, required=True,
                        help='Path to trained policy model')
    parser.add_argument('--benchmark', type=str, default='all',
                        choices=['all', 'quadratic', 'logistic', 'robust_linear', 'mlp'],
                        help='Which benchmark to run (default: all)')
    parser.add_argument('--iterations', type=int, default=40,
                        help='Number of optimization iterations (default: 40)')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to files instead of displaying')
    args = parser.parse_args()
    
    # Load trained policy
    print(f"\nLoading policy from {args.policy_path}...")
    policy = PPO.load(args.policy_path)
    print("Policy loaded successfully!\n")
    
    # Run benchmarks
    if args.benchmark in ['all', 'quadratic']:
        benchmark_quadratic(policy, args.iterations, args.save_plots)
    
    if args.benchmark in ['all', 'logistic']:
        benchmark_logistic(policy, args.iterations, args.save_plots)
    
    if args.benchmark in ['all', 'robust_linear']:
        benchmark_robust_linear(policy, args.iterations, args.save_plots)
    
    if args.benchmark in ['all', 'mlp']:
        benchmark_mlp(policy, args.iterations, args.save_plots)
    
    print("\n" + "="*70)
    print("ALL BENCHMARKS COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

