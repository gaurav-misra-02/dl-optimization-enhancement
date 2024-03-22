"""
Training script for the autonomous optimizer policy using PPO.

This script trains an RL policy to act as an optimizer by learning from
a dataset of optimization problems.
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common import vec_env, monitor

import sys
sys.path.append('..')

from environment import OptimizationEnvironment
from benchmarks import convex_quadratic, logistic_regression, robust_linear_regression, mlp


def train_policy_on_problems(problem_type, num_problems, num_envs, total_timesteps, 
                             log_dir='./tb_logs', model_save_path='./trained_policy.zip'):
    """
    Train a policy on a specific type of optimization problem.
    
    Args:
        problem_type: Type of problem ('quadratic', 'logistic', 'robust_linear', 'mlp')
        num_problems: Number of problem instances in training dataset
        num_envs: Number of parallel environments
        total_timesteps: Total training timesteps
        log_dir: Directory for tensorboard logs
        model_save_path: Path to save trained policy
    
    Returns:
        Trained PPO policy
    """
    print(f"\n{'='*70}")
    print(f"Training policy on {problem_type} problems")
    print(f"{'='*70}\n")
    
    # Generate problem dataset
    print(f"Generating {num_problems} problem instances...")
    if problem_type == 'quadratic':
        dataset = [convex_quadratic() for _ in range(num_problems)]
    elif problem_type == 'logistic':
        dataset = [logistic_regression() for _ in range(num_problems)]
    elif problem_type == 'robust_linear':
        dataset = [robust_linear_regression() for _ in range(num_problems)]
    elif problem_type == 'mlp':
        dataset = [mlp() for _ in range(num_problems)]
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    print(f"Created dataset with {len(dataset)} problems\n")
    
    # Create vectorized environment
    print(f"Setting up {num_envs} parallel environments...")
    env = vec_env.DummyVecEnv([
        lambda: monitor.Monitor(
            OptimizationEnvironment(dataset, num_steps=40, history_len=25)
        )
        for _ in range(num_envs)
    ])
    
    print(f"Environment setup complete\n")
    
    # Create PPO policy
    print("Initializing PPO policy...")
    policy = PPO(
        'MlpPolicy',
        env,
        n_steps=2 if problem_type == 'quadratic' else 1,
        verbose=1,
        tensorboard_log=f'{log_dir}/{problem_type}'
    )
    
    print(f"Policy initialized\n")
    
    # Train
    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"TensorBoard logs: {log_dir}/{problem_type}\n")
    
    policy.learn(total_timesteps=total_timesteps)
    
    print(f"\nTraining complete!")
    
    # Save model
    policy.save(model_save_path)
    print(f"Policy saved to {model_save_path}\n")
    
    return policy


def main():
    parser = argparse.ArgumentParser(
        description='Train autonomous optimizer policy'
    )
    parser.add_argument('--problem-type', type=str, required=True,
                        choices=['quadratic', 'logistic', 'robust_linear', 'mlp'],
                        help='Type of optimization problem')
    parser.add_argument('--num-problems', type=int, default=90,
                        help='Number of problem instances (default: 90)')
    parser.add_argument('--num-envs', type=int, default=32,
                        help='Number of parallel environments (default: 32)')
    parser.add_argument('--num-passes', type=int, default=20,
                        help='Number of passes over dataset (default: 20)')
    parser.add_argument('--steps-per-episode', type=int, default=40,
                        help='Steps per optimization episode (default: 40)')
    parser.add_argument('--log-dir', type=str, default='./tb_logs',
                        help='TensorBoard log directory (default: ./tb_logs)')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save trained policy (default: policy_{problem_type}.zip)')
    args = parser.parse_args()
    
    # Calculate total timesteps
    total_timesteps = args.num_passes * args.steps_per_episode * args.num_problems
    
    # Default save path
    if args.save_path is None:
        args.save_path = f'policy_{args.problem_type}.zip'
    
    print("\nTraining Configuration:")
    print(f"  Problem type: {args.problem_type}")
    print(f"  Number of problems: {args.num_problems}")
    print(f"  Parallel environments: {args.num_envs}")
    print(f"  Passes over dataset: {args.num_passes}")
    print(f"  Steps per episode: {args.steps_per_episode}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Save path: {args.save_path}")
    
    # Train policy
    policy = train_policy_on_problems(
        problem_type=args.problem_type,
        num_problems=args.num_problems,
        num_envs=args.num_envs,
        total_timesteps=total_timesteps,
        log_dir=args.log_dir,
        model_save_path=args.save_path
    )
    
    print("\nTo view training progress, run:")
    print(f"  tensorboard --logdir {args.log_dir}")


if __name__ == '__main__':
    main()

