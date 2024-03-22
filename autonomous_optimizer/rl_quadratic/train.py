"""
Training script for the RL quadratic optimizer.

This script trains an actor network using PPO to optimize convex quadratic functions.
"""

import argparse
import torch
import numpy as np
import scipy.linalg

from environment import QuadraticEnv, DIM
from actor import ActorNetwork
from ppo import PPOTrainer
from utils import create_quadratic_problem, rollout


def train(
    n_episodes=90,
    print_freq=10,
    policy_lr=1e-3,
    clip_factor=0.2,
    hidden_dim=64,
    device=None
):
    """
    Train the quadratic optimizer using PPO.
    
    Args:
        n_episodes: Number of training episodes
        print_freq: Frequency of printing progress
        policy_lr: Learning rate for policy
        clip_factor: PPO clipping parameter
        hidden_dim: Hidden layer dimension for actor
        device: torch device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training on device: {device}\n")
    
    # Create environment and model
    problem_info = create_quadratic_problem(dim=DIM, device=device)
    env = QuadraticEnv(Q_mat=problem_info["Q"], c_mat=problem_info["c"], device=device)
    
    model = ActorNetwork(
        obs_state_space=DIM + 1,  # parameters + objective
        action_state_space=DIM,
        hidden_dim=hidden_dim
    ).to(device)
    
    print("Environment and model created")
    print(f"Problem dimension: {DIM}")
    print(f"Actor hidden dimension: {hidden_dim}\n")
    
    # Test rollout
    train_data, reward = rollout(model, env, device=device)
    print(f"Test rollout complete. Reward: {reward:.2f}\n")
    
    # Create PPO trainer
    ppo = PPOTrainer(
        actor=model,
        policy_lr=policy_lr,
        clip_factor=clip_factor,
        policy_kldiv_bound=0.01,
        max_policy_train_steps=50,
        device=device
    )
    print("PPO trainer initialized\n")
    
    # Training loop
    episodic_rewards = []
    print("="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    for episode_num in range(n_episodes):
        # Perform rollout
        train_data, reward = rollout(model, env, device=device)
        episodic_rewards.append(reward)
        
        # Prepare training data
        permute_idxs = np.random.permutation(len(train_data[0]))
        obs = torch.tensor(train_data[0][permute_idxs], dtype=torch.float32, device=device)
        acts = torch.tensor(train_data[1][permute_idxs], dtype=torch.float32, device=device)
        gaes = torch.tensor(train_data[3][permute_idxs], dtype=torch.float32, device=device)
        act_log_probs = torch.tensor(train_data[4][permute_idxs], dtype=torch.float32, device=device)
        
        # Update policy
        ppo.train_policy(obs, acts, gaes, act_log_probs)
        
        # Print progress
        if (episode_num + 1) % print_freq == 0:
            avg_reward = np.mean(episodic_rewards[-print_freq:])
            print(f'Episode {episode_num + 1}/{n_episodes} | '
                  f'Avg Reward: {avg_reward:.2f} | '
                  f'Last Reward: {reward:.2f} | '
                  f'Final Position: {env._agent_location.cpu().numpy()}')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70 + "\n")
    
    # Compute optimal solution
    solution = scipy.linalg.solve(
        problem_info["Q"].numpy(),
        -problem_info["c"].numpy(),
        assume_a="pos"
    )
    
    print(f"Optimal solution: {solution}")
    print(f"Optimal value: {problem_info['optimal_val']:.6f}")
    print(f"\nFinal training statistics:")
    print(f"  Average reward (last 10 episodes): {np.mean(episodic_rewards[-10:]):.2f}")
    print(f"  Best reward: {max(episodic_rewards):.2f}")
    print(f"  Worst reward: {min(episodic_rewards):.2f}")
    
    return model, episodic_rewards


def main():
    parser = argparse.ArgumentParser(
        description='Train RL quadratic optimizer'
    )
    parser.add_argument('--n-episodes', type=int, default=90,
                        help='Number of training episodes (default: 90)')
    parser.add_argument('--print-freq', type=int, default=10,
                        help='Print frequency (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--clip', type=float, default=0.2,
                        help='PPO clip factor (default: 0.2)')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden layer dimension (default: 64)')
    parser.add_argument('--save-model', type=str, default=None,
                        help='Path to save trained model')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Train
    model, rewards = train(
        n_episodes=args.n_episodes,
        print_freq=args.print_freq,
        policy_lr=args.lr,
        clip_factor=args.clip,
        hidden_dim=args.hidden_dim
    )
    
    # Save model if requested
    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"\nModel saved to {args.save_model}")


if __name__ == "__main__":
    main()

