"""Run a trained model for N episodes and log episode durations to WandB and CSV.

Usage:
    python run_tests.py --env CartPole-v1 --model models/CartPole-v1_dqn.pt --episodes 100
"""
import argparse
import csv
import os
from pathlib import Path

import gymnasium as gym
from gymnasium import ActionWrapper, spaces
import numpy as np
import torch
import wandb

from rl.agents import DQNAgent


class DiscretizeAction(ActionWrapper):
    """Wrapper to discretize continuous action spaces for DQN."""
    def __init__(self, env, n_actions=11):
        super().__init__(env)
        self.n_actions = n_actions
        self.action_space = spaces.Discrete(n_actions)
        
        # Handle multi-dimensional action spaces
        if hasattr(env.action_space, 'shape') and len(env.action_space.shape) > 0:
            self.action_dim = env.action_space.shape[0]
            self.continuous_actions = []
            for i in range(self.action_dim):
                low = env.action_space.low[i]
                high = env.action_space.high[i]
                self.continuous_actions.append(np.linspace(low, high, n_actions))
        else:
            # Single dimension
            self.action_dim = 1
            low = env.action_space.low[0] if hasattr(env.action_space.low, '__getitem__') else env.action_space.low
            high = env.action_space.high[0] if hasattr(env.action_space.high, '__getitem__') else env.action_space.high
            self.continuous_actions = [np.linspace(low, high, n_actions)]
    
    def action(self, action):
        """Convert discrete action to continuous."""
        if self.action_dim == 1:
            return np.array([self.continuous_actions[0][action]])
        else:
            # For multi-dimensional, map single discrete action to continuous vector
            # Simple approach: use same discrete index for all dimensions
            return np.array([self.continuous_actions[i][action] for i in range(self.action_dim)])


def make_env(env_name, seed=None, record_video=False, video_folder="videos", algo=None,
             n_discrete_actions=11, video_frequency=50):
    # Create environment with appropriate render mode
    render_mode = 'rgb_array' if record_video else None
    env = gym.make(env_name, render_mode=render_mode)
    
    # Check if action space is continuous (Box) and discretize it
    if isinstance(env.action_space, spaces.Box):
        print(f"Detected continuous action space for {env_name}. Discretizing into {n_discrete_actions} actions.")
        env = DiscretizeAction(env, n_actions=n_discrete_actions)
    
    if seed is not None:
        try:
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        except Exception:
            pass
    
    if record_video:
        # Organize videos: videos/tests/{algo}/{env_name}/
        if algo:
            video_path = os.path.join(video_folder, "tests", algo.upper(), env_name)
        else:
            video_path = os.path.join(video_folder, "tests" ,env_name)
        os.makedirs(video_path, exist_ok=True)
        def should_record(episode_id):
            # Capture the first episode and then every Nth episode for evaluation snapshots.
            return episode_id == 0 or ((episode_id + 1) % video_frequency == 0)

        env = gym.wrappers.RecordVideo(env, video_path, episode_trigger=should_record, name_prefix=env_name)
    
    return env


def run_tests(env_name: str, model_path: str, episodes: int = 100, device: str = None, project: str = "rl-dqn", entity: str = None, record_video: bool = False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    run = wandb.init(project=project, entity=entity, config={'env': env_name, 'model_path': model_path, 'episodes': episodes})

    # Infer algo from model path (e.g., "CartPole-v1_dqn.pt" -> "dqn")
    model_name = Path(model_path).stem
    algo = None
    if '_dqn' in model_name.lower():
        algo = 'dqn'
    elif '_ddqn' in model_name.lower():
        algo = 'ddqn'
    
    env = make_env(env_name, record_video=record_video, algo=algo)
    obs, _ = env.reset()
    n_observations = len(obs)
    n_actions = env.action_space.n

    # instantiate agent only to hold network structure; weights will be loaded
    agent = DQNAgent(n_observations, n_actions, device=device)
    agent.load(model_path)
    agent.policy_net.eval()

    durations = []
    os.makedirs('results', exist_ok=True)
    csv_path = f'results/{Path(model_path).stem}_test_results.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['episode', 'length'])
        for i in range(episodes):
            obs, _ = env.reset()
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            done = False
            t = 0
            while not done:
                with torch.no_grad():
                    action = agent.policy_net(state).max(1).indices.view(1, 1)
                obs, reward, terminated, truncated, _ = env.step(int(action.item()))
                done = terminated or truncated
                if terminated:
                    state = None
                else:
                    state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                t += 1
            durations.append(t)
            writer.writerow([i, t])
            run.log({'test/episode_length': t, 'test/episode': i})

    run.finish()
    env.close()
    print(f"Saved results to {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--record-video', action='store_true')
    parser.add_argument('--project', type=str, default='rl-dqn')
    parser.add_argument('--entity', type=str, default=None)
    args = parser.parse_args()

    run_tests(env_name=args.env, model_path=args.model, episodes=args.episodes, record_video=args.record_video, project=args.project, entity=args.entity)
