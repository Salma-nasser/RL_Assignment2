"""Run a trained model for N episodes and log episode durations to WandB and CSV.

Usage:
    python run_tests.py --env CartPole-v1 --model models/CartPole-v1_dqn.pt --episodes 100
"""
import argparse
import csv
import os
from pathlib import Path

import gymnasium as gym
import torch
import wandb

from rl.agents import DQNAgent


def make_env(env_name, seed=None, record_video=False, video_folder="videos"):
    env = gym.make(env_name)
    if seed is not None:
        try:
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        except Exception:
            pass
    if record_video:
        env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda e: True, name_prefix=env_name)
    return env


def run_tests(env_name: str, model_path: str, episodes: int = 100, device: str = None, project: str = "rl-dqn", entity: str = None, record_video: bool = False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    run = wandb.init(project=project, entity=entity, config={'env': env_name, 'model_path': model_path, 'episodes': episodes})

    env = make_env(env_name, record_video=record_video)
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
