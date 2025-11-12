"""Train DQN/DDQN agents on Gymnasium environments with WandB logging.

Usage examples (from project root):
    python train.py --env CartPole-v1 --algo dqn --episodes 200 --record-video
    python train.py --env MountainCar-v0 --algo ddqn --episodes 400 --batch-size 128 --learning-rate 1e-4

The script saves the final model to `models/{env}_{algo}.pt` and logs metrics to WandB.
"""
import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import wandb

from rl.agents import DQNAgent, DDQNAgent


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
        # This will record episodes where the trigger is True; here we record all episodes.
        env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda e: True, name_prefix=env_name)
    return env


def train(
    env_name: str,
    algo: str,
    episodes: int = 200,
    lr: float = 3e-4,
    gamma: float = 0.99,
    batch_size: int = 128,
    memory_size: int = 10000,
    eps_start: float = 0.9,
    eps_end: float = 0.01,
    eps_decay: float = 2500,
    tau: float = 0.005,
    record_video: bool = False,
    device: str = None,
    project: str = "rl-dqn",
    entity: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    run = wandb.init(project=project, entity=entity, config={
        'env': env_name,
        'algo': algo,
        'episodes': episodes,
        'lr': lr,
        'gamma': gamma,
        'batch_size': batch_size,
        'memory_size': memory_size,
        'eps_start': eps_start,
        'eps_end': eps_end,
        'eps_decay': eps_decay,
        'tau': tau,
    })
    cfg = run.config

    env = make_env(env_name, record_video=record_video)
    n_actions = env.action_space.n
    obs, _ = env.reset()
    n_observations = len(obs)

    AgentClass = DQNAgent if algo.lower() == 'dqn' else DDQNAgent
    agent = AgentClass(n_observations, n_actions, device=device, lr=lr, gamma=gamma,
                       batch_size=batch_size, memory_size=memory_size, tau=tau)

    episode_durations = []

    for i_episode in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0.0
        t = 0
        while True:
            eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * agent.steps_done / eps_decay)
            action = agent.select_action(state, eps_threshold)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            reward_t = torch.tensor([reward], device=device)
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            agent.push_transition(state, action, next_state, reward_t)
            state = next_state
            total_reward += float(reward)
            t += 1

            loss = agent.optimize()
            if loss is not None:
                run.log({'train/loss': loss, 'train/eps': eps_threshold, 'train/step': agent.steps_done})

            agent.soft_update_target()

            if done:
                episode_durations.append(t)
                run.log({'train/episode_reward': total_reward, 'train/episode_length': t, 'train/episode': i_episode})
                break

        # periodic checkpoint
        if (i_episode + 1) % 50 == 0:
            os.makedirs('models', exist_ok=True)
            model_path = f"models/{env_name}_{algo}.pt"
            agent.save(model_path)
            run.save(model_path)

    # final save
    os.makedirs('models', exist_ok=True)
    model_path = f"models/{env_name}_{algo}.pt"
    agent.save(model_path)
    run.save(model_path)
    run.finish()
    env.close()


if __name__ == '__main__':
    import math

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--algo', type=str, choices=['dqn', 'ddqn'], default='dqn')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--learning-rate', '--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--memory-size', type=int, default=10000)
    parser.add_argument('--eps-start', type=float, default=0.9)
    parser.add_argument('--eps-end', type=float, default=0.01)
    parser.add_argument('--eps-decay', type=float, default=2500)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--record-video', action='store_true')
    parser.add_argument('--project', type=str, default='rl-dqn')
    parser.add_argument('--entity', type=str, default=None)
    args = parser.parse_args()

    train(env_name=args.env, algo=args.algo, episodes=args.episodes, lr=args.learning_rate,
          gamma=args.gamma, batch_size=args.batch_size, memory_size=args.memory_size,
          eps_start=args.eps_start, eps_end=args.eps_end, eps_decay=args.eps_decay,
          tau=args.tau, record_video=args.record_video, project=args.project, entity=args.entity)
