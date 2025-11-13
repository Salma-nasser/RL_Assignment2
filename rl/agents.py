import random
from collections import deque, namedtuple
import math
import torch
from torch import nn
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """Classic DQN agent (policy + target networks) with optimize logic.

    This class encapsulates neural nets and optimization utilities. It does not
    contain the outer environment loop; the train script should handle env steps
    and pass transitions to the agent.
    """

    def __init__(self, n_observations, n_actions, device, lr=1e-3, gamma=0.99,
                 batch_size=64, memory_size=10000, tau=0.005, amsgrad=False):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.policy_net = MLP(n_observations, n_actions).to(device)
        self.target_net = MLP(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(memory_size)
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=amsgrad)
        self.steps_done = 0

    def select_action(self, state, eps_threshold):
        # state: tensor shaped [1, obs]
        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def push_transition(self, *args):
        self.memory.push(*args)

    def soft_update_target(self):
        policy_state = self.policy_net.state_dict()
        target_state = self.target_net.state_dict()
        for k in policy_state:
            target_state[k] = policy_state[k] * self.tau + target_state[k] * (1 - self.tau)
        self.target_net.load_state_dict(target_state)

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return None
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def save(self, path: str):
        torch.save({'policy_state_dict': self.policy_net.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy_state_dict'])
        self.target_net.load_state_dict(ckpt['policy_state_dict'])


class DDQNAgent(DQNAgent):
    """Double DQN: uses policy_net to select best action and target_net to evaluate it."""

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return None
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            # action selection with policy_net
            next_state_actions = self.policy_net(non_final_next_states).max(1).indices.unsqueeze(1)
            # value evaluation with target_net
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_state_actions).squeeze(1)

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()
