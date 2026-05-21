"""Deep Q-Network agent implementation."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, NamedTuple

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    mask: np.ndarray
    next_mask: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for DQNAgent. Install requirements.txt first.")


class DQNNetwork(nn.Module if nn is not None else object):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        _require_torch()
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):  # type: ignore[no-untyped-def]
        return self.net(x)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 128
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 50_000
    target_update_interval: int = 1_000
    replay_capacity: int = 100_000


class DQNAgent:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: DQNConfig | None = None,
        device: str | None = None,
    ) -> None:
        _require_torch()
        self.config = config or DQNConfig()
        self.action_dim = action_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = DQNNetwork(observation_dim, action_dim).to(self.device)
        self.target = DQNNetwork(observation_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=self.config.lr)
        self.replay = ReplayBuffer(self.config.replay_capacity)
        self.steps = 0

    def select_action(self, state: np.ndarray, mask: np.ndarray, training: bool = True) -> int:
        legal_actions = np.flatnonzero(mask)
        if len(legal_actions) == 0:
            raise RuntimeError("No legal actions available")

        epsilon = self.epsilon if training else 0.0
        if random.random() < epsilon:
            return int(random.choice(legal_actions))

        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online(state_tensor).squeeze(0).cpu().numpy()
        q_values[mask == 0] = -np.inf
        return int(np.argmax(q_values))

    @property
    def epsilon(self) -> float:
        fraction = min(1.0, self.steps / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + fraction * (self.config.epsilon_final - self.config.epsilon_start)

    def store(self, transition: Transition) -> None:
        self.replay.push(transition)

    def train_step(self) -> float | None:
        if len(self.replay) < self.config.batch_size:
            return None

        batch = self.replay.sample(self.config.batch_size)
        states = torch.as_tensor(np.stack([item.state for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([item.next_state for item in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(np.stack([item.next_mask for item in batch]), dtype=torch.bool, device=self.device)

        q_values = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target(next_states)
            next_q_values = next_q_values.masked_fill(~next_masks, -1e9)
            targets = rewards + self.config.gamma * (1.0 - dones) * next_q_values.max(dim=1).values

        loss = torch.nn.functional.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.config.target_update_interval == 0:
            self.update_target()
        return float(loss.item())

    def update_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "steps": self.steps,
                "config": self.config,
            },
            Path(path),
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device)
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint["target"])
        self.steps = int(checkpoint.get("steps", 0))

