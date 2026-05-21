"""Proximal Policy Optimization agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    from torch.distributions import Categorical
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    Categorical = None


def _require_torch() -> None:
    if torch is None or nn is None or Categorical is None:
        raise ImportError("PyTorch is required for PPOAgent. Install requirements.txt first.")


class ActorCritic(nn.Module if nn is not None else object):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        _require_torch()
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, observations):  # type: ignore[no-untyped-def]
        features = self.body(observations)
        return self.actor(features), self.critic(features).squeeze(-1)


@dataclass
class PPORolloutBuffer:
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.masks.clear()


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr: float = 3e-4
    epochs: int = 4
    batch_size: int = 256


class PPOAgent:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: PPOConfig | None = None,
        device: str | None = None,
    ) -> None:
        _require_torch()
        self.config = config or PPOConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ActorCritic(observation_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.buffer = PPORolloutBuffer()

    def select_action(self, state: np.ndarray, mask: np.ndarray) -> tuple[int, float, float]:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(mask.astype(bool), device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(state_tensor)
            logits = logits.masked_fill(~mask_tensor, -1e9)
            dist = Categorical(logits=logits)
            action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        mask: np.ndarray,
    ) -> None:
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)
        self.buffer.log_probs.append(log_prob)
        self.buffer.values.append(value)
        self.buffer.masks.append(mask)

    def update(self, next_value: float = 0.0) -> dict[str, float]:
        if not self.buffer.states:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        returns, advantages = self._compute_gae(next_value)
        states = torch.as_tensor(np.stack(self.buffer.states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(np.stack(self.buffer.masks).astype(bool), device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        stats = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        indices = np.arange(len(self.buffer.states))
        for _ in range(self.config.epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.config.batch_size):
                batch_idx = indices[start:start + self.config.batch_size]
                logits, values = self.model(states[batch_idx])
                logits = logits.masked_fill(~masks[batch_idx], -1e9)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(actions[batch_idx])
                ratio = torch.exp(log_probs - old_log_probs[batch_idx])

                unclipped = ratio * advantages_t[batch_idx]
                clipped = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages_t[batch_idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(values, returns_t[batch_idx])
                entropy = dist.entropy().mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                stats = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                }

        self.buffer.clear()
        return stats

    def _compute_gae(self, next_value: float) -> tuple[np.ndarray, np.ndarray]:
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = [*self.buffer.values, next_value]
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        for step in reversed(range(len(rewards))):
            non_terminal = 1.0 - float(dones[step])
            delta = rewards[step] + self.config.gamma * values[step + 1] * non_terminal - values[step]
            gae = delta + self.config.gamma * self.config.gae_lambda * non_terminal * gae
            advantages[step] = gae
        returns = advantages + np.asarray(self.buffer.values, dtype=np.float32)
        return returns, advantages

    def save(self, path: str | Path) -> None:
        torch.save({"model": self.model.state_dict(), "config": self.config}, Path(path))

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])

