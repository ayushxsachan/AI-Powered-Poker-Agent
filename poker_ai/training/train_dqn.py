"""Train a DQN poker agent."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from poker_ai.agents.dqn_agent import DQNAgent, Transition
from poker_ai.engine.poker_env import OBSERVATION_SIZE, PokerEnv


def train_local_dqn(total_steps: int, output: Path) -> None:
    env = PokerEnv()
    agent = DQNAgent(OBSERVATION_SIZE, env.action_space.n)
    state, _ = env.reset()
    mask = env.legal_action_mask()

    losses: list[float] = []
    for step in range(total_steps):
        action = agent.select_action(state, mask, training=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_mask = env.legal_action_mask()
        agent.store(Transition(state, action, reward, next_state, terminated or truncated, mask, next_mask))
        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)

        state, mask = next_state, next_mask
        if terminated or truncated:
            state, _ = env.reset()
            mask = env.legal_action_mask()

        if (step + 1) % 1_000 == 0:
            recent_loss = np.mean(losses[-100:]) if losses else 0.0
            print(f"step={step + 1} epsilon={agent.epsilon:.3f} loss={recent_loss:.4f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    agent.save(output)


def train_sb3_dqn(total_steps: int, output: Path) -> None:
    try:
        from stable_baselines3 import DQN
    except ImportError as exc:
        raise ImportError("stable-baselines3 is required for --backend sb3") from exc

    env = PokerEnv()
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./poker_ai/logs/tensorboard")
    model.learn(total_timesteps=total_steps)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--backend", choices=["local", "sb3"], default="local")
    parser.add_argument("--output", type=Path, default=Path("poker_ai/models/dqn_agent.pt"))
    args = parser.parse_args()

    if args.backend == "sb3":
        train_sb3_dqn(args.steps, args.output)
    else:
        train_local_dqn(args.steps, args.output)


if __name__ == "__main__":
    main()

