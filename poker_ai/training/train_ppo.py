"""Train a PPO poker agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from poker_ai.agents.ppo_agent import PPOAgent
from poker_ai.engine.poker_env import OBSERVATION_SIZE, PokerEnv


def train_local_ppo(total_steps: int, rollout_steps: int, output: Path) -> None:
    env = PokerEnv()
    agent = PPOAgent(OBSERVATION_SIZE, env.action_space.n)
    state, _ = env.reset()

    for step in range(total_steps):
        mask = env.legal_action_mask()
        action, log_prob, value = agent.select_action(state, mask)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.remember(state, action, reward, terminated or truncated, log_prob, value, mask)
        state = next_state

        if terminated or truncated:
            state, _ = env.reset()

        if (step + 1) % rollout_steps == 0:
            stats = agent.update(next_value=0.0)
            print(f"step={step + 1} loss={stats['loss']:.4f} entropy={stats['entropy']:.4f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    agent.save(output)


def train_sb3_ppo(total_steps: int, output: Path) -> None:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError("stable-baselines3 is required for --backend sb3") from exc

    env = PokerEnv()
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./poker_ai/logs/tensorboard")
    model.learn(total_timesteps=total_steps)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--rollout-steps", type=int, default=1_024)
    parser.add_argument("--backend", choices=["local", "sb3"], default="local")
    parser.add_argument("--output", type=Path, default=Path("poker_ai/models/ppo_agent.pt"))
    args = parser.parse_args()

    if args.backend == "sb3":
        train_sb3_ppo(args.steps, args.output)
    else:
        train_local_ppo(args.steps, args.rollout_steps, args.output)


if __name__ == "__main__":
    main()

