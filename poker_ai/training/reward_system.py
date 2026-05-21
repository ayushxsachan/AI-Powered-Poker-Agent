"""Reward shaping helpers for RL poker agents."""

from __future__ import annotations

from dataclasses import dataclass

from poker_ai.engine.betting import Action


@dataclass
class RewardConfig:
    chip_scale: float = 10.0
    smart_fold_bonus: float = 0.1
    bluff_success_bonus: float = 0.4
    reckless_all_in_penalty: float = 0.5


class RewardSystem:
    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def calculate(
        self,
        *,
        stack_delta: int,
        big_blind: int,
        action: Action,
        won_hand: bool,
        smart_fold: bool = False,
        bluff_success: bool = False,
        reckless_all_in: bool = False,
    ) -> float:
        reward = stack_delta / max(1.0, big_blind * self.config.chip_scale)
        if smart_fold:
            reward += self.config.smart_fold_bonus
        if bluff_success:
            reward += self.config.bluff_success_bonus
        if action == Action.ALL_IN and reckless_all_in and not won_hand:
            reward -= self.config.reckless_all_in_penalty
        return float(reward)

