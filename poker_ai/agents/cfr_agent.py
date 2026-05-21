"""Counterfactual Regret Minimization agents for small poker games."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np


ACTIONS = ("p", "b")


@dataclass
class CFRInfoSet:
    regret_sum: np.ndarray = field(default_factory=lambda: np.zeros(len(ACTIONS), dtype=np.float64))
    strategy_sum: np.ndarray = field(default_factory=lambda: np.zeros(len(ACTIONS), dtype=np.float64))

    def strategy(self, reach_probability: float) -> np.ndarray:
        positive_regrets = np.maximum(self.regret_sum, 0.0)
        normalizer = positive_regrets.sum()
        if normalizer > 0:
            strategy = positive_regrets / normalizer
        else:
            strategy = np.full(len(ACTIONS), 1.0 / len(ACTIONS))
        self.strategy_sum += reach_probability * strategy
        return strategy

    def average_strategy(self) -> np.ndarray:
        normalizer = self.strategy_sum.sum()
        if normalizer > 0:
            return self.strategy_sum / normalizer
        return np.full(len(ACTIONS), 1.0 / len(ACTIONS))


class KuhnCFRTrainer:
    """Classic tabular CFR trainer for Kuhn Poker."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()
        self.info_sets: dict[str, CFRInfoSet] = {}

    def train(self, iterations: int = 10_000) -> float:
        utility = 0.0
        cards = [1, 2, 3]
        for _ in range(iterations):
            self.rng.shuffle(cards)
            utility += self._cfr(cards[:2], "", 1.0, 1.0)
        return utility / max(1, iterations)

    def average_strategy(self) -> dict[str, dict[str, float]]:
        return {
            info_set: {
                action: float(probability)
                for action, probability in zip(ACTIONS, node.average_strategy())
            }
            for info_set, node in sorted(self.info_sets.items())
        }

    def _cfr(self, cards: list[int], history: str, p0: float, p1: float) -> float:
        plays = len(history)
        player = plays % 2
        opponent = 1 - player

        terminal = self._terminal_utility(cards, history, player)
        if terminal is not None:
            return terminal

        info_set_key = f"{cards[player]}:{history}"
        node = self.info_sets.setdefault(info_set_key, CFRInfoSet())
        strategy = node.strategy(p0 if player == 0 else p1)

        utilities = np.zeros(len(ACTIONS), dtype=np.float64)
        node_utility = 0.0
        for index, action in enumerate(ACTIONS):
            next_history = history + action
            if player == 0:
                utilities[index] = -self._cfr(cards, next_history, p0 * strategy[index], p1)
            else:
                utilities[index] = -self._cfr(cards, next_history, p0, p1 * strategy[index])
            node_utility += strategy[index] * utilities[index]

        reach = p1 if player == 0 else p0
        node.regret_sum += reach * (utilities - node_utility)
        return node_utility

    @staticmethod
    def _terminal_utility(cards: list[int], history: str, player: int) -> float | None:
        opponent = 1 - player
        if history == "pp":
            return 1.0 if cards[player] > cards[opponent] else -1.0
        if history in {"bp", "pbp"}:
            return 1.0
        if history in {"bb", "pbb"}:
            return 2.0 if cards[player] > cards[opponent] else -2.0
        return None


class LeducCFRTrainer(KuhnCFRTrainer):
    """Small Leduc-style CFR abstraction.

    This keeps the same two-action tabular interface but adds a public card to
    the information set and uses pair-vs-high-card showdown values. It is a
    compact bridge between Kuhn and full Hold'em abstractions.
    """

    def train(self, iterations: int = 10_000) -> float:
        utility = 0.0
        deck = [0, 0, 1, 1, 2, 2]
        for _ in range(iterations):
            self.rng.shuffle(deck)
            private_cards = deck[:2]
            public_card = deck[2]
            utility += self._cfr_leduc(private_cards, public_card, "", 1.0, 1.0)
        return utility / max(1, iterations)

    def _cfr_leduc(
        self,
        private_cards: list[int],
        public_card: int,
        history: str,
        p0: float,
        p1: float,
    ) -> float:
        plays = len(history)
        player = plays % 2
        terminal = self._terminal_utility_leduc(private_cards, public_card, history, player)
        if terminal is not None:
            return terminal

        info_set_key = f"{private_cards[player]}|{public_card}:{history}"
        node = self.info_sets.setdefault(info_set_key, CFRInfoSet())
        strategy = node.strategy(p0 if player == 0 else p1)

        utilities = np.zeros(len(ACTIONS), dtype=np.float64)
        node_utility = 0.0
        for index, action in enumerate(ACTIONS):
            next_history = history + action
            if player == 0:
                utilities[index] = -self._cfr_leduc(private_cards, public_card, next_history, p0 * strategy[index], p1)
            else:
                utilities[index] = -self._cfr_leduc(private_cards, public_card, next_history, p0, p1 * strategy[index])
            node_utility += strategy[index] * utilities[index]

        reach = p1 if player == 0 else p0
        node.regret_sum += reach * (utilities - node_utility)
        return node_utility

    @staticmethod
    def _terminal_utility_leduc(
        private_cards: list[int],
        public_card: int,
        history: str,
        player: int,
    ) -> float | None:
        opponent = 1 - player
        if history in {"bp", "pbp"}:
            return 1.0
        if history in {"pp", "bb", "pbb"}:
            own_pair = private_cards[player] == public_card
            opp_pair = private_cards[opponent] == public_card
            pot_units = 2.0 if history in {"bb", "pbb"} else 1.0
            if own_pair != opp_pair:
                return pot_units if own_pair else -pot_units
            if private_cards[player] == private_cards[opponent]:
                return 0.0
            return pot_units if private_cards[player] > private_cards[opponent] else -pot_units
        return None


class CFRPokerAgent:
    """Thin wrapper that trains and samples from a tabular CFR strategy."""

    def __init__(self, game: str = "kuhn", iterations: int = 10_000) -> None:
        if game == "kuhn":
            self.trainer: KuhnCFRTrainer = KuhnCFRTrainer()
        elif game == "leduc":
            self.trainer = LeducCFRTrainer()
        else:
            raise ValueError("game must be 'kuhn' or 'leduc'")
        self.trainer.train(iterations)

    def strategy(self) -> dict[str, dict[str, float]]:
        return self.trainer.average_strategy()

