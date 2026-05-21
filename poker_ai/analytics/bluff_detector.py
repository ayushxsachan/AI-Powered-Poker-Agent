"""Bluff opportunity scoring."""

from __future__ import annotations

import random
from dataclasses import dataclass

from poker_ai.engine.cards import Card


@dataclass
class BluffContext:
    board_cards: list[Card]
    in_position: bool
    opponent_weakness: float
    stack_pressure: float
    base_frequency: float = 0.12


class BluffStrategy:
    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def should_bluff(self, context: BluffContext) -> bool:
        texture = self.board_texture_score(context.board_cards)
        probability = context.base_frequency
        probability += 0.12 if context.in_position else -0.03
        probability += 0.16 * context.opponent_weakness
        probability += 0.10 * min(1.0, context.stack_pressure)
        probability += 0.08 * (1.0 - texture)
        probability = min(0.45, max(0.02, probability))
        return self.rng.random() < probability

    @staticmethod
    def board_texture_score(board_cards: list[Card]) -> float:
        """Higher values mean wetter boards with more made/drawing hands."""

        if len(board_cards) < 3:
            return 0.25
        ranks = sorted({int(card.rank) for card in board_cards})
        suits = [card.suit for card in board_cards]
        suit_density = max(suits.count(suit) for suit in set(suits)) / len(suits)
        connected = 0.0
        for left, right in zip(ranks, ranks[1:]):
            if right - left <= 2:
                connected += 1.0
        connected_score = connected / max(1, len(ranks) - 1)
        paired = 1.0 - len(ranks) / len(board_cards)
        return min(1.0, 0.45 * suit_density + 0.4 * connected_score + 0.15 * paired)


class BluffDetector:
    """Simple detector for suspicious low-showdown-strength aggression."""

    def suspicious_aggression(
        self,
        *,
        bet_fraction_of_pot: float,
        board_texture: float,
        player_bluff_frequency: float,
    ) -> float:
        pressure = min(1.0, bet_fraction_of_pot)
        score = 0.5 * pressure + 0.25 * (1.0 - board_texture) + 0.25 * player_bluff_frequency
        return min(1.0, max(0.0, score))

