"""Deck implementation."""

from __future__ import annotations

import random
from collections.abc import Iterable

from poker_ai.engine.cards import Card, Rank, Suit


class Deck:
    """Standard 52-card deck with injectable RNG for reproducible simulations."""

    def __init__(
        self,
        cards: Iterable[Card] | None = None,
        rng: random.Random | None = None,
        shuffle: bool = True,
    ) -> None:
        self.rng = rng or random.Random()
        self.cards = list(cards) if cards is not None else [
            Card(rank, suit) for suit in Suit for rank in Rank
        ]
        if shuffle:
            self.shuffle()

    def __len__(self) -> int:
        return len(self.cards)

    def shuffle(self) -> None:
        self.rng.shuffle(self.cards)

    def draw(self, count: int = 1) -> list[Card]:
        if count < 1:
            raise ValueError("count must be positive")
        if count > len(self.cards):
            raise ValueError("Cannot draw more cards than remain in the deck")

        drawn = self.cards[:count]
        del self.cards[:count]
        return drawn

    def draw_one(self) -> Card:
        return self.draw(1)[0]

    def remove(self, cards: Iterable[Card]) -> None:
        for card in cards:
            self.cards.remove(card)

