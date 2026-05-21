"""Betting primitives and player state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from poker_ai.engine.cards import Card


class Action(IntEnum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    SMALL_RAISE = 3
    MEDIUM_RAISE = 4
    LARGE_RAISE = 5
    ALL_IN = 6

    @property
    def label(self) -> str:
        return self.name.lower()


RAISE_ACTIONS = {Action.SMALL_RAISE, Action.MEDIUM_RAISE, Action.LARGE_RAISE}


@dataclass
class BettingDecision:
    action: Action
    amount: int | None = None


@dataclass
class PlayerState:
    name: str
    stack: int
    hole_cards: list[Card] = field(default_factory=list)
    current_bet: int = 0
    committed: int = 0
    folded: bool = False
    all_in: bool = False
    acted: bool = False

    @property
    def active(self) -> bool:
        return not self.folded and not self.all_in

    @property
    def contesting(self) -> bool:
        return not self.folded

    def reset_for_hand(self) -> None:
        self.hole_cards.clear()
        self.current_bet = 0
        self.committed = 0
        self.folded = self.stack == 0
        self.all_in = self.stack == 0
        self.acted = False

    def reset_for_street(self) -> None:
        self.current_bet = 0
        self.acted = self.folded or self.all_in

    def commit(self, amount: int) -> int:
        if amount < 0:
            raise ValueError("amount must be non-negative")
        paid = min(self.stack, amount)
        self.stack -= paid
        self.current_bet += paid
        self.committed += paid
        if self.stack == 0:
            self.all_in = True
        return paid


def raise_size(action: Action, pot: int, big_blind: int) -> int:
    """Return the extra chips added on top of a call for a fixed raise bucket."""

    basis = max(pot, big_blind)
    if action == Action.SMALL_RAISE:
        return max(big_blind, basis // 2)
    if action == Action.MEDIUM_RAISE:
        return max(big_blind, basis)
    if action == Action.LARGE_RAISE:
        return max(big_blind, basis * 2)
    raise ValueError(f"{action} is not a raise action")
