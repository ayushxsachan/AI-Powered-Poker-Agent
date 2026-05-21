"""Monte Carlo equity and outcome estimation for Hold'em study."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from poker_ai.engine.cards import Card
from poker_ai.engine.deck import Deck
from poker_ai.engine.hand_evaluator import HandEvaluator


@dataclass(frozen=True)
class EquityReport:
    simulations: int
    wins: int
    ties: int
    losses: int
    equity: float
    win_probability: float
    tie_probability: float
    loss_probability: float
    hand_distribution: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "simulations": self.simulations,
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
            "equity": self.equity,
            "win_probability": self.win_probability,
            "tie_probability": self.tie_probability,
            "loss_probability": self.loss_probability,
            "hand_distribution": self.hand_distribution,
        }


class EquityCalculator:
    """Estimate hero equity against unknown or partially known opponents."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def estimate(
        self,
        hero_cards: Iterable[Card],
        board_cards: Iterable[Card] | None = None,
        *,
        opponent_count: int = 1,
        known_opponent_cards: list[list[Card]] | None = None,
        simulations: int = 10_000,
    ) -> EquityReport:
        hero = list(hero_cards)
        board = list(board_cards or [])
        known_opponents = known_opponent_cards or []

        self._validate_inputs(hero, board, opponent_count, known_opponents, simulations)

        wins = ties = losses = 0
        distribution: Counter[str] = Counter()
        dead_cards = [*hero, *board, *[card for hand in known_opponents for card in hand]]

        for _ in range(simulations):
            deck = Deck(rng=self.rng)
            deck.remove(dead_cards)

            opponents = [list(cards) for cards in known_opponents]
            while len(opponents) < opponent_count:
                opponents.append(deck.draw(2))

            cards_to_come = 5 - len(board)
            runout = [*board, *(deck.draw(cards_to_come) if cards_to_come else [])]
            hero_eval = HandEvaluator.evaluate([*hero, *runout])
            villain_evals = [
                HandEvaluator.evaluate([*villain, *runout])
                for villain in opponents
            ]
            best_villain_score = max(evaluation.score for evaluation in villain_evals)
            distribution[hero_eval.name] += 1

            if hero_eval.score > best_villain_score:
                wins += 1
            elif hero_eval.score == best_villain_score:
                ties += 1
            else:
                losses += 1

        total = max(1, simulations)
        return EquityReport(
            simulations=simulations,
            wins=wins,
            ties=ties,
            losses=losses,
            equity=(wins + ties * 0.5) / total,
            win_probability=wins / total,
            tie_probability=ties / total,
            loss_probability=losses / total,
            hand_distribution={
                name: count / total for name, count in sorted(distribution.items())
            },
        )

    @staticmethod
    def _validate_inputs(
        hero: list[Card],
        board: list[Card],
        opponent_count: int,
        known_opponents: list[list[Card]],
        simulations: int,
    ) -> None:
        if len(hero) != 2:
            raise ValueError("hero_cards must contain exactly two cards")
        if not 0 <= len(board) <= 5:
            raise ValueError("board_cards must contain zero to five cards")
        if opponent_count < 1:
            raise ValueError("opponent_count must be at least one")
        if len(known_opponents) > opponent_count:
            raise ValueError("known_opponent_cards cannot exceed opponent_count")
        if simulations < 1:
            raise ValueError("simulations must be positive")
        if any(len(cards) != 2 for cards in known_opponents):
            raise ValueError("each known opponent must have exactly two cards")

        all_cards = [*hero, *board, *[card for hand in known_opponents for card in hand]]
        if len(set(all_cards)) != len(all_cards):
            raise ValueError("duplicate cards are not allowed")
