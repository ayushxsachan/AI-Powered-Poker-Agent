"""Texas Hold'em hand evaluator.

The evaluator is deliberately straightforward: it checks every five-card
combination out of the available cards. That is plenty fast for gameplay,
testing, and small simulations, and it keeps ranking behavior easy to audit.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations
from typing import Iterable

from poker_ai.engine.cards import Card


class HandCategory(IntEnum):
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10


CATEGORY_NAMES = {
    HandCategory.HIGH_CARD: "High Card",
    HandCategory.PAIR: "Pair",
    HandCategory.TWO_PAIR: "Two Pair",
    HandCategory.THREE_OF_A_KIND: "Three of a Kind",
    HandCategory.STRAIGHT: "Straight",
    HandCategory.FLUSH: "Flush",
    HandCategory.FULL_HOUSE: "Full House",
    HandCategory.FOUR_OF_A_KIND: "Four of a Kind",
    HandCategory.STRAIGHT_FLUSH: "Straight Flush",
    HandCategory.ROYAL_FLUSH: "Royal Flush",
}


@dataclass(frozen=True)
class HandEvaluation:
    category: HandCategory
    tiebreakers: tuple[int, ...]
    cards: tuple[Card, ...]

    @property
    def score(self) -> tuple[int, ...]:
        return (int(self.category), *self.tiebreakers)

    @property
    def name(self) -> str:
        return CATEGORY_NAMES[self.category]

    def __lt__(self, other: "HandEvaluation") -> bool:
        return self.score < other.score

    def __str__(self) -> str:
        rendered = " ".join(card.pretty for card in sorted(self.cards, reverse=True))
        return f"{self.name}: {rendered}"


class HandEvaluator:
    """Evaluate and compare poker hands."""

    @classmethod
    def evaluate(cls, cards: Iterable[Card]) -> HandEvaluation:
        available = tuple(cards)
        if len(available) < 5:
            raise ValueError("At least five cards are required for evaluation")
        if len(set(available)) != len(available):
            raise ValueError("Duplicate cards are not allowed")

        return max(cls.evaluate_five(combo) for combo in combinations(available, 5))

    @staticmethod
    def evaluate_five(cards: Iterable[Card]) -> HandEvaluation:
        hand = tuple(cards)
        if len(hand) != 5:
            raise ValueError("Exactly five cards are required")

        ranks = sorted((int(card.rank) for card in hand), reverse=True)
        counts = Counter(ranks)
        count_rank_pairs = sorted(
            ((count, rank) for rank, count in counts.items()),
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        is_flush = len({card.suit for card in hand}) == 1
        straight_high = HandEvaluator._straight_high(ranks)

        if is_flush and straight_high:
            category = (
                HandCategory.ROYAL_FLUSH
                if straight_high == 14 and set(ranks) == {10, 11, 12, 13, 14}
                else HandCategory.STRAIGHT_FLUSH
            )
            return HandEvaluation(category, (straight_high,), hand)

        if count_rank_pairs[0][0] == 4:
            quad_rank = count_rank_pairs[0][1]
            kicker = max(rank for rank in ranks if rank != quad_rank)
            return HandEvaluation(HandCategory.FOUR_OF_A_KIND, (quad_rank, kicker), hand)

        if [pair[0] for pair in count_rank_pairs] == [3, 2]:
            return HandEvaluation(
                HandCategory.FULL_HOUSE,
                (count_rank_pairs[0][1], count_rank_pairs[1][1]),
                hand,
            )

        if is_flush:
            return HandEvaluation(HandCategory.FLUSH, tuple(ranks), hand)

        if straight_high:
            return HandEvaluation(HandCategory.STRAIGHT, (straight_high,), hand)

        if count_rank_pairs[0][0] == 3:
            trips = count_rank_pairs[0][1]
            kickers = tuple(rank for rank in ranks if rank != trips)
            return HandEvaluation(HandCategory.THREE_OF_A_KIND, (trips, *kickers), hand)

        if [pair[0] for pair in count_rank_pairs] == [2, 2, 1]:
            pair_ranks = sorted(
                [rank for rank, count in counts.items() if count == 2],
                reverse=True,
            )
            kicker = max(rank for rank, count in counts.items() if count == 1)
            return HandEvaluation(HandCategory.TWO_PAIR, (*pair_ranks, kicker), hand)

        if count_rank_pairs[0][0] == 2:
            pair_rank = count_rank_pairs[0][1]
            kickers = tuple(rank for rank in ranks if rank != pair_rank)
            return HandEvaluation(HandCategory.PAIR, (pair_rank, *kickers), hand)

        return HandEvaluation(HandCategory.HIGH_CARD, tuple(ranks), hand)

    @staticmethod
    def _straight_high(ranks: list[int]) -> int | None:
        unique = sorted(set(ranks), reverse=True)
        if len(unique) != 5:
            return None
        if unique == [14, 5, 4, 3, 2]:
            return 5
        if unique[0] - unique[-1] == 4:
            return unique[0]
        return None

    @classmethod
    def winners(
        cls,
        contenders: Iterable[int],
        hole_cards: dict[int, list[Card]],
        community_cards: list[Card],
    ) -> tuple[list[int], dict[int, HandEvaluation]]:
        evaluations = {
            index: cls.evaluate([*hole_cards[index], *community_cards])
            for index in contenders
        }
        best_score = max(evaluation.score for evaluation in evaluations.values())
        winners = [
            index for index, evaluation in evaluations.items()
            if evaluation.score == best_score
        ]
        return winners, evaluations

