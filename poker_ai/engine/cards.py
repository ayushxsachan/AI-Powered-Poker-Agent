"""Card primitives used by the poker engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class Suit(str, Enum):
    CLUBS = "c"
    DIAMONDS = "d"
    HEARTS = "h"
    SPADES = "s"

    @property
    def symbol(self) -> str:
        return {
            Suit.CLUBS: "C",
            Suit.DIAMONDS: "D",
            Suit.HEARTS: "H",
            Suit.SPADES: "S",
        }[self]


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


RANK_TO_SYMBOL = {
    Rank.TWO: "2",
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "T",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
}

SYMBOL_TO_RANK = {symbol: rank for rank, symbol in RANK_TO_SYMBOL.items()}
SYMBOL_TO_RANK.update({"10": Rank.TEN})

SUIT_SYMBOLS = {
    "c": Suit.CLUBS,
    "d": Suit.DIAMONDS,
    "h": Suit.HEARTS,
    "s": Suit.SPADES,
}


@dataclass(frozen=True)
class Card:
    """Immutable playing card.

    String form uses compact poker notation, for example ``As`` for ace of
    spades and ``Td`` for ten of diamonds.
    """

    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        return f"{RANK_TO_SYMBOL[self.rank]}{self.suit.value}"

    def __repr__(self) -> str:
        return f"Card.from_str('{self}')"

    def __lt__(self, other: "Card") -> bool:
        return (int(self.rank), self.suit.value) < (int(other.rank), other.suit.value)

    @property
    def pretty(self) -> str:
        return f"{RANK_TO_SYMBOL[self.rank]}{self.suit.symbol}"

    @classmethod
    def from_str(cls, value: str) -> "Card":
        text = value.strip()
        if len(text) < 2:
            raise ValueError(f"Invalid card notation: {value!r}")

        rank_text = text[:-1].upper()
        suit_text = text[-1].lower()
        if rank_text not in SYMBOL_TO_RANK:
            raise ValueError(f"Unknown card rank in {value!r}")
        if suit_text not in SUIT_SYMBOLS:
            raise ValueError(f"Unknown card suit in {value!r}")
        return cls(SYMBOL_TO_RANK[rank_text], SUIT_SYMBOLS[suit_text])


def cards_from_strings(values: list[str] | tuple[str, ...]) -> list[Card]:
    """Parse a collection of compact card strings."""

    return [Card.from_str(value) for value in values]
