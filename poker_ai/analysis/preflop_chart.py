"""Compact preflop chart hints for study mode."""

from __future__ import annotations

from dataclasses import dataclass

from poker_ai.engine.cards import Card, RANK_TO_SYMBOL


POSITION_GROUPS = {
    "early": {"utg", "utg+1", "early"},
    "middle": {"mp", "lj", "middle"},
    "late": {"hj", "co", "btn", "button", "dealer", "late"},
    "blind": {"sb", "bb", "small blind", "big blind", "blind"},
}


@dataclass(frozen=True)
class PreflopAdvice:
    hand_code: str
    strength_bucket: str
    recommendation: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "hand_code": self.hand_code,
            "strength_bucket": self.strength_bucket,
            "recommendation": self.recommendation,
            "reason": self.reason,
        }


class PreflopChart:
    """A simple tight-aggressive preflop chart.

    This is a training heuristic, not a solved strategy. It gives users a
    useful first-pass plan before postflop equity takes over.
    """

    PREMIUM = {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}
    STRONG = {
        "TT", "99", "88", "AQs", "AJs", "ATs", "KQs", "KJs", "QJs",
        "AQo", "AJo", "KQo",
    }
    PLAYABLE = {
        "77", "66", "55", "44", "33", "22", "A9s", "A8s", "A7s", "A6s",
        "A5s", "A4s", "A3s", "A2s", "KTs", "K9s", "QTs", "Q9s", "JTs",
        "J9s", "T9s", "98s", "87s", "76s", "65s", "AJo", "ATo", "KJo",
        "QJo",
    }
    SPECULATIVE = {
        "K8s", "K7s", "K6s", "K5s", "Q8s", "J8s", "T8s", "97s", "86s",
        "75s", "54s", "A9o", "A8o", "KTo", "QTo", "JTo",
    }

    def advise(
        self,
        hero_cards: list[Card],
        *,
        position: str = "unknown",
        facing_raise: bool = False,
        stack_depth_bb: float | None = None,
    ) -> PreflopAdvice | None:
        if len(hero_cards) != 2:
            return None

        code = self.hand_code(hero_cards)
        group = self._position_group(position)
        bucket = self._bucket(code)
        shallow = stack_depth_bb is not None and stack_depth_bb <= 25

        if bucket == "premium":
            recommendation = "raise or re-raise for value"
            reason = "This hand is strong enough from every position."
        elif bucket == "strong":
            if facing_raise and group in {"early", "middle"}:
                recommendation = "continue carefully"
                reason = "Strong hand, but early-position raises deserve respect."
            else:
                recommendation = "open raise"
                reason = "This is a profitable tight-aggressive opening hand."
        elif bucket == "playable":
            if facing_raise:
                recommendation = "usually fold to pressure"
                reason = "Playable hands lose value when someone has already raised."
            elif group in {"late", "blind"}:
                recommendation = "open in position"
                reason = "Late position lets this hand realize equity more often."
            else:
                recommendation = "fold or mix cautiously"
                reason = "This hand is marginal from early or middle position."
        elif bucket == "speculative":
            if not facing_raise and group == "late" and not shallow:
                recommendation = "steal or call selectively"
                reason = "Speculative suited/connected hands prefer position and deeper stacks."
            else:
                recommendation = "fold"
                reason = "This hand needs favorable position and implied odds."
        else:
            recommendation = "fold"
            reason = "This hand is outside a conservative training chart."

        if shallow and bucket in {"playable", "speculative", "trash"}:
            recommendation = "tighten up"
            reason += " Shallow stacks reduce implied odds."

        return PreflopAdvice(code, bucket, recommendation, reason)

    @staticmethod
    def hand_code(cards: list[Card]) -> str:
        left, right = sorted(cards, key=lambda card: int(card.rank), reverse=True)
        left_rank = RANK_TO_SYMBOL[left.rank]
        right_rank = RANK_TO_SYMBOL[right.rank]
        if left.rank == right.rank:
            return f"{left_rank}{right_rank}"
        suffix = "s" if left.suit == right.suit else "o"
        return f"{left_rank}{right_rank}{suffix}"

    @classmethod
    def _bucket(cls, code: str) -> str:
        if code in cls.PREMIUM:
            return "premium"
        if code in cls.STRONG:
            return "strong"
        if code in cls.PLAYABLE:
            return "playable"
        if code in cls.SPECULATIVE:
            return "speculative"
        return "trash"

    @staticmethod
    def _position_group(position: str) -> str:
        text = position.strip().lower()
        for group, values in POSITION_GROUPS.items():
            if text in values:
                return group
        return "unknown"

