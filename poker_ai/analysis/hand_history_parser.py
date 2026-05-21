"""Generic post-session hand-history parser.

The parser recognizes common plain-text phrases from exported poker hand
histories. It is intentionally conservative and source-agnostic; unsupported
lines are ignored instead of guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from poker_ai.analytics.opponent_model import OpponentModel
from poker_ai.engine.betting import Action


HAND_START_RE = re.compile(r"^(?:PokerStars|Hand|Table|#)?\s*Hand\b|^Game #", re.IGNORECASE)
PLAYER_ACTION_RE = re.compile(
    r"^(?P<player>[^:\[]+):\s+"
    r"(?P<verb>folds|checks|calls|bets|raises|raises to|all-in|is all-in|posts)"
    r"(?:\s+(?P<amount>\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)
COLLECTED_RE = re.compile(
    r"^(?P<player>[^:]+?)\s+(?:collected|wins)\s+(?P<amount>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
SHOWS_RE = re.compile(r"^(?P<player>[^:]+):\s+shows\s+\[", re.IGNORECASE)
STREET_RE = re.compile(r"^\*\*\*\s+(?P<street>HOLE CARDS|FLOP|TURN|RIVER|SHOW DOWN|SUMMARY)", re.IGNORECASE)


@dataclass
class ParsedAction:
    player: str
    action: Action
    street: str
    amount: float | None = None


@dataclass
class ParsedHand:
    hand_id: int
    actions: list[ParsedAction] = field(default_factory=list)
    winners: set[str] = field(default_factory=set)
    showed: set[str] = field(default_factory=set)


@dataclass
class ParseReport:
    hands: list[ParsedHand]
    opponent_model: OpponentModel

    @property
    def hand_count(self) -> int:
        return len(self.hands)

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            name: stats.as_dict()
            for name, stats in sorted(self.opponent_model.players.items())
        }


class HandHistoryParser:
    """Parse exported hand histories for post-session analysis."""

    def parse_file(self, path: str | Path) -> ParseReport:
        return self.parse_text(Path(path).read_text(encoding="utf-8", errors="ignore"))

    def parse_text(self, text: str) -> ParseReport:
        hands: list[ParsedHand] = []
        current: ParsedHand | None = None
        street = "preflop"

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if self._starts_new_hand(line, current):
                if current and current.actions:
                    hands.append(current)
                current = ParsedHand(hand_id=len(hands) + 1)
                street = "preflop"
                continue

            if current is None:
                current = ParsedHand(hand_id=1)

            street_match = STREET_RE.match(line)
            if street_match:
                street = self._normalize_street(street_match.group("street"))
                continue

            action = self._parse_action(line, street)
            if action:
                current.actions.append(action)
                continue

            collected = COLLECTED_RE.match(line)
            if collected:
                current.winners.add(collected.group("player").strip())
                continue

            shown = SHOWS_RE.match(line)
            if shown:
                current.showed.add(shown.group("player").strip())

        if current and current.actions:
            hands.append(current)

        model = self._build_model(hands)
        return ParseReport(hands=hands, opponent_model=model)

    @staticmethod
    def _starts_new_hand(line: str, current: ParsedHand | None) -> bool:
        return current is not None and bool(HAND_START_RE.match(line))

    @staticmethod
    def _normalize_street(street: str) -> str:
        text = street.lower().replace(" ", "_")
        if text == "hole_cards":
            return "preflop"
        if text == "show_down":
            return "showdown"
        return text

    @staticmethod
    def _parse_action(line: str, street: str) -> ParsedAction | None:
        match = PLAYER_ACTION_RE.match(line)
        if not match:
            return None
        player = match.group("player").strip()
        verb = match.group("verb").lower()
        amount_text = match.group("amount")
        amount = float(amount_text) if amount_text else None

        if verb == "folds":
            action = Action.FOLD
        elif verb == "checks":
            action = Action.CHECK
        elif verb == "calls":
            action = Action.CALL
        elif verb in {"bets", "raises", "raises to"}:
            action = Action.MEDIUM_RAISE
        elif verb in {"all-in", "is all-in"}:
            action = Action.ALL_IN
        elif verb == "posts":
            return None
        else:
            return None
        return ParsedAction(player=player, action=action, street=street, amount=amount)

    @staticmethod
    def _build_model(hands: list[ParsedHand]) -> OpponentModel:
        model = OpponentModel()
        for hand in hands:
            players = sorted({action.player for action in hand.actions})
            model.start_hand(players)
            seen_vpip: set[str] = set()
            seen_pfr: set[str] = set()
            preflop_raisers = {
                action.player
                for action in hand.actions
                if action.street == "preflop"
                and action.action in {Action.SMALL_RAISE, Action.MEDIUM_RAISE, Action.LARGE_RAISE, Action.ALL_IN}
            }

            for action in hand.actions:
                faced_raise = bool(preflop_raisers - {action.player})
                suspected_bluff = (
                    action.action in {Action.SMALL_RAISE, Action.MEDIUM_RAISE, Action.LARGE_RAISE, Action.ALL_IN}
                    and action.street in {"turn", "river"}
                    and action.player not in hand.showed
                )
                is_raise = action.action in {
                    Action.SMALL_RAISE,
                    Action.MEDIUM_RAISE,
                    Action.LARGE_RAISE,
                    Action.ALL_IN,
                }
                voluntary = action.player not in seen_vpip
                preflop_for_rate = action.street == "preflop" and (
                    action.player not in seen_vpip or (is_raise and action.player not in seen_pfr)
                )
                model.observe_action(
                    action.player,
                    action.action,
                    preflop=preflop_for_rate,
                    voluntary=voluntary,
                    faced_raise=faced_raise and action.action == Action.FOLD,
                    suspected_bluff=suspected_bluff,
                )
                if action.street == "preflop" and action.action in {
                    Action.CALL,
                    Action.SMALL_RAISE,
                    Action.MEDIUM_RAISE,
                    Action.LARGE_RAISE,
                    Action.ALL_IN,
                }:
                    seen_vpip.add(action.player)
                if action.street == "preflop" and is_raise:
                    seen_pfr.add(action.player)

            for player in hand.showed | hand.winners:
                model.observe_showdown(
                    player,
                    won=player in hand.winners,
                    bluff_attempted=False,
                )
        return model
