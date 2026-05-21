"""Tight-aggressive rule-based poker bot."""

from __future__ import annotations

import random
from dataclasses import dataclass

from poker_ai.analytics.bluff_detector import BluffContext, BluffStrategy
from poker_ai.engine.betting import Action
from poker_ai.engine.game_manager import GamePhase, TexasHoldemGame
from poker_ai.engine.hand_evaluator import HandCategory, HandEvaluator


PREMIUM_RANKS = {14, 13, 12, 11, 10}


@dataclass
class RuleBasedConfig:
    aggression: float = 0.72
    bluff_frequency: float = 0.12
    call_tolerance: float = 0.36


class RuleBasedAgent:
    """Baseline bot using hand strength, pot odds, position, and bluff spots."""

    def __init__(
        self,
        config: RuleBasedConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config or RuleBasedConfig()
        self.rng = rng or random.Random()
        self.bluff_strategy = BluffStrategy(rng=self.rng)

    def act(self, game: TexasHoldemGame, player_index: int) -> Action:
        legal = game.legal_actions(player_index)
        if not legal:
            raise RuntimeError("No legal actions available")

        player = game.players[player_index]
        call_amount = max(0, game.current_bet - player.current_bet)
        pot_odds = call_amount / max(1, game.pot + call_amount)
        strength = self._hand_strength(game, player_index)
        in_position = game.dealer_index == player_index

        if self._should_value_raise(strength, legal, in_position):
            return self._raise_bucket(strength, legal)

        if self._should_bluff(game, player_index, legal, strength, in_position):
            return self._raise_bucket(0.55, legal)

        if call_amount == 0 and Action.CHECK in legal:
            return Action.CHECK

        if strength + (0.08 if in_position else 0.0) >= pot_odds + self.config.call_tolerance:
            if Action.CALL in legal:
                return Action.CALL

        if Action.FOLD in legal:
            return Action.FOLD
        return self._safe_action(legal)

    def _hand_strength(self, game: TexasHoldemGame, player_index: int) -> float:
        player = game.players[player_index]
        ranks = sorted((int(card.rank) for card in player.hole_cards), reverse=True)
        suited = len({card.suit for card in player.hole_cards}) == 1

        if game.phase == GamePhase.PREFLOP:
            pair_bonus = 0.35 if ranks[0] == ranks[1] else 0.0
            high_card = (ranks[0] + ranks[1]) / 28.0
            suited_bonus = 0.08 if suited else 0.0
            connector_bonus = 0.06 if abs(ranks[0] - ranks[1]) <= 1 else 0.0
            premium_bonus = 0.08 if ranks[0] in PREMIUM_RANKS and ranks[1] >= 10 else 0.0
            return min(1.0, 0.18 + high_card * 0.45 + pair_bonus + suited_bonus + connector_bonus + premium_bonus)

        evaluation = HandEvaluator.evaluate([*player.hole_cards, *game.community_cards])
        category_strength = int(evaluation.category) / int(HandCategory.ROYAL_FLUSH)
        kicker_strength = sum(evaluation.tiebreakers[:2]) / 28.0 * 0.12
        return min(1.0, category_strength + kicker_strength)

    def _should_value_raise(
        self,
        strength: float,
        legal: list[Action],
        in_position: bool,
    ) -> bool:
        if not any(action in legal for action in (Action.SMALL_RAISE, Action.MEDIUM_RAISE, Action.LARGE_RAISE)):
            return False
        threshold = 0.67 - (0.05 if in_position else 0.0)
        return strength >= threshold and self.rng.random() < self.config.aggression

    def _should_bluff(
        self,
        game: TexasHoldemGame,
        player_index: int,
        legal: list[Action],
        strength: float,
        in_position: bool,
    ) -> bool:
        if strength > 0.48 or not any(action in legal for action in (Action.SMALL_RAISE, Action.MEDIUM_RAISE)):
            return False
        opponents = [p for i, p in enumerate(game.players) if i != player_index and not p.folded]
        weak_opponents = sum(1 for opponent in opponents if opponent.current_bet == 0 or opponent.acted)
        context = BluffContext(
            board_cards=game.community_cards,
            in_position=in_position,
            opponent_weakness=min(1.0, weak_opponents / max(1, len(opponents))),
            stack_pressure=game.pot / max(1, game.players[player_index].stack + game.pot),
            base_frequency=self.config.bluff_frequency,
        )
        return self.bluff_strategy.should_bluff(context)

    @staticmethod
    def _raise_bucket(strength: float, legal: list[Action]) -> Action:
        if strength > 0.82 and Action.LARGE_RAISE in legal:
            return Action.LARGE_RAISE
        if strength > 0.63 and Action.MEDIUM_RAISE in legal:
            return Action.MEDIUM_RAISE
        if Action.SMALL_RAISE in legal:
            return Action.SMALL_RAISE
        return RuleBasedAgent._safe_action(legal)

    @staticmethod
    def _safe_action(legal: list[Action]) -> Action:
        for action in (Action.CHECK, Action.CALL, Action.FOLD, Action.ALL_IN):
            if action in legal:
                return action
        return legal[0]

