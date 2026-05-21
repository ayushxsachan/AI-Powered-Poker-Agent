"""No-limit Texas Hold'em game manager."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from poker_ai.engine.betting import Action, BettingDecision, PlayerState, RAISE_ACTIONS, raise_size
from poker_ai.engine.cards import Card
from poker_ai.engine.deck import Deck
from poker_ai.engine.hand_evaluator import HandEvaluation, HandEvaluator


class GamePhase(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"
    HAND_OVER = "hand_over"


@dataclass
class HandResult:
    winners: list[int]
    payouts: dict[int, int]
    evaluations: dict[int, HandEvaluation] = field(default_factory=dict)
    pot: int = 0
    summary: str = ""


class TexasHoldemGame:
    """Manage a single-table no-limit Texas Hold'em hand.

    The game supports 2-9 players, blinds, betting rounds, all-in side pots,
    showdown evaluation, and action masks for RL agents.
    """

    def __init__(
        self,
        player_names: list[str],
        starting_stack: int = 1_000,
        small_blind: int = 5,
        big_blind: int = 10,
        rng: random.Random | None = None,
    ) -> None:
        if not 2 <= len(player_names) <= 9:
            raise ValueError("Texas Hold'em requires 2 to 9 players")
        if small_blind <= 0 or big_blind <= 0 or small_blind > big_blind:
            raise ValueError("Blinds must be positive and small_blind <= big_blind")

        self.rng = rng or random.Random()
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.players = [PlayerState(name, starting_stack) for name in player_names]
        self.dealer_index = -1
        self.phase = GamePhase.HAND_OVER
        self.deck = Deck(rng=self.rng)
        self.community_cards: list[Card] = []
        self.current_bet = 0
        self.current_player_index = 0
        self.last_result: HandResult | None = None
        self.hand_number = 0

    @property
    def pot(self) -> int:
        return sum(player.committed for player in self.players)

    @property
    def hand_over(self) -> bool:
        return self.phase == GamePhase.HAND_OVER

    def reset_hand(self) -> dict[str, Any]:
        """Start a new hand and return the public state."""

        if sum(player.stack > 0 for player in self.players) < 2:
            raise RuntimeError("At least two players need chips to start a hand")

        self.hand_number += 1
        self.last_result = None
        self.community_cards = []
        self.deck = Deck(rng=self.rng)
        self.phase = GamePhase.PREFLOP
        self.current_bet = 0
        self.dealer_index = self._next_index(self.dealer_index)

        for player in self.players:
            player.reset_for_hand()

        for _ in range(2):
            for player in self.players:
                if player.stack > 0:
                    player.hole_cards.append(self.deck.draw_one())

        small_blind_index, big_blind_index = self._blind_indices()
        self._post_blind(small_blind_index, self.small_blind)
        self._post_blind(big_blind_index, self.big_blind)
        self.current_bet = max(player.current_bet for player in self.players)
        self.current_player_index = self._first_preflop_actor(big_blind_index)
        self._skip_to_next_actionable_or_advance()
        return self.public_state()

    def legal_actions(self, player_index: int | None = None) -> list[Action]:
        index = self.current_player_index if player_index is None else player_index
        player = self.players[index]
        if self.hand_over or player.folded or player.all_in or player.stack <= 0:
            return []

        call_amount = max(0, self.current_bet - player.current_bet)
        actions: list[Action] = []
        if call_amount > 0:
            actions.extend([Action.FOLD, Action.CALL])
        else:
            actions.append(Action.CHECK)

        if player.stack > call_amount:
            actions.extend([Action.SMALL_RAISE, Action.MEDIUM_RAISE, Action.LARGE_RAISE])
        if player.stack > 0:
            actions.append(Action.ALL_IN)
        return actions

    def action_mask(self, player_index: int | None = None) -> list[int]:
        legal = set(self.legal_actions(player_index))
        return [1 if action in legal else 0 for action in Action]

    def step(self, decision: Action | BettingDecision | int) -> dict[str, Any]:
        if self.hand_over:
            raise RuntimeError("Hand is already over")

        if isinstance(decision, BettingDecision):
            action = decision.action
            custom_amount = decision.amount
        else:
            action = Action(decision)
            custom_amount = None

        legal = self.legal_actions(self.current_player_index)
        if action not in legal:
            raise ValueError(
                f"Illegal action {action.label} for {self.players[self.current_player_index].name}; "
                f"legal actions are {[item.label for item in legal]}"
            )

        player = self.players[self.current_player_index]
        call_amount = max(0, self.current_bet - player.current_bet)
        old_table_bet = self.current_bet

        if action == Action.FOLD:
            player.folded = True
            player.acted = True
        elif action == Action.CHECK:
            if call_amount:
                raise ValueError("Cannot check facing a bet")
            player.acted = True
        elif action == Action.CALL:
            player.commit(call_amount)
            player.acted = True
        elif action in RAISE_ACTIONS:
            extra = custom_amount if custom_amount is not None else raise_size(action, self.pot, self.big_blind)
            player.commit(call_amount + max(0, extra))
            player.acted = True
            self._handle_bet_increase(old_table_bet, player)
        elif action == Action.ALL_IN:
            player.commit(player.stack)
            player.acted = True
            self._handle_bet_increase(old_table_bet, player)
        else:
            raise ValueError(f"Unhandled action {action}")

        self._advance_after_action()
        return self.public_state()

    def public_state(self, reveal_hole_cards: bool = False) -> dict[str, Any]:
        return {
            "hand_number": self.hand_number,
            "phase": self.phase.value,
            "dealer_index": self.dealer_index,
            "current_player_index": self.current_player_index,
            "community_cards": [str(card) for card in self.community_cards],
            "pot": self.pot,
            "current_bet": self.current_bet,
            "action_mask": self.action_mask() if not self.hand_over else [0] * len(Action),
            "players": [
                {
                    "name": player.name,
                    "stack": player.stack,
                    "current_bet": player.current_bet,
                    "committed": player.committed,
                    "folded": player.folded,
                    "all_in": player.all_in,
                    "acted": player.acted,
                    "hole_cards": [str(card) for card in player.hole_cards]
                    if reveal_hole_cards or self.hand_over
                    else [],
                }
                for player in self.players
            ],
            "last_result": self._serialize_result(self.last_result),
        }

    def _handle_bet_increase(self, old_table_bet: int, bettor: PlayerState) -> None:
        if bettor.current_bet <= old_table_bet:
            return

        self.current_bet = bettor.current_bet
        for other in self.players:
            if other is not bettor and not other.folded and not other.all_in:
                other.acted = False

    def _advance_after_action(self) -> None:
        if self._contesting_count() == 1:
            self._award_uncontested()
            return

        if self._should_run_out_board():
            self._deal_remaining_board()
            self._showdown()
            return

        if self._betting_round_complete():
            self._advance_street()
            return

        self.current_player_index = self._next_actionable_from(self.current_player_index)

    def _skip_to_next_actionable_or_advance(self) -> None:
        if self.hand_over:
            return
        if self._contesting_count() == 1:
            self._award_uncontested()
            return
        if self._should_run_out_board():
            self._deal_remaining_board()
            self._showdown()
            return
        if self._betting_round_complete():
            self._advance_street()
            return
        if not self._is_actionable(self.current_player_index):
            self.current_player_index = self._next_actionable_from(self.current_player_index)

    def _advance_street(self) -> None:
        if self.phase == GamePhase.PREFLOP:
            self.community_cards.extend(self.deck.draw(3))
            self.phase = GamePhase.FLOP
        elif self.phase == GamePhase.FLOP:
            self.community_cards.extend(self.deck.draw(1))
            self.phase = GamePhase.TURN
        elif self.phase == GamePhase.TURN:
            self.community_cards.extend(self.deck.draw(1))
            self.phase = GamePhase.RIVER
        elif self.phase == GamePhase.RIVER:
            self._showdown()
            return

        self.current_bet = 0
        for player in self.players:
            player.reset_for_street()

        if self._should_run_out_board():
            self._deal_remaining_board()
            self._showdown()
            return

        self.current_player_index = self._first_postflop_actor()
        self._skip_to_next_actionable_or_advance()

    def _betting_round_complete(self) -> bool:
        for player in self.players:
            if player.folded or player.all_in:
                continue
            if not player.acted:
                return False
            if player.current_bet != self.current_bet:
                return False
        return True

    def _should_run_out_board(self) -> bool:
        return self._contesting_count() > 1 and self._actionable_count() <= 1

    def _deal_remaining_board(self) -> None:
        while len(self.community_cards) < 5:
            self.community_cards.extend(self.deck.draw(1 if self.community_cards else 3))
        self.phase = GamePhase.SHOWDOWN

    def _showdown(self) -> None:
        contenders = [index for index, player in enumerate(self.players) if not player.folded]
        payouts = {index: 0 for index in range(len(self.players))}
        all_evaluations: dict[int, HandEvaluation] = {}
        winners_overall: set[int] = set()

        for amount, eligible in self._side_pots():
            winners, evaluations = HandEvaluator.winners(
                eligible,
                {index: self.players[index].hole_cards for index in eligible},
                self.community_cards,
            )
            all_evaluations.update(evaluations)
            share, remainder = divmod(amount, len(winners))
            for winner in winners:
                payouts[winner] += share
                winners_overall.add(winner)
            for winner in winners[:remainder]:
                payouts[winner] += 1

        for index, payout in payouts.items():
            self.players[index].stack += payout

        self.phase = GamePhase.HAND_OVER
        pot = sum(payouts.values())
        self.last_result = HandResult(
            winners=sorted(winners_overall),
            payouts={index: value for index, value in payouts.items() if value},
            evaluations=all_evaluations,
            pot=pot,
            summary=self._result_summary(sorted(winners_overall), payouts, all_evaluations),
        )

    def _award_uncontested(self) -> None:
        winner = next(index for index, player in enumerate(self.players) if not player.folded)
        pot = self.pot
        self.players[winner].stack += pot
        self.phase = GamePhase.HAND_OVER
        self.last_result = HandResult(
            winners=[winner],
            payouts={winner: pot},
            pot=pot,
            summary=f"{self.players[winner].name} wins {pot} uncontested.",
        )

    def _side_pots(self) -> list[tuple[int, list[int]]]:
        commitments = {
            index: player.committed
            for index, player in enumerate(self.players)
            if player.committed > 0
        }
        levels = sorted(set(commitments.values()))
        pots: list[tuple[int, list[int]]] = []
        previous = 0
        for level in levels:
            contributors = [index for index, amount in commitments.items() if amount >= level]
            amount = (level - previous) * len(contributors)
            eligible = [index for index in contributors if not self.players[index].folded]
            if amount > 0 and eligible:
                pots.append((amount, eligible))
            previous = level
        return pots

    def _result_summary(
        self,
        winners: list[int],
        payouts: dict[int, int],
        evaluations: dict[int, HandEvaluation],
    ) -> str:
        parts = []
        for winner in winners:
            hand = evaluations.get(winner)
            hand_text = f" with {hand.name}" if hand else ""
            parts.append(f"{self.players[winner].name} wins {payouts[winner]}{hand_text}")
        return "; ".join(parts)

    def _blind_indices(self) -> tuple[int, int]:
        if len(self.players) == 2:
            return self.dealer_index, self._next_index(self.dealer_index)
        small_blind_index = self._next_index(self.dealer_index)
        return small_blind_index, self._next_index(small_blind_index)

    def _post_blind(self, player_index: int, amount: int) -> None:
        player = self.players[player_index]
        player.commit(amount)
        player.acted = False

    def _first_preflop_actor(self, big_blind_index: int) -> int:
        if len(self.players) == 2:
            return self.dealer_index
        return self._next_index(big_blind_index)

    def _first_postflop_actor(self) -> int:
        return self._next_actionable_from(self.dealer_index)

    def _next_actionable_from(self, index: int) -> int:
        candidate = index
        for _ in range(len(self.players)):
            candidate = self._next_index(candidate)
            if self._is_actionable(candidate):
                return candidate
        return index

    def _next_index(self, index: int) -> int:
        for offset in range(1, len(self.players) + 1):
            candidate = (index + offset) % len(self.players)
            if self.players[candidate].stack > 0 or self.players[candidate].committed > 0:
                return candidate
        return (index + 1) % len(self.players)

    def _is_actionable(self, index: int) -> bool:
        player = self.players[index]
        return not self.hand_over and not player.folded and not player.all_in and player.stack > 0

    def _contesting_count(self) -> int:
        return sum(player.contesting for player in self.players)

    def _actionable_count(self) -> int:
        return sum(self._is_actionable(index) for index in range(len(self.players)))

    @staticmethod
    def _serialize_result(result: HandResult | None) -> dict[str, Any] | None:
        if result is None:
            return None
        return {
            "winners": result.winners,
            "payouts": result.payouts,
            "pot": result.pot,
            "summary": result.summary,
            "evaluations": {
                index: str(evaluation)
                for index, evaluation in result.evaluations.items()
            },
        }

