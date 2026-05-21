"""Self-play tournaments and ELO ranking."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

from poker_ai.analytics.statistics import HandRecord, SessionStatistics
from poker_ai.engine.betting import Action
from poker_ai.engine.game_manager import TexasHoldemGame


class AgentProtocol(Protocol):
    def act(self, game: TexasHoldemGame, player_index: int) -> Action:
        ...


@dataclass
class EloTable:
    ratings: dict[str, float] = field(default_factory=dict)
    k_factor: float = 24.0

    def rating(self, name: str) -> float:
        return self.ratings.setdefault(name, 1500.0)

    def update(self, winner: str, loser: str, draw: bool = False) -> None:
        winner_rating = self.rating(winner)
        loser_rating = self.rating(loser)
        expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
        score_winner = 0.5 if draw else 1.0
        score_loser = 0.5 if draw else 0.0
        self.ratings[winner] = winner_rating + self.k_factor * (score_winner - expected_winner)
        self.ratings[loser] = loser_rating + self.k_factor * (score_loser - (1 - expected_winner))


@dataclass
class SelfPlayTournament:
    agents: dict[str, AgentProtocol]
    starting_stack: int = 1_000
    small_blind: int = 5
    big_blind: int = 10
    rng: random.Random = field(default_factory=random.Random)
    elo: EloTable = field(default_factory=EloTable)
    stats: SessionStatistics = field(default_factory=SessionStatistics)

    def run_heads_up(self, agent_a: str, agent_b: str, hands: int = 100) -> SessionStatistics:
        game = TexasHoldemGame(
            [agent_a, agent_b],
            starting_stack=self.starting_stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            rng=self.rng,
        )
        for _ in range(hands):
            before = {player.name: player.stack for player in game.players}
            game.reset_hand()
            while not game.hand_over:
                index = game.current_player_index
                name = game.players[index].name
                game.step(self.agents[name].act(game, index))
            self._record_hand(game, before)
            self._rebuy_if_needed(game)
        return self.stats

    def _record_hand(self, game: TexasHoldemGame, before: dict[str, int]) -> None:
        result = game.last_result
        if result is None:
            return
        winners = {game.players[index].name for index in result.winners}
        for index, player in enumerate(game.players):
            name = player.name
            evaluation = result.evaluations.get(index)
            self.stats.add(
                HandRecord(
                    hand_number=game.hand_number,
                    player=name,
                    profit=player.stack - before[name],
                    won=name in winners,
                    category=evaluation.name if evaluation else None,
                    elo_before=self.elo.rating(name),
                )
            )

        if len(game.players) == 2 and len(winners) == 1:
            winner = next(iter(winners))
            loser = next(player.name for player in game.players if player.name != winner)
            self.elo.update(winner, loser)

    def _rebuy_if_needed(self, game: TexasHoldemGame) -> None:
        for player in game.players:
            if player.stack < self.big_blind:
                player.stack = self.starting_stack

