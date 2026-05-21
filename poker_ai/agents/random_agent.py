"""Random baseline agent."""

from __future__ import annotations

import random

from poker_ai.engine.betting import Action
from poker_ai.engine.game_manager import TexasHoldemGame


class RandomAgent:
    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def act(self, game: TexasHoldemGame, player_index: int) -> Action:
        legal = game.legal_actions(player_index)
        if not legal:
            raise RuntimeError("No legal actions available")
        return self.rng.choice(legal)

