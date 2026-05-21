"""Gymnasium-compatible Texas Hold'em environment."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from poker_ai.engine.betting import Action
from poker_ai.engine.cards import Card
from poker_ai.engine.game_manager import GamePhase, TexasHoldemGame

try:  # pragma: no cover - exercised when Gymnasium is installed
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - lightweight fallback for core tests
    class _FallbackEnv:
        metadata: dict[str, Any] = {}

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            return None

    class _Discrete:
        def __init__(self, n: int) -> None:
            self.n = n

        def sample(self) -> int:
            return random.randrange(self.n)

    class _Box:
        def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: Any) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Spaces:
        Box = _Box
        Discrete = _Discrete

    class _Gym:
        Env = _FallbackEnv

    gym = _Gym()
    spaces = _Spaces()


OBSERVATION_SIZE = 64


class PokerEnv(gym.Env):
    """Two-player Hold'em environment for RL experiments.

    The default opponent uses a compact stochastic policy, keeping this module
    independent from the higher-level agent package and safe to import in tests.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        starting_stack: int = 1_000,
        small_blind: int = 5,
        big_blind: int = 10,
        max_steps: int = 200,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.rng = random.Random(seed)
        self.hero_index = 0
        self.max_steps = max_steps
        self.game = TexasHoldemGame(
            ["hero", "villain"],
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            rng=self.rng,
        )
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self._initial_hero_stack = starting_stack
        self._previous_hero_stack = starting_stack
        self._steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)
        self._steps = 0
        self.game.reset_hand()
        self._initial_hero_stack = self.game.players[self.hero_index].stack
        self._previous_hero_stack = self._initial_hero_stack
        self._play_until_hero_or_terminal()
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._steps += 1
        reward = 0.0
        illegal = False

        if self.game.hand_over:
            return self._observation(), 0.0, True, False, self._info()

        legal = self.game.legal_actions(self.hero_index)
        chosen = Action(action)
        if chosen not in legal:
            illegal = True
            chosen = self._safe_fallback_action(legal)
            reward -= 1.0

        before = self.game.players[self.hero_index].stack
        self.game.step(chosen)
        self._play_until_hero_or_terminal()
        after = self.game.players[self.hero_index].stack

        reward += (after - before) / max(1, self.game.big_blind)
        terminated = self.game.hand_over
        truncated = self._steps >= self.max_steps

        if terminated:
            reward += (after - self._initial_hero_stack) / max(1, self.game.big_blind)
            reward += self._terminal_shaping(chosen)

        self._previous_hero_stack = after
        info = self._info()
        info["illegal_action"] = illegal
        return self._observation(), float(reward), terminated, truncated, info

    def render(self) -> str:
        state = self.game.public_state(reveal_hole_cards=True)
        players = ", ".join(
            f"{player['name']} stack={player['stack']} bet={player['current_bet']}"
            for player in state["players"]
        )
        return (
            f"{state['phase']} pot={state['pot']} board={state['community_cards']} "
            f"current={state['current_player_index']} | {players}"
        )

    def legal_action_mask(self) -> np.ndarray:
        return np.asarray(self.game.action_mask(self.hero_index), dtype=np.int8)

    def _play_until_hero_or_terminal(self) -> None:
        while not self.game.hand_over and self.game.current_player_index != self.hero_index:
            legal = self.game.legal_actions()
            self.game.step(self._opponent_action(legal))

    def _opponent_action(self, legal: list[Action]) -> Action:
        if Action.CHECK in legal:
            weights = [
                (Action.CHECK, 0.65),
                (Action.SMALL_RAISE, 0.15),
                (Action.MEDIUM_RAISE, 0.08),
                (Action.ALL_IN, 0.02),
            ]
        else:
            weights = [
                (Action.FOLD, 0.25),
                (Action.CALL, 0.6),
                (Action.SMALL_RAISE, 0.1),
                (Action.ALL_IN, 0.05),
            ]
        choices = [(action, weight) for action, weight in weights if action in legal]
        if not choices:
            return self._safe_fallback_action(legal)
        actions, weights_only = zip(*choices)
        return self.rng.choices(actions, weights=weights_only, k=1)[0]

    @staticmethod
    def _safe_fallback_action(legal: list[Action]) -> Action:
        for action in (Action.CHECK, Action.CALL, Action.FOLD, Action.ALL_IN):
            if action in legal:
                return action
        if not legal:
            raise RuntimeError("No legal actions available")
        return legal[0]

    def _observation(self) -> np.ndarray:
        values: list[float] = []
        hero = self.game.players[self.hero_index]
        villain = self.game.players[1 - self.hero_index]
        max_stack = max(1, hero.stack + villain.stack + self.game.pot)

        for card in self._pad_cards(hero.hole_cards, 2):
            values.extend(self._encode_card(card))
        for card in self._pad_cards(self.game.community_cards, 5):
            values.extend(self._encode_card(card))

        values.extend(
            [
                self.game.pot / max_stack,
                hero.stack / max_stack,
                villain.stack / max_stack,
                hero.current_bet / max_stack,
                villain.current_bet / max_stack,
                self.game.current_bet / max_stack,
                float(self.game.dealer_index == self.hero_index),
                float(hero.folded),
                float(villain.folded),
                float(hero.all_in),
                float(villain.all_in),
            ]
        )
        values.extend(self._phase_one_hot())
        values.extend(float(item) for item in self.game.action_mask(self.hero_index))

        vector = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        vector[: min(len(values), OBSERVATION_SIZE)] = values[:OBSERVATION_SIZE]
        return vector

    @staticmethod
    def _pad_cards(cards: list[Card], target: int) -> list[Card | None]:
        return [*cards[:target], *([None] * max(0, target - len(cards)))]

    @staticmethod
    def _encode_card(card: Card | None) -> list[float]:
        if card is None:
            return [0.0, 0.0]
        suit_index = ["c", "d", "h", "s"].index(card.suit.value) + 1
        return [int(card.rank) / 14.0, suit_index / 4.0]

    def _phase_one_hot(self) -> list[float]:
        phases = [GamePhase.PREFLOP, GamePhase.FLOP, GamePhase.TURN, GamePhase.RIVER]
        return [float(self.game.phase == phase) for phase in phases]

    def _terminal_shaping(self, last_action: Action) -> float:
        result = self.game.last_result
        if result is None:
            return 0.0
        won = self.hero_index in result.winners
        if last_action == Action.FOLD and not won:
            return 0.1
        if last_action == Action.ALL_IN and not won:
            return -0.5
        return 0.25 if won else -0.25

    def _info(self) -> dict[str, Any]:
        return {
            "phase": self.game.phase.value,
            "pot": self.game.pot,
            "legal_actions": [action.label for action in self.game.legal_actions(self.hero_index)],
            "action_mask": self.game.action_mask(self.hero_index),
            "result": self.game.public_state(reveal_hole_cards=True)["last_result"],
        }

