"""Opponent modeling metrics common in poker tracking software."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from poker_ai.engine.betting import Action


@dataclass
class OpponentStats:
    hands: int = 0
    vpip: int = 0
    pfr: int = 0
    aggressive_actions: int = 0
    passive_actions: int = 0
    bluff_attempts: int = 0
    bluff_successes: int = 0
    fold_to_raise_opportunities: int = 0
    folds_to_raise: int = 0
    showdowns: int = 0
    wins: int = 0

    @property
    def vpip_rate(self) -> float:
        return self.vpip / self.hands if self.hands else 0.0

    @property
    def pfr_rate(self) -> float:
        return self.pfr / self.hands if self.hands else 0.0

    @property
    def aggression_factor(self) -> float:
        return self.aggressive_actions / max(1, self.passive_actions)

    @property
    def bluff_frequency(self) -> float:
        return self.bluff_attempts / max(1, self.aggressive_actions)

    @property
    def bluff_success_rate(self) -> float:
        return self.bluff_successes / max(1, self.bluff_attempts)

    @property
    def fold_to_raise_rate(self) -> float:
        return self.folds_to_raise / max(1, self.fold_to_raise_opportunities)

    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.showdowns)

    def as_dict(self) -> dict[str, float]:
        return {
            "hands": float(self.hands),
            "vpip": self.vpip_rate,
            "pfr": self.pfr_rate,
            "aggression_factor": self.aggression_factor,
            "bluff_frequency": self.bluff_frequency,
            "bluff_success_rate": self.bluff_success_rate,
            "fold_to_raise": self.fold_to_raise_rate,
            "showdown_win_rate": self.win_rate,
        }

    def to_counts(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}

    @classmethod
    def from_counts(cls, data: dict[str, int | float]) -> "OpponentStats":
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{key: int(data.get(key, 0)) for key in valid_keys})

    def merge(self, other: "OpponentStats") -> None:
        for key in self.__dataclass_fields__:
            setattr(self, key, int(getattr(self, key)) + int(getattr(other, key)))


@dataclass
class OpponentModel:
    players: dict[str, OpponentStats] = field(default_factory=dict)

    def stats_for(self, player_name: str) -> OpponentStats:
        return self.players.setdefault(player_name, OpponentStats())

    def start_hand(self, player_names: list[str]) -> None:
        for name in player_names:
            self.stats_for(name).hands += 1

    def observe_action(
        self,
        player_name: str,
        action: Action,
        *,
        preflop: bool,
        voluntary: bool = True,
        faced_raise: bool = False,
        suspected_bluff: bool = False,
    ) -> None:
        stats = self.stats_for(player_name)
        if voluntary and action in {Action.CALL, Action.SMALL_RAISE, Action.MEDIUM_RAISE, Action.LARGE_RAISE, Action.ALL_IN}:
            stats.vpip += int(preflop)
        if preflop and action in {Action.SMALL_RAISE, Action.MEDIUM_RAISE, Action.LARGE_RAISE, Action.ALL_IN}:
            stats.pfr += 1
        if action in {Action.SMALL_RAISE, Action.MEDIUM_RAISE, Action.LARGE_RAISE, Action.ALL_IN}:
            stats.aggressive_actions += 1
            if suspected_bluff:
                stats.bluff_attempts += 1
        elif action in {Action.CALL, Action.CHECK}:
            stats.passive_actions += 1
        if faced_raise:
            stats.fold_to_raise_opportunities += 1
            if action == Action.FOLD:
                stats.folds_to_raise += 1

    def observe_showdown(
        self,
        player_name: str,
        *,
        won: bool,
        bluff_attempted: bool = False,
    ) -> None:
        stats = self.stats_for(player_name)
        stats.showdowns += 1
        stats.wins += int(won)
        if bluff_attempted and won:
            stats.bluff_successes += 1

    def tendencies(self, player_name: str) -> dict[str, float]:
        return self.stats_for(player_name).as_dict()

    def to_counts_dict(self) -> dict[str, dict[str, int]]:
        return {
            name: stats.to_counts()
            for name, stats in sorted(self.players.items())
        }

    @classmethod
    def from_counts_dict(cls, data: dict[str, dict[str, int | float]]) -> "OpponentModel":
        model = cls()
        for name, counts in data.items():
            model.players[name] = OpponentStats.from_counts(counts)
        return model

    def merge(self, other: "OpponentModel") -> None:
        for name, stats in other.players.items():
            self.stats_for(name).merge(stats)
