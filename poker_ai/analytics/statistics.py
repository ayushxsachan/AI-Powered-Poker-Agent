"""Session and training statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None


@dataclass
class HandRecord:
    hand_number: int
    player: str
    profit: int
    won: bool
    category: str | None = None
    bluff_attempted: bool = False
    bluff_success: bool = False
    elo_before: float | None = None
    elo_after: float | None = None


@dataclass
class SessionStatistics:
    records: list[HandRecord] = field(default_factory=list)

    def add(self, record: HandRecord) -> None:
        self.records.append(record)

    @property
    def hands(self) -> int:
        return len(self.records)

    def profit(self, player: str) -> int:
        return sum(record.profit for record in self.records if record.player == player)

    def win_rate(self, player: str) -> float:
        player_records = [record for record in self.records if record.player == player]
        return sum(record.won for record in player_records) / max(1, len(player_records))

    def bluff_success_rate(self, player: str) -> float:
        attempts = [record for record in self.records if record.player == player and record.bluff_attempted]
        return sum(record.bluff_success for record in attempts) / max(1, len(attempts))

    def hand_distribution(self) -> dict[str, int]:
        distribution: dict[str, int] = {}
        for record in self.records:
            if record.category:
                distribution[record.category] = distribution.get(record.category, 0) + 1
        return distribution

    def to_dataframe(self):
        if pd is None:
            raise ImportError("pandas is required to export SessionStatistics as a DataFrame")
        return pd.DataFrame([record.__dict__ for record in self.records])

    def save_csv(self, path: str | Path) -> None:
        self.to_dataframe().to_csv(Path(path), index=False)

    def plot_profit(self, player: str, path: str | Path | None = None) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("matplotlib is required for plotting") from exc

        running = []
        total = 0
        for record in self.records:
            if record.player == player:
                total += record.profit
                running.append(total)
        plt.figure(figsize=(10, 4))
        plt.plot(running)
        plt.title(f"{player} profit")
        plt.xlabel("Hands")
        plt.ylabel("Chips")
        plt.tight_layout()
        if path:
            plt.savefig(Path(path))
        else:
            plt.show()

