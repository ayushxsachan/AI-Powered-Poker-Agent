"""SQLite persistence for sessions and hand records."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from poker_ai.analytics.statistics import HandRecord


SCHEMA = """
CREATE TABLE IF NOT EXISTS hand_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_number INTEGER NOT NULL,
    player TEXT NOT NULL,
    profit INTEGER NOT NULL,
    won INTEGER NOT NULL,
    category TEXT,
    bluff_attempted INTEGER NOT NULL,
    bluff_success INTEGER NOT NULL,
    elo_before REAL,
    elo_after REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class SQLiteStore:
    def __init__(self, path: str | Path = "poker_ai/logs/poker_ai.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(SCHEMA)
        self.connection.commit()

    def insert_hand_record(self, record: HandRecord) -> None:
        self.connection.execute(
            """
            INSERT INTO hand_records (
                hand_number, player, profit, won, category, bluff_attempted,
                bluff_success, elo_before, elo_after
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.hand_number,
                record.player,
                record.profit,
                int(record.won),
                record.category,
                int(record.bluff_attempted),
                int(record.bluff_success),
                record.elo_before,
                record.elo_after,
            ),
        )
        self.connection.commit()

    def insert_many(self, records: Iterable[HandRecord]) -> None:
        for record in records:
            self.insert_hand_record(record)

    def close(self) -> None:
        self.connection.close()

