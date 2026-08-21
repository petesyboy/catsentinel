"""Logs every sighting (both "our cat" and "stranger", including borderline scores)
to SQLite plus a saved snapshot image, so thresholds can be reviewed/tuned later and
a real labeled dataset accumulates from actual footage over time.

`reviewed_label` starts NULL on every row; scripts/review_events.py lets you
confirm or correct the pipeline's own verdict against each snapshot by eye, and
that human-confirmed label -- not the pipeline's guess -- is what gets fed back
into retraining (see scripts/train_classifier.py).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sightings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    verdict TEXT NOT NULL,           -- 'mine' | 'stranger' (the pipeline's own guess)
    probability REAL NOT NULL,       -- classifier's P(this is your cat)
    detector_confidence REAL NOT NULL,
    deterrent_triggered INTEGER NOT NULL,
    snapshot_path TEXT
);
"""

# Columns added after the initial release -- applied with ALTER TABLE if missing,
# so existing events.db files pick them up without losing any logged history.
_MIGRATIONS = [
    ("reviewed_label", "TEXT"),   # NULL (not yet reviewed) | 'mine' | 'stranger' | 'skip'
    ("reviewed_at", "TEXT"),
]


@dataclass
class Sighting:
    id: int
    timestamp: str
    verdict: str
    probability: float
    detector_confidence: float
    snapshot_path: str | None


class EventLog:
    def __init__(self, db_path: str, events_dir: str, save_snapshots: bool = True):
        self._events_dir = Path(events_dir)
        self._events_dir.mkdir(parents=True, exist_ok=True)
        self._save_snapshots = save_snapshots

        self._conn = sqlite3.connect(db_path)
        self._conn.execute(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(sightings)")}
        for column, column_type in _MIGRATIONS:
            if column not in existing:
                self._conn.execute(f"ALTER TABLE sightings ADD COLUMN {column} {column_type}")

    def record(
        self,
        frame: np.ndarray,
        verdict: str,
        probability: float,
        detector_confidence: float,
        deterrent_triggered: bool,
    ) -> None:
        timestamp = datetime.now()
        snapshot_path = None

        if self._save_snapshots:
            filename = f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}_{verdict}.jpg"
            snapshot_path = str(self._events_dir / filename)
            cv2.imwrite(snapshot_path, frame)

        self._conn.execute(
            "INSERT INTO sightings "
            "(timestamp, verdict, probability, detector_confidence, deterrent_triggered, snapshot_path) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                timestamp.isoformat(),
                verdict,
                probability,
                detector_confidence,
                int(deterrent_triggered),
                snapshot_path,
            ),
        )
        self._conn.commit()

    def unreviewed(self, limit: int | None = None) -> list[Sighting]:
        """Sightings with a snapshot that haven't been human-reviewed yet, oldest first."""
        query = (
            "SELECT id, timestamp, verdict, probability, detector_confidence, snapshot_path "
            "FROM sightings WHERE reviewed_label IS NULL AND snapshot_path IS NOT NULL "
            "ORDER BY timestamp ASC"
        )
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        rows = self._conn.execute(query).fetchall()
        return [Sighting(*row) for row in rows]

    def set_reviewed_label(self, sighting_id: int, label: str) -> None:
        """label: 'mine' | 'stranger' | 'skip'."""
        self._conn.execute(
            "UPDATE sightings SET reviewed_label = ?, reviewed_at = ? WHERE id = ?",
            (label, datetime.now().isoformat(), sighting_id),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
