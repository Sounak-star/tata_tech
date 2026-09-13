"""
brain/wage_log.py — Append-only digital wage log for tracking operator working hours.
"""

import sqlite3
import time
from typing import Optional

from .paths import EVENTS_DB


class WageLog:
    """Local, append-only SQLite log of operator shift durations."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.path = str(db_path or EVENTS_DB)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS wage_log (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   operator TEXT,
                   start_ts REAL,
                   end_ts REAL,
                   duration REAL
               )"""
        )
        self._conn.commit()

    def log_shift(self, operator: str, start_ts: float, end_ts: float) -> None:
        """Logs a shift. Ignores shifts that are practically zero seconds."""
        duration = end_ts - start_ts
        if duration <= 1.0:
            return  # skip micro-glitches
        
        self._conn.execute(
            "INSERT INTO wage_log (operator, start_ts, end_ts, duration)"
            " VALUES (?, ?, ?, ?)",
            (operator, start_ts, end_ts, duration),
        )
        self._conn.commit()

    def get_total_seconds(self, operator: str) -> float:
        """Returns the total accumulated working seconds for an operator."""
        cur = self._conn.execute(
            "SELECT SUM(duration) FROM wage_log WHERE operator = ?",
            (operator,)
        )
        row = cur.fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    def get_all_wages(self) -> dict[str, float]:
        """Returns a dictionary mapping operator IDs to their total accumulated seconds."""
        cur = self._conn.execute(
            "SELECT operator, SUM(duration) FROM wage_log GROUP BY operator HAVING SUM(duration) > 0"
        )
        return {row[0]: float(row[1]) for row in cur.fetchall()}

    def format_total_hours(self, total_seconds: float) -> str:
        """Formats seconds into a human-readable 'X hrs Y mins Z secs' string."""
        total_seconds = int(total_seconds)
        hrs = total_seconds // 3600
        mins = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        parts = []
        if hrs > 0:
            parts.append(f"{hrs} hrs")
        if hrs > 0 or mins > 0:
            parts.append(f"{mins} mins")
        parts.append(f"{secs} secs")
        return " ".join(parts)
