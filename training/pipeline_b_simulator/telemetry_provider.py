"""
telemetry_provider.py  —  Pipeline B / Dev 1 (Simulation & Environment Lead)

Telemetry Replay Parser.

Reads a `telemetry.csv` file containing synchronised excavator metrics
(joystick inputs, engine speed, machine speed, tilt, reversing status) — a CSV
structure derived from the FlywheelAI excavator control dataset — and replays it
one row per simulator tick. When the end of the data is reached it loops back to
the start so an RL agent can train for arbitrarily many steps.

Expected CSV columns (extra columns are preserved and passed through untouched):
    timestamp        float   seconds since session start
    left_joystick    float   [-1, 1]  boom / arm command
    right_joystick   float   [-1, 1]  swing / bucket command
    engine_rpm       float   engine speed
    machine_speed    float   m/s ground speed (>= 0)
    tilt_angle       float   degrees of chassis tilt (abs)
    is_reversing     int     0 = forward / idle, 1 = reversing

If the file is missing, a small synthetic-but-realistic session is generated so
the simulator still runs end to end. The synthetic generator is clearly marked.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

# Canonical schema the rest of Pipeline B relies on.
REQUIRED_COLUMNS: List[str] = [
    "timestamp",
    "left_joystick",
    "right_joystick",
    "engine_rpm",
    "machine_speed",
    "tilt_angle",
    "is_reversing",
]

# Anchor data paths to the project root (…/tata_tech) so this works whether run
# from the package dir, the repo root, or imported by the live system.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TELEMETRY_PATH = str(PROJECT_ROOT / "data" / "telemetry.csv")


class TelemetryProvider:
    """Replays excavator telemetry one row at a time, looping at the end.

    Usage:
        tp = TelemetryProvider("data/telemetry.csv")
        tp.reset()
        row = tp.step()          # dict of the current tick's metrics
        # ... or iterate ...
        for row in tp:           # infinite loop — break when you want to stop
            ...
    """

    def __init__(
        self,
        csv_path: Optional[str] = DEFAULT_TELEMETRY_PATH,
        loop: bool = True,
    ) -> None:
        self.csv_path = csv_path
        self.loop = loop
        self._cursor = 0

        if csv_path and os.path.exists(csv_path):
            self.df = self._load_and_validate(csv_path)
            self.source = csv_path
        else:
            self.df = self._synthesize_session()
            self.source = "synthetic"

        self.n_rows = len(self.df)
        if self.n_rows == 0:
            raise ValueError("Telemetry source produced zero rows.")

    # ── loading ────────────────────────────────────────────────────────
    @staticmethod
    def _load_and_validate(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"telemetry.csv is missing required columns: {missing}. "
                f"Expected at least: {REQUIRED_COLUMNS}"
            )
        # Coerce types defensively; replay must never blow up on a stray cell.
        for col in REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
        df["is_reversing"] = (df["is_reversing"] > 0.5).astype(int)
        df["machine_speed"] = df["machine_speed"].clip(lower=0.0)
        df["tilt_angle"] = df["tilt_angle"].abs()
        return df

    @staticmethod
    def _synthesize_session(n: int = 600, seed: int = 7) -> pd.DataFrame:
        """Fallback session shaped like real excavator control data.

        NOTE: This is a derived stand-in used only when no telemetry.csv is
        present. Replace with FlywheelAI-derived data for real training.
        """
        rng = np.random.default_rng(seed)
        t = np.arange(n)
        # A dig-cycle-ish rhythm: swing, dig, lift, travel, reverse-to-dump.
        phase = (t % 120) / 120.0
        machine_speed = np.clip(1.5 * np.sin(2 * np.pi * phase) ** 2, 0, None)
        engine_rpm = 1200 + 600 * np.abs(np.sin(2 * np.pi * phase)) + rng.normal(0, 30, n)
        left_joy = 0.6 * np.sin(2 * np.pi * phase) + rng.normal(0, 0.05, n)
        right_joy = 0.6 * np.cos(2 * np.pi * phase) + rng.normal(0, 0.05, n)
        # Occasional slope traversal: tilt rises during travel segments.
        tilt = np.abs(8 * np.sin(2 * np.pi * (phase + 0.25)) + rng.normal(0, 1.5, n))
        # A couple of scripted steep-slope events for the hard-rule path.
        tilt[150:165] += 25
        tilt[400:410] += 30
        is_reversing = ((phase > 0.8) & (machine_speed > 0.2)).astype(int)

        return pd.DataFrame(
            {
                "timestamp": t.astype(float),
                "left_joystick": np.clip(left_joy, -1, 1),
                "right_joystick": np.clip(right_joy, -1, 1),
                "engine_rpm": engine_rpm,
                "machine_speed": machine_speed,
                "tilt_angle": tilt,
                "is_reversing": is_reversing,
            }
        )

    # ── replay API ─────────────────────────────────────────────────────
    def reset(self) -> Dict[str, float]:
        """Rewind to the first row and return it."""
        self._cursor = 0
        return self._row(self._cursor)

    def step(self) -> Dict[str, float]:
        """Return the current row, then advance the cursor (looping)."""
        row = self._row(self._cursor)
        self._cursor += 1
        if self._cursor >= self.n_rows:
            self._cursor = 0 if self.loop else self.n_rows - 1
        return row

    def _row(self, idx: int) -> Dict[str, float]:
        record = self.df.iloc[idx].to_dict()
        record["is_reversing"] = int(record["is_reversing"])
        return record

    def __iter__(self) -> Iterator[Dict[str, float]]:
        self.reset()
        while True:
            yield self.step()
            if not self.loop and self._cursor == self.n_rows - 1:
                yield self._row(self._cursor)
                break

    def __len__(self) -> int:
        return self.n_rows


if __name__ == "__main__":
    tp = TelemetryProvider()
    print(f"Telemetry source: {tp.source}  ({len(tp)} rows)")
    tp.reset()
    for i in range(5):
        print(f"tick {i}: {tp.step()}")
