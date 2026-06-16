"""
brain/reason.py — Reason Cards (Block 4) + the local event log (Block 5).

Every alert shows WHY in plain words. SHAP supplies the fatigue reasons (from
Pipeline A); the rules explain themselves. Nothing here touches video — only
numbers and timestamps, audit-ready by design.
"""

from __future__ import annotations

import sqlite3
import time
from typing import List, Optional

from .paths import EVENTS_DB

_FEATURE_LABELS = {
    "perclos": "eyes nearly shut",
    "ear_mean": "average eye openness",
    "ear_min": "lowest eye openness",
    "blink_rate": "blink rate",
    "mar_mean": "mouth opening (yawn)",
    "mar_max": "peak mouth opening",
    "is_yawn": "yawn detected",
    "pitch_mean": "head nodding forward",
    "pitch_std": "head bobbing",
}


def build_reason_card(level: int, tier: str, hard_reason: str,
                      fatigue: dict, zone_name: str, tilt: dict) -> Optional[dict]:
    """Compose the Reason Card for the dominant cause of this alert."""
    if level <= 0:
        return None

    # Hard-rule emergencies explain themselves.
    if tier == "hard-rule":
        return {
            "title": "EMERGENCY — hard rule",
            "detail": hard_reason,
            "factors": [],
            "source": "rule",
        }

    # Otherwise pick the dominant soft cause.
    p = fatigue.get("p_at_risk", 0.0)
    if zone_name == "RED" or zone_name == "AMBER":
        return {
            "title": f"Person in {zone_name} zone",
            "detail": f"Worker detected in the {zone_name.lower()} blind-spot zone.",
            "factors": [{"feature": "blind_spot_zone", "value": zone_name, "direction": "high"}],
            "source": "rule",
        }
    if tilt.get("status") in ("warning", "critical"):
        return {
            "title": "Slope / tilt warning",
            "detail": f"Chassis tilt {tilt.get('tilt_angle')}° while moving.",
            "factors": [{"feature": "tilt_angle", "value": tilt.get("tilt_angle"), "direction": "high"}],
            "source": "rule",
        }

    # Fatigue reason card with SHAP factors.
    factors = []
    for r in fatigue.get("reasons", [])[:3]:
        label = _FEATURE_LABELS.get(r["feature"], r["feature"])
        factors.append({**r, "label": label})
    detail = f"Fatigue probability {int(p * 100)}%"
    if factors:
        top = factors[0]
        detail += f" — driven by {top['label']} ({top['value']})"
    return {"title": "Fatigue detected", "detail": detail, "factors": factors, "source": "shap"}


class EventLog:
    """Local, append-only SQLite log of alerts (timestamps + reasons, no video)."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.path = str(db_path or EVENTS_DB)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS events (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   ts REAL, operator TEXT, level INTEGER, tier TEXT,
                   title TEXT, detail TEXT, risk_score REAL
               )"""
        )
        self._conn.commit()

    def log(self, operator: str, level: int, tier: str,
            card: Optional[dict], risk_score: float) -> None:
        title = card["title"] if card else ""
        detail = card["detail"] if card else ""
        self._conn.execute(
            "INSERT INTO events (ts, operator, level, tier, title, detail, risk_score)"
            " VALUES (?,?,?,?,?,?,?)",
            (time.time(), operator, level, tier, title, detail, risk_score),
        )
        self._conn.commit()

    def recent(self, limit: int = 20) -> List[dict]:
        cur = self._conn.execute(
            "SELECT ts, operator, level, tier, title, detail, risk_score"
            " FROM events ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        cols = ["ts", "operator", "level", "tier", "title", "detail", "risk_score"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
