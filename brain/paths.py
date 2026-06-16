"""
brain/paths.py — central path + sys.path wiring for the SAARTHI live system.

Importing this module makes the offline training packages importable from the
live `brain` package:
  • Pipeline B simulator (env/, policies/, rewards/, …)  → reused by engine.py
  • Pipeline A fatigue (fatigue_monitor.py)              → reused by fatigue.py
  • Context risk model (context_risk_api.py)             → reused by risk.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAINING = PROJECT_ROOT / "training"
PIPELINE_A = TRAINING / "pipeline_a_fatigue"
PIPELINE_B = TRAINING / "pipeline_b_simulator"
CONTEXT_RISK = TRAINING / "context_risk"

DATA = PROJECT_ROOT / "data"
DASHBOARD = PROJECT_ROOT / "dashboard"
PHONE = PROJECT_ROOT / "phone"

PROFILES_JSON = DATA / "profiles.json"
TELEMETRY_CSV = DATA / "telemetry.csv"
EVENTS_DB = DATA / "events.db"

# Make the offline packages importable.
for _p in (PIPELINE_B, PIPELINE_A, CONTEXT_RISK):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
