"""
brain/pipeline.py — the live per-tick orchestration (Pipeline C).

About 10x/second: signals in → the brain answers its four questions → the Hybrid
Decision Engine picks a level → the personaliser picks delivery → outputs fire
with a Reason Card → the event is logged locally. Video is processed and thrown
away — it never exists on disk.

A "tick" consumes a raw signal dict (from a webcam pipeline or the demo source):
    {
      "features": {...12 fatigue features...} | None,
      "drowsiness": float,          # used only to synth features when none given
      "zone": int (0/1/2),
      "machine_speed": float, "tilt_angle": float, "is_reversing": int,
      "responded": bool | None,     # did the operator heed the previous alert?
    }
"""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional

from .engine import HybridDecisionEngine
from .fatigue import FatigueEngine, synth_window
from .persons import PersonDetector
from .personalize import personalise
from .profiles import ProfileStore, FaceID
from .reason import EventLog, build_reason_card
from .risk import live_risk_score
from .tilt import assess_tilt


class Brain:
    """The full in-cab copilot: fuses Pipelines A, B and the context model."""

    def __init__(self, store: Optional[ProfileStore] = None,
                 event_log: Optional[EventLog] = None,
                 calibration_rows: Optional[List[Dict]] = None) -> None:
        self.store = store or ProfileStore()
        self.faceid = FaceID(self.store)
        self.fatigue = FatigueEngine(calibration_rows)
        self.persons = PersonDetector()
        self.engine = HybridDecisionEngine()
        self.log = event_log or EventLog()
        self.context = {"score": 35.0, "bucket": "Medium", "color": "#FF9800", "weather": "Clear/Unknown", "time": "Morning"}
        self._tick = 0
        # Timeline throttle: track previous alert level to detect onset transitions only.
        # An entry is written ONCE per rising edge (level > _prev_level), not every tick.
        self._prev_level = 0

    def switch_operator(self, operator_id: str) -> dict:
        prof = self.store.switch(operator_id)
        self.engine.reset()
        self.fatigue.reset()
        return prof

    def tick(self, signal: dict) -> dict:
        self._tick += 1
        profile = self.faceid.recognise()

        # Q2 — fatigue (real XGBoost model; features from webcam or synthesised).
        raw_feats = signal.get("features")
        feat_src = "camera" if raw_feats is not None else "demo-synth"
        features = raw_feats if raw_feats is not None else synth_window(signal.get("drowsiness", 0.0))
        fatigue = self.fatigue.update(features)

        # ── Warm-up Guard ────────────────────────────────────────────────────
        # Suppress spurious fatigue spikes when the camera trailing buffer is empty.
        is_warming_up = raw_feats is not None and raw_feats.get("is_warming_up", 0.0) > 0.5
        if is_warming_up:
            fatigue["p_at_risk"] = 0.0
            fatigue["p_raw"] = 0.0
            fatigue["decision"] = "INITIALIZING"
            fatigue["severity"] = "calibrating sensor"
        # ─────────────────────────────────────────────────────────────────────

        # ── Per-tick diagnostic log ──────────────────────────────────────────
        _nan_ct = sum(1 for v in features.values() if isinstance(v, float) and math.isnan(v))
        print(
            f"[TICK {self._tick:04d}] src={feat_src} phase={signal.get('phase')} "
            f"drown={signal.get('drowsiness', 0.0):.3f} NaNs={_nan_ct} "
            f"p_raw={fatigue.get('p_raw', fatigue['p_at_risk']):.4f} "
            f"p_smooth={fatigue['p_at_risk']:.4f} "
            f"{fatigue['decision']}/{fatigue['severity']}",
            file=sys.stderr, flush=True,
        )
        if raw_feats is not None:
            _fstr = " ".join(f"{k}={v:.4f}" for k, v in raw_feats.items())
            print(f"  RAW: {_fstr}", file=sys.stderr, flush=True)
        else:
            print(f"  RAW: NONE → synth(drown={signal.get('drowsiness', 0.0):.3f})",
                  file=sys.stderr, flush=True)
        # ────────────────────────────────────────────────────────────────────

        # Q3 — blind-spot zone.
        zinfo = self.persons.detect(simulated_zone=signal.get("zone", 0))
        zone, zone_name = zinfo["zone"], zinfo["zone_name"]

        # Q4 — machine tilt.
        speed = float(signal.get("machine_speed", 0.0))
        reversing = int(signal.get("is_reversing", 0))
        tinfo = assess_tilt(float(signal.get("tilt_angle", 0.0)), speed)

        # Feed back the previous alert's outcome → adaptive thresholds.
        if signal.get("responded") is not None and self.engine._last_action != 0:
            self.engine.feedback(bool(signal["responded"]))

        # Hybrid Decision Engine.
        decision = self.engine.decide(
            fatigue_p=fatigue["p_at_risk"], zone=zone,
            tilt=tinfo["tilt_angle"], machine_speed=speed,
            is_reversing=reversing, experience=profile.get("experience", "expert"),
        )
        level, tier = decision["level"], decision["tier"]

        # Transparent live risk score + personalised delivery + reason card.
        risk = live_risk_score(fatigue["p_at_risk"], zone, tinfo["tilt_angle"])
        delivery = personalise(level, profile)
        card = build_reason_card(level, tier, decision["reason"], fatigue, zone_name, tinfo)

        # Calculate dynamic context risk using the actual XGBoost model
        context_risk = _calculate_dynamic_context_risk(
            role=profile.get("role", "Excavator Operator"),
            phase=signal.get("phase", "calm"),
            fatigue_p=fatigue["p_at_risk"],
            zone=zone,
            tilt_angle=tinfo["tilt_angle"],
            tick=self._tick
        )

        # Throttle: log only on upward level transition (onset), not every tick.
        if level >= 2 and level > self._prev_level:
            self.log.log(profile["id"], level, tier, card, risk["score"])
        self._prev_level = level

        return {
            "tick": self._tick,
            "operator": {
                "id": profile["id"], "name": profile["name"], "role": profile["role"],
                "hearing": profile["hearing"], "color_vision": profile["color_vision"],
                "language": profile["language"], "experience": profile["experience"],
            },
            "fatigue": {
                "p": fatigue["p_at_risk"], "decision": fatigue["decision"],
                "severity": fatigue["severity"], "backend": self.fatigue.backend,
            },
            "zone": zone_name,
            "tilt": tinfo,
            "machine": {"speed": round(speed, 2), "reversing": bool(reversing)},
            "risk_score": risk,
            "alert": {"level": level, "label": decision["label"], "tier": tier,
                      "reason": decision["reason"], "delivery": delivery},
            "reason_card": card,
            "context_risk": context_risk,
        }


def _calculate_dynamic_context_risk(
    role: str,
    phase: str,
    fatigue_p: float,
    zone: int,
    tilt_angle: float,
    tick: int
) -> dict:
    try:
        from context_risk_api import get_context_risk  # type: ignore[import]

        # 1. Map operator role to machine_type
        role_lower = role.lower()
        if "crane" in role_lower:
            machine_type = "Crane"
        elif "bulldozer" in role_lower:
            machine_type = "Bulldozer"
        elif "dumper" in role_lower or "truck" in role_lower:
            machine_type = "Dump Truck"
        elif "loader" in role_lower:
            machine_type = "Loader"
        else:
            machine_type = "Excavator"

        # 2. Map demo phase to task_type and weather
        if phase == "blind_spot":
            task_type = "Loading/Unloading"
            weather_condition = "Wind"
        elif phase == "tilt":
            task_type = "Excavation"
            weather_condition = "Rain"
        elif phase in ("switch_priya", "switch_ravi"):
            task_type = "Idling/Parked"
            weather_condition = "Clear/Unknown"
        elif phase == "trainee_run":
            task_type = "Setup/Positioning"
            weather_condition = "Clear/Unknown"
        else:
            task_type = "Operating"
            weather_condition = "Clear/Unknown"

        # 3. Vary time_of_day across a simulated cycle
        tick_mod = tick % 120
        if tick_mod < 30:
            time_of_day = "Morning"
        elif tick_mod < 60:
            time_of_day = "Afternoon"
        elif tick_mod < 90:
            time_of_day = "Evening"
        else:
            time_of_day = "Night"

        # 4. Predict baseline risk via XGBoost model
        r = get_context_risk(
            machine_type=machine_type,
            task_type=task_type,
            time_of_day=time_of_day,
            weather_condition=weather_condition
        )
        score = r["risk_score"]
        bucket = r["risk_bucket"]
        color = r["risk_color"]

        # 5. Escalate based on real-time sensor detections
        if fatigue_p > 0.7 or zone >= 2 or tilt_angle > 20.0:
            if bucket == "Low":
                bucket = "Medium"
                color = "#FF9800"
                score = max(score, 35.0)
            elif bucket == "Medium":
                bucket = "High"
                color = "#F44336"
                score = max(score, 60.0)
            elif bucket == "High":
                bucket = "Critical"
                color = "#9C27B0"
                score = max(score, 80.0)
        elif fatigue_p < 0.2 and zone == 0 and tilt_angle < 10.0:
            if bucket == "High":
                bucket = "Medium"
                color = "#FF9800"
                score = min(score, 45.0)
            elif bucket == "Medium":
                bucket = "Low"
                color = "#4CAF50"
                score = min(score, 20.0)

        return {
            "score": score,
            "bucket": bucket,
            "color": color,
            "weather": weather_condition,
            "time": time_of_day
        }
    except Exception as exc:
        print(f"[Context Risk Fallback] Error: {exc}")
        return {
            "score": 35.0,
            "bucket": "Medium",
            "color": "#FF9800",
            "weather": "Clear/Unknown",
            "time": "Morning"
        }
