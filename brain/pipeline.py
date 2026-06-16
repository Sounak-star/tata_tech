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

from typing import Optional

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
                 event_log: Optional[EventLog] = None) -> None:
        self.store = store or ProfileStore()
        self.faceid = FaceID(self.store)
        self.fatigue = FatigueEngine()
        self.persons = PersonDetector()
        self.engine = HybridDecisionEngine()
        self.log = event_log or EventLog()
        self.context = _load_context_risk()
        self._tick = 0

    def switch_operator(self, operator_id: str) -> dict:
        prof = self.store.switch(operator_id)
        self.engine.reset()
        self.fatigue.reset()
        return prof

    def tick(self, signal: dict) -> dict:
        self._tick += 1
        profile = self.faceid.recognise()

        # Q2 — fatigue (real XGBoost model; features from webcam or synthesised).
        features = signal.get("features") or synth_window(signal.get("drowsiness", 0.0))
        fatigue = self.fatigue.update(features)

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

        if level >= 2:
            self.log.log(profile["id"], level, tier, card, risk["score"])

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
            "context_risk": self.context,
        }


def _load_context_risk() -> dict:
    """Dev 3 context-risk badge for the supervisor view (best-effort)."""
    try:
        from context_risk_api import get_context_risk

        r = get_context_risk(machine_type="Excavator", task_type="Excavation",
                             time_of_day="Morning", weather_condition="Clear/Unknown")
        return {"score": r["risk_score"], "bucket": r["risk_bucket"], "color": r["risk_color"]}
    except Exception:
        return {"score": None, "bucket": "n/a", "color": "#888888"}
