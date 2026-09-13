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
import time
from typing import Dict, List, Optional

from .engine import HybridDecisionEngine
from .fatigue import FatigueEngine, synth_window
from .persons import PersonDetector
from .personalize import personalise
from .pose import PoseMetrics
from .posture_fatigue import PostureBaseline, PostureFatigue
from .profiles import ProfileStore, FaceID
from .reason import EventLog, build_reason_card
from .risk import live_risk_score
from .tilt import assess_tilt
from .wage_log import WageLog


# Mode-switch hysteresis, in ticks at server.TICK_HZ (6 Hz). A face flickers
# constantly under vibration and glare; without hysteresis the mode would toggle
# on every dropped frame and the operator would see a strobing banner.
POSTURE_ENTER_TICKS = 6   # ~1 s of no face before falling back to posture
POSTURE_EXIT_TICKS = 6     # ~1 s of steady face before handing back


class Brain:
    """The full in-cab copilot: fuses Pipelines A, B and the context model."""

    def __init__(self, store: Optional[ProfileStore] = None,
                 event_log: Optional[EventLog] = None,
                 calibration_rows: Optional[List[Dict]] = None) -> None:
        self.store = store or ProfileStore()
        self.faceid = FaceID(self.store)
        self.fatigue = FatigueEngine(calibration_rows)
        # The engine this session started with. Restored whenever the active
        # operator has no personal baseline of their own.
        self._session_fatigue = self.fatigue
        self.calibration_source = "session"
        self.wage_log = WageLog()
        self._shift_start = time.time()

        # Posture fallback: used only when the face signal is gone AND pose can
        # still see somebody in the seat.
        self.posture = PostureFatigue(None)
        self.posture_mode = False
        self.posture_reason = ""
        self.occlusion_reason = ""
        self._no_face_ticks = 0
        self._face_ticks = 0
        self.persons = PersonDetector()
        self.engine = HybridDecisionEngine()
        self.log = event_log or EventLog()
        self.context = {"score": 35.0, "bucket": "Medium", "color": "#FF9800", "weather": "Clear/Unknown", "time": "Morning"}
        self._tick = 0
        # Timeline throttle: track previous alert level to detect onset transitions only.
        # An entry is written ONCE per rising edge (level > _prev_level), not every tick.
        self._prev_level = 0

    def _close_shift(self) -> None:
        current_id = self.store.active_id
        if hasattr(self, '_shift_start') and current_id and not self.store.is_guest:
            self.wage_log.log_shift(current_id, self._shift_start, time.time())
        self._shift_start = time.time()

    def switch_operator(self, operator_id: str, *, source: str = "manual") -> dict:
        """Make `operator_id` active and re-personalise everything downstream.

        If the operator was enrolled in-cab, their fatigue baseline is already on
        file — we rebuild the fatigue engine from it and skip calibration
        entirely.  If they have no stored baseline we fall back to the engine
        this session started with; we must NOT keep the previous operator's
        personal baseline, which would silently read the new face against the
        wrong normal.
        """
        self._close_shift()
        prof = self.store.switch(operator_id)
        self.engine.reset()
        self._load_posture_baseline(prof)

        baseline = self.store.baseline_for(operator_id)
        if baseline:
            engine = FatigueEngine.from_baseline(baseline)
            if engine is not None:
                self.fatigue = engine
                self.calibration_source = f"enrolment:{operator_id}"
                print(f"[Brain] {operator_id} ({source}) — personal baseline loaded, "
                      f"calibration skipped.", flush=True)
                return prof

        if self.fatigue is not self._session_fatigue:
            # Coming off someone else's personal baseline — go back to the
            # session default rather than judging this operator by that one.
            self.fatigue = self._session_fatigue
            self.calibration_source = "session"
        self.fatigue.reset()
        return prof

    def _load_posture_baseline(self, profile: dict) -> None:
        """Load this operator's seated-posture baseline, captured at enrolment.

        Without one, PostureFatigue reports NO SIGNAL rather than guessing: what
        counts as "slumped" is meaningless without knowing how this particular
        person sits.
        """
        stored = (profile.get("calibration") or {}).get("posture")
        self.posture = PostureFatigue(PostureBaseline.from_dict(stored or {}))
        self.posture_mode = False
        self._no_face_ticks = self._face_ticks = 0

    def switch_to_guest(self, reason: str = "unidentified") -> dict:
        """Fail-safe profile for an unrecognised operator."""
        self._close_shift()
        prof = self.store.switch_to_guest()
        self.engine.reset()
        self._load_posture_baseline(prof)
        # Ensure FaceID forgets the last operator so it rescans when they return
        if self.faceid.identifier is not None:
            self.faceid.identifier.reset()
        if self.fatigue is not self._session_fatigue:
            self.fatigue = self._session_fatigue
            self.calibration_source = "session"
        self.fatigue.reset()
        print(f"[Brain] operator unidentified ({reason}) — conservative profile.",
              flush=True)
        return prof

    def _update_mode(self, *, face_ok: bool, pose_present: bool) -> None:
        """Switch between face and posture fatigue, with hysteresis.

        Entering posture mode needs BOTH a sustained loss of the face AND pose
        confirming somebody is still sitting there. Without the second condition
        an empty cab would look identical to an occluded operator, and we would
        happily score the posture of a seat.
        """
        if face_ok:
            self._face_ticks += 1
            self._no_face_ticks = 0
        else:
            self._no_face_ticks += 1
            self._face_ticks = 0

        if not self.posture_mode:
            if self._no_face_ticks >= POSTURE_ENTER_TICKS and pose_present:
                self.posture_mode = True
                self.posture.reset()
                self.posture_reason = (self.occlusion_reason
                                       or "face not visible — operator still "
                                          "detected, monitoring posture")
                print(f"[Brain] posture mode ON ({self.posture_reason})", flush=True)
        else:
            if self._face_ticks >= POSTURE_EXIT_TICKS:
                self.posture_mode = False
                self.posture_reason = ""
                print("[Brain] posture mode OFF — face visible again", flush=True)
            elif not pose_present and self._no_face_ticks >= POSTURE_ENTER_TICKS:
                # No face and no body: the seat is empty, not occluded.
                self.posture_mode = False
                self.posture_reason = "seat appears empty"
                if not self.store.is_guest:
                    self.switch_to_guest("seat appears empty")

    def tick(self, signal: dict) -> dict:
        self._tick += 1
        profile = self.faceid.recognise()

        # Q2 — fatigue. Normally the XGBoost model on face features; if the face
        # is hidden (sunglasses, dust mask) but pose can still see the operator,
        # posture takes over. Note `raw_feats` is None once the last face window
        # has expired — live_camera refuses to serve stale features, precisely so
        # this branch can happen instead of reporting an old reading forever.
        raw_feats = signal.get("features")
        pose: Optional[PoseMetrics] = signal.get("pose")
        pose_present = bool(pose is not None and pose.present)

        # A face can be present, tracked, and still useless: sunglasses leave the
        # mesh fitted while EAR and PERCLOS describe eyelids nobody can see. That
        # is not a face signal, however confident the numbers look.
        eyes_covered = bool(signal.get("eyes_covered"))
        self.occlusion_reason = signal.get("occlusion_reason", "") if eyes_covered else ""
        face_ok = raw_feats is not None and not eyes_covered

        self._update_mode(face_ok=face_ok, pose_present=pose_present)

        if self.posture_mode:
            feat_src = "posture"
            features = synth_window(0.0)          # keeps the log shape consistent
            fatigue = self.posture.update(pose)
        else:
            feat_src = "camera" if face_ok else "demo-synth"
            features = raw_feats if face_ok else synth_window(signal.get("drowsiness", 0.0))
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
            posture_mode=self.posture_mode,
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

        # Calculate ongoing shift duration if they are actively working
        ongoing_secs = time.time() - self._shift_start if (not self.store.is_guest and profile["id"] == self.store.active_id) else 0.0
        total_worked = self.wage_log.get_total_seconds(profile["id"]) + ongoing_secs
        verified_hours_str = self.wage_log.format_total_hours(total_worked)

        return {
            "tick": self._tick,
            "operator": {
                "id": profile["id"], "name": profile["name"], "role": profile["role"],
                "hearing": profile["hearing"], "color_vision": profile["color_vision"],
                "language": profile["language"], "experience": profile["experience"],
                "verified_hours": verified_hours_str,
            },
            "fatigue": {
                "p": fatigue["p_at_risk"],
                "p_at_risk": fatigue["p_at_risk"],
                "decision": fatigue["decision"],
                "severity": fatigue["severity"],
                # Which pipeline produced this number. The dashboard must not
                # render a posture estimate the same way it renders PERCLOS.
                "source": fatigue.get("source", "face"),
                "backend": (self.posture.backend if self.posture_mode
                            else self.fatigue.backend),
            },
            "posture": {
                "mode": self.posture_mode,
                "reason": self.posture_reason,
                "eyes_covered": eyes_covered,
                "present": bool(pose is not None and pose.present),
                "ready": self.posture.ready,
                "sigmas": fatigue.get("sigmas", {}),
                "metrics": pose.as_dict() if pose is not None else None,
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
