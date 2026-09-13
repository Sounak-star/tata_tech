"""
tests/test_posture_mode.py — the face -> posture handover, at Brain level.

The behaviours that matter are the ones that are dangerous if wrong:
  * a lost face must NOT leave the pipeline reporting the last good reading
  * posture mode requires a BODY, not just a missing face (an empty cab must
    never be scored as an operator)
  * posture may warn earlier, but may not stop the machine on its own
  * the mode must not flap on a single dropped frame

Run:  python -m tests.test_posture_mode
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_TMP = tempfile.mkdtemp()
os.environ.setdefault("SAARTHI_TEMPLATES", str(Path(_TMP) / "t.enc"))
os.environ.setdefault("SAARTHI_FACEID_KEY", str(Path(_TMP) / "k.key"))
os.environ.setdefault("SAARTHI_SUPERVISORS", str(Path(_TMP) / "sup.json"))

from types import SimpleNamespace                                   # noqa: E402

from brain.engine import (                                          # noqa: E402
    FATIGUE_CRITICAL, HybridDecisionEngine, POSTURE_WARN,
)
from brain.fatigue import FEATURES, synth_window                    # noqa: E402
from brain.pipeline import (                                        # noqa: E402
    POSTURE_ENTER_TICKS, POSTURE_EXIT_TICKS, Brain,
)
from brain.pose import L_EAR, L_SHOULDER, NOSE, R_EAR, R_SHOULDER   # noqa: E402
from brain.pose import metrics_from_landmarks                       # noqa: E402
from brain.posture_fatigue import build_baseline                    # noqa: E402
from brain.profiles import ProfileStore                             # noqa: E402
from brain.reason import build_reason_card                          # noqa: E402


def landmarks(*, head_y=0.40, shoulder_y=0.70, shoulder_half=0.16, centre=0.50,
              vis=0.95):
    pts = [SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]

    def put(i, x, y, v):
        pts[i] = SimpleNamespace(x=float(x), y=float(y), visibility=float(v))

    put(L_SHOULDER, centre - shoulder_half, shoulder_y, vis)
    put(R_SHOULDER, centre + shoulder_half, shoulder_y, vis)
    put(L_EAR, centre - 0.05, head_y, vis)
    put(R_EAR, centre + 0.05, head_y, vis)
    put(NOSE, centre, head_y + 0.03, vis)
    return pts


ALERT_POSE = metrics_from_landmarks(landmarks())
DROOPED_POSE = metrics_from_landmarks(landmarks(head_y=0.64, shoulder_y=0.78))
NO_BODY = metrics_from_landmarks([SimpleNamespace(x=0, y=0, visibility=0.0)
                                  for _ in range(33)])

POSTURE_BASELINE = build_baseline(
    [metrics_from_landmarks(landmarks()) for _ in range(40)]).as_dict()


def make_brain(with_posture_baseline: bool = True) -> Brain:
    store = ProfileStore()
    calib = {"calibrated": True, "ear_baseline": 0.30, "perclos_baseline": 0.05,
             "blink_rate_baseline": 14.0,
             "baseline": {f: {"mean": 1.0, "std": 0.5} for f in FEATURES}}
    if with_posture_baseline:
        calib["posture"] = POSTURE_BASELINE
    store.profiles = {"op_1": {
        "id": "op_1", "name": "Test", "role": "Excavator", "hearing": "normal",
        "color_vision": "normal", "language": "en", "experience": "expert",
        "calibration": calib}}
    store._active = "op_1"
    brain = Brain(store=store)
    brain.switch_operator("op_1")
    return brain


def signal(*, features=None, pose=None, speed=1.5, zone=0, tilt=3.0):
    return {"features": features, "pose": pose, "drowsiness": 0.0, "zone": zone,
            "machine_speed": speed, "tilt_angle": tilt, "is_reversing": 0,
            "responded": None}


def alert_features() -> dict:
    return synth_window(0.02)


# ── 1. the handover ──────────────────────────────────────────────────────
def test_face_loss_with_a_body_switches_to_posture():
    brain = make_brain()
    for _ in range(5):
        brain.tick(signal(features=alert_features(), pose=ALERT_POSE))
    assert not brain.posture_mode, "switched while the face was fine"

    for _ in range(POSTURE_ENTER_TICKS):
        out = brain.tick(signal(features=None, pose=ALERT_POSE))
    assert brain.posture_mode, "never fell back to posture"
    assert out["fatigue"]["source"] == "posture", out["fatigue"].get("source")
    print(f"ok  handover: posture mode after {POSTURE_ENTER_TICKS} faceless ticks")


def test_face_returning_hands_control_back():
    brain = make_brain()
    for _ in range(POSTURE_ENTER_TICKS):
        brain.tick(signal(features=None, pose=ALERT_POSE))
    assert brain.posture_mode

    for _ in range(POSTURE_EXIT_TICKS):
        out = brain.tick(signal(features=alert_features(), pose=ALERT_POSE))
    assert not brain.posture_mode, "stayed on posture with the face back"
    assert out["fatigue"].get("source") != "posture"
    print("ok  handback: face returns, PERCLOS takes over again")


def test_mode_does_not_flap_on_dropped_frames():
    """A cab vibrates and the face flickers. One dropped window must not toggle
    the mode, or the operator sees a strobing banner."""
    brain = make_brain()
    flips = 0
    was = brain.posture_mode
    for i in range(120):
        feats = None if i % 7 == 0 else alert_features()      # occasional dropout
        brain.tick(signal(features=feats, pose=ALERT_POSE))
        if brain.posture_mode != was:
            flips += 1
            was = brain.posture_mode
    assert flips == 0, f"mode toggled {flips} times on intermittent dropouts"
    print("ok  hysteresis: 120 ticks with 1-in-7 dropouts, zero mode flips")


# ── 2. an empty seat is not an occluded operator ─────────────────────────
def test_empty_seat_never_enters_posture_mode():
    brain = make_brain()
    for _ in range(POSTURE_ENTER_TICKS * 3):
        out = brain.tick(signal(features=None, pose=NO_BODY))
    assert not brain.posture_mode, "scored the posture of an empty seat"
    print("ok  empty seat: no face and no body -> posture mode stays off")


def test_body_leaving_exits_posture_mode():
    brain = make_brain()
    for _ in range(POSTURE_ENTER_TICKS):
        brain.tick(signal(features=None, pose=ALERT_POSE))
    assert brain.posture_mode

    for _ in range(POSTURE_ENTER_TICKS):
        brain.tick(signal(features=None, pose=NO_BODY))
    assert not brain.posture_mode, "kept monitoring after the operator left"
    assert brain.posture_reason == "seat appears empty"
    print("ok  departure: operator leaves mid-occlusion, monitoring stands down")


# ── 3. the stale-reading bug this feature exists to fix ──────────────────
def test_a_lost_face_does_not_keep_reporting_the_old_reading():
    """The original failure: features froze at the last good window, so the cab
    reported a healthy ALERT indefinitely while the operator nodded off."""
    brain = make_brain()
    for _ in range(5):
        brain.tick(signal(features=alert_features(), pose=ALERT_POSE))

    out = None
    for _ in range(POSTURE_ENTER_TICKS + 10):
        out = brain.tick(signal(features=None, pose=DROOPED_POSE))

    assert brain.posture_mode
    assert out["fatigue"]["p_at_risk"] > 0.3, \
        f"drooping operator still scored {out['fatigue']['p_at_risk']}"
    assert out["fatigue"]["source"] == "posture"
    print(f"ok  no stale reading: hidden face + slumped body -> "
          f"p={out['fatigue']['p_at_risk']:.2f}, not a frozen ALERT")


def test_no_posture_baseline_reports_no_signal_rather_than_guessing():
    brain = make_brain(with_posture_baseline=False)
    out = None
    for _ in range(POSTURE_ENTER_TICKS + 5):
        out = brain.tick(signal(features=None, pose=DROOPED_POSE))
    assert brain.posture_mode
    assert out["fatigue"]["p_at_risk"] == 0.0
    assert out["fatigue"]["decision"] == "NO SIGNAL"
    print("ok  unenrolled: posture mode says NO SIGNAL instead of inventing one")


# ── 4. posture may warn, but may not stop the machine ────────────────────
def test_posture_cannot_trigger_a_level_3_stop():
    eng = HybridDecisionEngine()
    critical = FATIGUE_CRITICAL + 0.10

    for _ in range(60):                                  # 10 s at 6 Hz
        face = eng.decide(fatigue_p=critical, zone=0, tilt=3.0, machine_speed=1.5,
                          is_reversing=0, posture_mode=False)
    assert face["level"] == 3, "face mode should still hard-stop on eyes closed"

    eng2 = HybridDecisionEngine()
    for _ in range(60):
        pose = eng2.decide(fatigue_p=critical, zone=0, tilt=3.0, machine_speed=1.5,
                           is_reversing=0, posture_mode=True)
    assert pose["level"] <= 2, \
        f"posture alone reached level {pose['level']} ({pose['tier']})"
    print(f"ok  cap: same score gives L{face['level']} on face, "
          f"L{pose['level']} on posture")


def test_hard_rules_that_never_needed_a_face_still_fire():
    """Tilt and blind-spot do not depend on the operator's eyes, so posture mode
    must not weaken them."""
    eng = HybridDecisionEngine()
    tilt = eng.decide(fatigue_p=0.0, zone=0, tilt=30.0, machine_speed=1.5,
                      is_reversing=0, posture_mode=True)
    assert tilt["level"] == 3 and "Tilt" in tilt["reason"], tilt

    eng2 = HybridDecisionEngine()
    zone = eng2.decide(fatigue_p=0.0, zone=2, tilt=3.0, machine_speed=1.5,
                       is_reversing=1, posture_mode=True)
    assert zone["level"] == 3 and "RED zone" in zone["reason"], zone
    print("ok  hard rules: tilt and blind-spot still reach L3 in posture mode")


def test_posture_escalates_earlier_than_the_policy_would():
    eng = HybridDecisionEngine()
    quiet = eng.decide(fatigue_p=POSTURE_WARN + 0.05, zone=0, tilt=3.0,
                       machine_speed=1.5, is_reversing=0, posture_mode=False)
    loud = eng.decide(fatigue_p=POSTURE_WARN + 0.05, zone=0, tilt=3.0,
                      machine_speed=1.5, is_reversing=0, posture_mode=True)
    assert loud["level"] >= 2, f"posture mode did not escalate ({loud})"
    assert loud["level"] >= quiet["level"], "posture mode alerted later, not sooner"
    print(f"ok  earlier warning: L{quiet['level']} face vs L{loud['level']} posture "
          f"at p={POSTURE_WARN + 0.05:.2f}")


# ── 5. the operator is told which evidence they are looking at ───────────
def test_reason_card_never_presents_posture_as_shap():
    posture_fat = {"p_at_risk": 0.62, "source": "posture",
                   "reasons": [{"feature": "head_drop", "value": 3.1,
                                "direction": "up",
                                "text": "head dropped toward chest (3.1 SD above your normal)"}]}
    card = build_reason_card(2, "ppo-policy", "", posture_fat, "GREEN",
                             {"status": "ok", "tilt_angle": 2.0})
    assert card["source"] == "posture", card["source"]
    assert "posture" in card["title"].lower()
    assert "Face not visible" in card["detail"], card["detail"]

    face_fat = {"p_at_risk": 0.62, "reasons": [
        {"feature": "perclos", "value": 0.4, "direction": "up"}]}
    face_card = build_reason_card(2, "ppo-policy", "", face_fat, "GREEN",
                                  {"status": "ok", "tilt_angle": 2.0})
    assert face_card["source"] == "shap"
    print(f"ok  honesty: \"{card['title']}\" — {card['detail'][:52]}...")


# ── 6. sunglasses vs a dust mask ─────────────────────────────────────────
def test_covered_eyes_switch_even_though_the_face_is_still_tracked():
    """The case a naive "no face" trigger misses entirely. MediaPipe keeps
    fitting a mesh through sunglasses, so features keep arriving and has_face
    stays True — but EAR and PERCLOS now describe eyelids nobody can see."""
    brain = make_brain()
    for _ in range(5):
        brain.tick(signal(features=alert_features(), pose=ALERT_POSE))
    assert not brain.posture_mode

    out = None
    for _ in range(POSTURE_ENTER_TICKS + 2):
        sig = signal(features=alert_features(), pose=ALERT_POSE)
        sig["eyes_covered"] = True                     # landmarker still happy
        sig["occlusion_reason"] = "eyes appear covered (sunglasses or visor)"
        out = brain.tick(sig)

    assert brain.posture_mode,         "sunglasses kept the system in face mode on fictional eye data"
    assert out["fatigue"]["source"] == "posture"
    assert out["posture"]["eyes_covered"] is True
    assert "covered" in out["posture"]["reason"], out["posture"]["reason"]
    print(f"ok  sunglasses: face still tracked, but switched — "
          f"\"{out['posture']['reason']}\"")


def test_a_mask_over_the_mouth_stays_in_face_mode():
    """The screenshot case. The mouth is gone but the eyes are visible, so
    PERCLOS is intact — and PERCLOS beats posture. Switching would be a
    DOWNGRADE, so staying put is the correct behaviour, not a bug."""
    brain = make_brain()
    out = None
    for _ in range(POSTURE_ENTER_TICKS * 2):
        sig = signal(features=alert_features(), pose=ALERT_POSE)
        sig["eyes_covered"] = False                    # mask, eyes clear
        out = brain.tick(sig)

    assert not brain.posture_mode,         "a dust mask downgraded us off PERCLOS onto weaker posture evidence"
    assert out["fatigue"]["source"] != "posture"
    print("ok  dust mask: eyes visible, so PERCLOS keeps control (correct)")


def test_uncovering_the_eyes_hands_control_back():
    brain = make_brain()
    for _ in range(POSTURE_ENTER_TICKS + 2):
        sig = signal(features=alert_features(), pose=ALERT_POSE)
        sig["eyes_covered"] = True
        brain.tick(sig)
    assert brain.posture_mode

    for _ in range(POSTURE_EXIT_TICKS + 2):
        out = brain.tick(signal(features=alert_features(), pose=ALERT_POSE))
    assert not brain.posture_mode, "glasses came off but posture mode stuck"
    assert out["fatigue"]["source"] != "posture"
    print("ok  glasses off: PERCLOS takes back over")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    raise SystemExit(1 if failed else 0)
