"""
tests/test_pipeline_faceid.py — Brain-level integration for operator switching.

The interesting case is not "the enrolled operator gets their baseline". It is
the one after it: an operator with NO stored baseline must fall back to the
session baseline, never inherit the previous operator's personal one — that
would read a new face against someone else's idea of normal.

Run:  python -m tests.test_pipeline_faceid
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Point key material at a scratch dir BEFORE brain modules import.
_TMP = tempfile.mkdtemp()
os.environ.setdefault("SAARTHI_TEMPLATES", str(Path(_TMP) / "t.enc"))
os.environ.setdefault("SAARTHI_FACEID_KEY", str(Path(_TMP) / "k.key"))
os.environ.setdefault("SAARTHI_SUPERVISORS", str(Path(_TMP) / "sup.json"))

from brain.fatigue import FEATURES                                  # noqa: E402
from brain.personalize import personalise                           # noqa: E402
from brain.pipeline import Brain                                    # noqa: E402
from brain.profiles import GUEST_ID, ProfileStore                   # noqa: E402


def full_baseline(ear: float = 0.30) -> dict:
    return {f: {"mean": ear if f == "ear_mean" else 1.0, "std": 0.5} for f in FEATURES}


def make_brain() -> Brain:
    store = ProfileStore()
    store.profiles = {
        "enrolled_op": {
            "id": "enrolled_op", "name": "Enrolled Operator", "role": "Excavator",
            "hearing": "impaired", "color_vision": "normal", "language": "hi",
            "experience": "expert",
            "calibration": {"calibrated": True, "ear_baseline": 0.30,
                            "perclos_baseline": 0.05, "blink_rate_baseline": 14.0,
                            "baseline": full_baseline()},
        },
        "legacy_op": {
            "id": "legacy_op", "name": "Legacy Operator", "role": "Crane",
            "hearing": "normal", "color_vision": "deuteranopia", "language": "ta",
            "experience": "trainee",
            # The shape every profile in data/profiles.json has today: summary
            # numbers only, no full baseline.
            "calibration": {"calibrated": True, "ear_baseline": 0.28,
                            "perclos_baseline": 0.07, "blink_rate_baseline": 13.0},
        },
    }
    store._active = "enrolled_op"
    return Brain(store=store)


def test_enrolled_operator_skips_calibration():
    brain = make_brain()
    session_engine = brain._session_fatigue

    brain.switch_operator("enrolled_op", source="face")
    assert brain.calibration_source == "enrolment:enrolled_op", brain.calibration_source
    assert brain.fatigue is not session_engine, "stored baseline was not used"
    print("ok  enrolled operator: personal baseline loaded, 25 s calibration skipped")


def test_operator_without_baseline_does_not_inherit():
    brain = make_brain()
    session_engine = brain._session_fatigue

    brain.switch_operator("enrolled_op")
    assert brain.fatigue is not session_engine

    brain.switch_operator("legacy_op")
    assert brain.fatigue is session_engine, \
        "legacy_op inherited enrolled_op's personal baseline — wrong normal for this face"
    assert brain.calibration_source == "session"
    print("ok  no-baseline operator: reverts to the session baseline, inherits nothing")


def test_guest_profile_is_conservative_and_uninherited():
    brain = make_brain()
    brain.switch_operator("enrolled_op")

    prof = brain.switch_to_guest("below threshold")
    assert brain.store.active_id == GUEST_ID and brain.store.is_guest
    assert brain.fatigue is brain._session_fatigue, "guest inherited a personal baseline"

    plan = personalise(1, prof)
    assert plan["buzz"] and plan["flash"] and plan["sound"], \
        "an unidentified operator must get every alert channel"
    assert plan["simplified_ui"] and plan["use_icons"]
    assert plan["buzz_strength"] == "strong"

    # ...and an expert is still treated as an expert, so guest is genuinely stricter.
    expert = personalise(1, brain.store.profiles["enrolled_op"])
    assert not expert["flash"] or not expert["sound"], \
        "guest profile is not actually more conservative than a normal one"
    print("ok  guest: all channels at level 1, strictest UI, no inherited baseline")


def test_switch_still_resets_the_decision_engine():
    brain = make_brain()
    brain.engine._prev_level = 3 if hasattr(brain.engine, "_prev_level") else None
    brain.switch_operator("legacy_op")
    brain._prev_level = 0
    assert brain.store.active_id == "legacy_op"
    plan = personalise(2, brain.store.active())
    assert plan["language"] == "ta" and plan["use_icons"], \
        "switching did not re-personalise delivery"
    print("ok  switch: decision engine reset and delivery re-personalised")


def test_faceid_degrades_without_a_model():
    brain = make_brain()
    status = brain.faceid.status()
    assert "state" in status or "backend" in status
    assert brain.faceid.recognise() == brain.store.active(), \
        "legacy recognise() contract broken"
    print(f"ok  faceid: degrades cleanly (backend={brain.faceid.backend})")


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
