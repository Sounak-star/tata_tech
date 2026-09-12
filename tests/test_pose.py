"""
tests/test_pose.py — posture measurement and the posture fatigue score.

No camera and no pose model: synthetic landmark sets stand in for an operator
sitting up, nodding off, slumping and swaying.

The claims under test are the ones the design rests on:
  * measurements are normalised, so the same posture scores the same whether the
    operator is tall, close to the camera, or sitting in a raised seat
  * "present but face hidden" is distinguishable from "seat empty"
  * a score is only produced against the operator's OWN baseline

Run:  python -m tests.test_pose
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.pose import (                                        # noqa: E402
    ABSENCE_FRAMES, PRESENCE_FRAMES, L_EAR, L_SHOULDER, NOSE, R_EAR, R_SHOULDER,
    PoseMetrics, PoseTracker, metrics_from_landmarks,
)
from brain.posture_fatigue import (                             # noqa: E402
    PostureBaseline, PostureFatigue, build_baseline,
)


def landmarks(*, shoulder_y=0.70, shoulder_half=0.16, head_y=0.40,
              centre=0.50, head_dx=0.0, roll=0.0, vis=0.95, head_vis=0.95):
    """A seated upper body. Defaults are an alert operator sitting upright."""
    pts = [SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]

    def put(i, x, y, v):
        pts[i] = SimpleNamespace(x=float(x), y=float(y), visibility=float(v))

    put(L_SHOULDER, centre - shoulder_half, shoulder_y - roll, vis)
    put(R_SHOULDER, centre + shoulder_half, shoulder_y + roll, vis)
    put(L_EAR, centre - 0.05 + head_dx, head_y, head_vis)
    put(R_EAR, centre + 0.05 + head_dx, head_y, head_vis)
    put(NOSE, centre + head_dx, head_y + 0.03, head_vis)
    return pts


def empty_seat():
    return [SimpleNamespace(x=0.0, y=0.0, visibility=0.0) for _ in range(33)]


def sample_run(n=30, **kw) -> list[PoseMetrics]:
    return [metrics_from_landmarks(landmarks(**kw)) for _ in range(n)]


# ── 1. measurement is scale- and position-invariant ──────────────────────
def test_measurements_are_normalised_by_shoulder_width():
    """A big operator close to the camera and a small one further away, in the
    same posture, must measure the same. Raw pixels would not."""
    near = metrics_from_landmarks(landmarks(shoulder_half=0.24, head_y=0.34,
                                            shoulder_y=0.76))
    far = metrics_from_landmarks(landmarks(shoulder_half=0.12, head_y=0.51,
                                           shoulder_y=0.72))
    assert near.present and far.present
    assert abs(near.head_height - far.head_height) < 0.02, \
        f"same posture measured differently: {near.head_height} vs {far.head_height}"
    print(f"ok  normalisation: head_height {near.head_height:.2f} vs "
          f"{far.head_height:.2f} across a 2x scale change")


def test_head_drop_reduces_head_height():
    up = metrics_from_landmarks(landmarks(head_y=0.40))
    down = metrics_from_landmarks(landmarks(head_y=0.60))     # chin to chest
    assert down.head_height < up.head_height, "a dropped head must measure lower"
    print(f"ok  head drop: {up.head_height:.2f} -> {down.head_height:.2f}")


def test_shoulder_roll_is_signed_and_bounded():
    flat = metrics_from_landmarks(landmarks(roll=0.0))
    tilted = metrics_from_landmarks(landmarks(roll=0.05))
    assert abs(flat.shoulder_roll) < 1.0
    assert abs(tilted.shoulder_roll) > 5.0 and abs(tilted.shoulder_roll) < 90.0
    print(f"ok  roll: flat {flat.shoulder_roll:.1f}deg, "
          f"tilted {tilted.shoulder_roll:.1f}deg")


def test_missing_head_still_measures_via_shoulders():
    """Sunglasses plus a mask can cost the ears; the shoulders remain the ruler."""
    m = metrics_from_landmarks(landmarks(head_vis=0.1))
    assert m.present, "lost the operator just because the head was unclear"
    assert m.shoulder_width > 0
    print("ok  occlusion: shoulders alone still yield a usable measurement")


def test_no_shoulders_means_no_measurement():
    m = metrics_from_landmarks(landmarks(vis=0.1))
    assert not m.present, "measured posture with no visible shoulders"
    assert m.shoulder_width == 0.0
    print("ok  no ruler: without shoulders, nothing is reported")


# ── 2. present-but-hidden vs empty seat ──────────────────────────────────
def test_presence_is_sticky_and_asymmetric():
    """Quick to notice someone; slow to declare the seat empty. Declaring empty
    wrongly switches monitoring off altogether, so it must be the harder call."""
    trk = PoseTracker.__new__(PoseTracker)      # no model needed for this logic
    trk._present_run = trk._absent_run = 0
    trk._present = False

    for _ in range(PRESENCE_FRAMES):
        trk._update_presence(True)
    assert trk.present, "did not register an operator who is clearly there"

    for _ in range(ABSENCE_FRAMES - 1):
        trk._update_presence(False)
    assert trk.present, "declared the seat empty too eagerly"

    trk._update_presence(False)
    assert not trk.present, "never declared the seat empty"
    assert ABSENCE_FRAMES > PRESENCE_FRAMES, "absence must be harder to conclude"
    print(f"ok  presence: {PRESENCE_FRAMES} frames to appear, "
          f"{ABSENCE_FRAMES} to be declared gone")


def test_empty_seat_is_not_a_hidden_operator():
    assert not metrics_from_landmarks(empty_seat()).present
    assert metrics_from_landmarks(landmarks(head_vis=0.05)).present
    print("ok  discrimination: empty seat and occluded operator are distinct")


# ── 3. scoring is baseline-relative, or it does not score ────────────────
def test_no_baseline_means_no_score():
    """Without this operator's normal, 'slumped' has no meaning. Refuse to guess."""
    pf = PostureFatigue(baseline=None)
    out = pf.update(metrics_from_landmarks(landmarks(head_y=0.65)))
    assert out["p_at_risk"] == 0.0
    assert out["confident"] is False and "baseline" in out["note"]
    assert out["decision"] == "NO SIGNAL"
    print("ok  honesty: no baseline -> NO SIGNAL, not an invented threshold")


def test_baseline_needs_enough_samples():
    assert build_baseline(sample_run(3)) is None, "built a baseline from 3 samples"
    assert build_baseline(sample_run(30)) is not None
    print("ok  baseline: refuses a too-short capture, accepts a full one")


def test_alert_posture_scores_near_zero():
    base = build_baseline(sample_run(40))
    pf = PostureFatigue(base)
    out = [pf.update(m) for m in sample_run(20)][-1]
    assert out["p_at_risk"] < 0.05, f"sitting normally scored {out['p_at_risk']}"
    assert out["decision"] == "ALERT"
    print(f"ok  alert posture: p={out['p_at_risk']:.3f} against own baseline")


def test_head_drop_raises_the_score_and_explains_it():
    base = build_baseline(sample_run(40, head_y=0.40))
    pf = PostureFatigue(base)
    out = [pf.update(m) for m in sample_run(20, head_y=0.62)][-1]

    assert out["p_at_risk"] > 0.3, f"a clear head drop scored only {out['p_at_risk']}"
    assert out["source"] == "posture", "posture score not labelled as such"
    top = out["reasons"][0]
    assert top["feature"] == "head_drop", f"top reason was {top['feature']}"
    assert "normal" in top["text"]
    print(f"ok  head drop: p={out['p_at_risk']:.2f} - \"{top['text']}\"")


def test_slump_is_detected_from_any_of_its_three_signs():
    base = build_baseline(sample_run(40))
    for label, kw in (("shoulders lower", {"shoulder_y": 0.80}),
                      ("leaning forward", {"head_dx": 0.10}),
                      ("shoulders foreshortened", {"shoulder_half": 0.115})):
        pf = PostureFatigue(build_baseline(sample_run(40)) or base)
        out = [pf.update(m) for m in sample_run(20, **kw)][-1]
        assert out["sigmas"]["slump"] > 1.0, \
            f"{label} did not register as a slump ({out['sigmas']})"
    print("ok  slump: detected from lower shoulders, forward lean or narrowing")


def test_sitting_up_straighter_is_not_drowsiness():
    """Only a drop counts. Sitting up more than baseline must not raise the score."""
    base = build_baseline(sample_run(40, head_y=0.45))
    pf = PostureFatigue(base)
    out = [pf.update(m) for m in sample_run(20, head_y=0.32)][-1]
    assert out["p_at_risk"] < 0.05, f"sitting up straight scored {out['p_at_risk']}"
    print("ok  direction: improved posture does not read as fatigue")


def test_sway_needs_a_full_window_then_registers():
    base = build_baseline(sample_run(60))
    pf = PostureFatigue(base)

    rng = np.random.default_rng(4)
    first = pf.update(metrics_from_landmarks(landmarks(centre=0.5)))
    assert first["sigmas"]["sway"] == 0.0, "scored sway before the window filled"

    out = None
    for _ in range(60):
        c = 0.5 + 0.05 * rng.standard_normal()
        out = pf.update(metrics_from_landmarks(landmarks(centre=c)))
    assert out["sigmas"]["sway"] > 1.0, f"sway not detected ({out['sigmas']})"
    print(f"ok  sway: silent until the window fills, then {out['sigmas']['sway']:.1f} SD")


def test_score_is_smoothed_against_single_frame_spikes():
    base = build_baseline(sample_run(40))
    pf = PostureFatigue(base)
    for m in sample_run(10):
        pf.update(m)
    spike = pf.update(metrics_from_landmarks(landmarks(head_y=0.75)))
    assert spike["p_at_risk"] < 0.35, \
        f"one bad frame jumped the score to {spike['p_at_risk']}"
    print(f"ok  smoothing: a single dropped frame only reaches "
          f"p={spike['p_at_risk']:.2f}")


def test_absent_operator_scores_nothing():
    pf = PostureFatigue(build_baseline(sample_run(40)))
    out = pf.update(metrics_from_landmarks(empty_seat()))
    assert out["p_at_risk"] == 0.0 and out["confident"] is False
    assert out["decision"] == "NO SIGNAL"
    print("ok  empty seat: no score, no alarm")


def test_baseline_survives_a_json_round_trip():
    base = build_baseline(sample_run(40))
    again = PostureBaseline.from_dict(base.as_dict())
    assert again is not None and again.samples == base.samples
    assert abs(again.head_height[0] - base.head_height[0]) < 1e-9
    assert PostureBaseline.from_dict({}) is None
    assert PostureBaseline.from_dict({"samples": 3}) is None
    print("ok  persistence: baseline survives the profile round trip")


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
