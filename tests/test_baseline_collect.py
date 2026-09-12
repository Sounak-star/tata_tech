"""
tests/test_baseline_collect.py — baseline capture picks the tracker that works.

Regression for a live failure: enrolment sat at "Processing baseline…" and then
"calibration produced no usable windows". The cab had TWO trackers — the local
OpenCV one (holding /dev/video0 from a failed startup calibration, producing
nothing) and the browser one (receiving every real frame over the WebSocket).
The collector chose by camera handle, so it polled the dead tracker for 25 s.

Run:  python -m tests.test_baseline_collect
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_TMP = tempfile.mkdtemp()
os.environ.setdefault("SAARTHI_TEMPLATES", str(Path(_TMP) / "t.enc"))
os.environ.setdefault("SAARTHI_FACEID_KEY", str(Path(_TMP) / "k.key"))
os.environ.setdefault("SAARTHI_SUPERVISORS", str(Path(_TMP) / "sup.json"))

import numpy as np                                              # noqa: E402

from brain import server                                        # noqa: E402
from brain.enrollment import EnrollmentSession, OperatorDetails  # noqa: E402
from brain.faceid import EMBED_DIM, FaceEmbedder, l2             # noqa: E402
from brain.fatigue import FEATURES                               # noqa: E402


class DeadTracker:
    """Holds a camera handle but never produces a feature window.

    This is exactly the local tracker's state when the browser owns the webcam:
    cv2.VideoCapture opened, calibration failed, run loop never started.
    """

    class _Cap:
        def isOpened(self):
            return True

    cap = _Cap()

    def get_latest_features(self):
        return None


class LiveTracker:
    """Produces a fresh feature window on every poll, like the browser path."""

    cap = None

    def __init__(self):
        self.n = 0

    def get_latest_features(self):
        self.n += 1
        row = {f: 1.0 + 0.01 * self.n for f in FEATURES}
        row["ear_mean"] = 0.30 + 0.001 * self.n
        row["perclos"] = 0.05
        row["blink_rate"] = 15.0 + 0.1 * self.n
        return row


def make_session() -> EnrollmentSession:
    emb = FaceEmbedder(session=lambda nchw: np.zeros(EMBED_DIM, np.float32))
    sess = EnrollmentSession(OperatorDetails(id="op_950", name="Baseline Test"),
                             emb, supervisor="sup_001", shuffle=False)
    sess.vectors = [l2(np.random.default_rng(i).standard_normal(EMBED_DIM)
                       .astype(np.float32)) for i in range(5)]
    sess.vector_poses = ["center"] * 5
    return sess


def run_collect(sess, *, local, remote, seconds=1.0, cap=3.0, minimum=3):
    saved = (server.tracker, server.remote_tracker, server.BASELINE_SECONDS,
             server.BASELINE_MAX_SECONDS, server.MIN_BASELINE_WINDOWS)
    server.tracker, server.remote_tracker = local, remote
    server.BASELINE_SECONDS, server.BASELINE_MAX_SECONDS = seconds, cap
    server.MIN_BASELINE_WINDOWS = minimum
    try:
        asyncio.run(server._collect_baseline(sess))
    finally:
        (server.tracker, server.remote_tracker, server.BASELINE_SECONDS,
         server.BASELINE_MAX_SECONDS, server.MIN_BASELINE_WINDOWS) = saved


# ── 1. the regression ────────────────────────────────────────────────────
def test_dead_local_tracker_does_not_shadow_the_browser():
    sess = make_session()
    run_collect(sess, local=DeadTracker(), remote=LiveTracker())

    assert not sess.error, f"baseline failed: {sess.error}"
    assert sess.baseline_summary, "no baseline captured from the live tracker"
    assert sess.baseline_summary["ear_baseline"] > 0
    assert sess.baseline_progress >= 3
    print(f"ok  regression: browser tracker used despite the local one holding a "
          f"camera handle ({sess.baseline_progress} windows)")


def test_local_tracker_is_used_when_it_is_the_live_one():
    """The cab-camera case must still work — this is not a browser-only fix."""
    sess = make_session()
    run_collect(sess, local=LiveTracker(), remote=DeadTracker())

    assert not sess.error, f"baseline failed: {sess.error}"
    assert sess.baseline_summary, "local tracker output was ignored"
    print(f"ok  cab camera: local tracker used when it is the producer "
          f"({sess.baseline_progress} windows)")


# ── 2. failure is explained, not silent ──────────────────────────────────
def test_no_windows_gives_an_actionable_error():
    sess = make_session()
    run_collect(sess, local=DeadTracker(), remote=DeadTracker())

    assert not sess.baseline_summary, "a baseline was invented from nothing"
    assert sess.error, "silent failure — the operator is told nothing"
    assert "camera" in sess.error.lower(), sess.error
    print(f"ok  no windows: '{sess.error[:58]}…'")


def test_too_few_windows_is_refused():
    """A handful of windows cannot support a per-feature mean and std."""
    sess = make_session()
    run_collect(sess, local=DeadTracker(), remote=LiveTracker(),
                seconds=0.3, cap=0.4, minimum=50)

    assert not sess.baseline_summary, "accepted a baseline built from too few windows"
    assert sess.error and "of 50 baseline windows" in sess.error, sess.error
    print(f"ok  short capture: refused — '{sess.error[:58]}…'")


def test_repeated_identical_windows_are_not_counted_twice():
    """The tracker republishes its last window between updates; counting those
    would let a frozen camera look like a healthy 25-second capture."""
    class Frozen:
        cap = None
        def get_latest_features(self):
            return {f: 1.0 for f in FEATURES}

    sess = make_session()
    run_collect(sess, local=DeadTracker(), remote=Frozen(), minimum=3)

    assert sess.baseline_progress <= 1, \
        f"a frozen tracker produced {sess.baseline_progress} distinct windows"
    assert sess.error, "a frozen camera passed as a valid baseline"
    print("ok  frozen camera: repeated windows deduplicated, capture refused")


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
