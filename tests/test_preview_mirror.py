"""
tests/test_preview_mirror.py — the camera preview is mirrored, the text is not.

Operators read the preview as a mirror: every mirror and video-call preview they
have used is flipped, so an un-flipped feed looks wrong. Flipping the finished
image in CSS would have been one line, but it would also reverse the "EAR 0.312
EYES OPEN" and "FATIGUE 22% ALERT OK" overlays into mirror writing. So the flip
happens on the frame, before anything is drawn.

Checks the image really is mirrored, the mesh follows it, and the overlay text
does not — without needing a camera.

Run:  python -m tests.test_preview_mirror
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.live_camera import BackgroundCameraTracker, RemoteCameraTracker  # noqa: E402

W, H = 320, 240
OVERLAY = {"p_pct": 22, "decision": "ALERT", "severity": "OK"}


def annotator(cls):
    """Bind _annotate_frame to a bare object — no camera, no MediaPipe."""
    obj = cls.__new__(cls)
    obj._preview_lock = threading.Lock()
    obj._fatigue_overlay = dict(OVERLAY)
    return obj


def marked_frame() -> np.ndarray:
    """Dark frame with a bright block on the LEFT third only."""
    img = np.full((H, W, 3), 40, np.uint8)
    cv2.rectangle(img, (10, 90), (70, 150), (255, 255, 255), -1)
    return img


def brightness(img, x0, x1) -> float:
    band = img[60:180, x0:x1]
    return float(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).mean())


def face_landmarks(x=0.2, y=0.5):
    """A single landmark at a known, deliberately off-centre position."""
    return [SimpleNamespace(x=x, y=y, z=0.0) for _ in range(478)]


# ── 1. the image is mirrored ─────────────────────────────────────────────
def _check_mirrored(cls, name):
    out = annotator(cls)._annotate_frame(marked_frame(), None, 0.0, H, W)
    left, right = brightness(out, 0, W // 3), brightness(out, 2 * W // 3, W)
    assert right > left + 50, \
        f"{name}: marker did not move to the right ({left:.0f} vs {right:.0f})"
    return left, right


def test_remote_preview_is_mirrored():
    left, right = _check_mirrored(RemoteCameraTracker, "remote")
    print(f"ok  browser feed mirrored: left {left:.0f} -> right {right:.0f}")


def test_local_preview_is_mirrored():
    left, right = _check_mirrored(BackgroundCameraTracker, "local")
    print(f"ok  cab camera mirrored: left {left:.0f} -> right {right:.0f}")


def test_input_frame_is_not_modified():
    """The annotator is display-only. Mutating the caller's frame would corrupt
    the features, the face template and the pose measurement all at once."""
    frame = marked_frame()
    before = frame.copy()
    annotator(RemoteCameraTracker)._annotate_frame(frame, None, 0.0, H, W)
    assert np.array_equal(frame, before), "annotator mutated the source frame"
    print("ok  non-destructive: the source frame is untouched")


# ── 2. the mesh follows the flip ─────────────────────────────────────────
def test_mesh_dots_are_mirrored_with_the_image():
    """If the picture flips but the mesh does not, the dots land on the wrong
    side of the face — which looks like a tracking failure."""
    out = annotator(RemoteCameraTracker)._annotate_frame(
        marked_frame(), face_landmarks(x=0.2), 0.30, H, W)

    # Search between the overlay bars only: the status text is green too, and
    # it is left-aligned by design, so including it would drag the mean left.
    band = out[35:H - 40]
    green = (band[:, :, 1].astype(int) - band[:, :, 2].astype(int) > 60)
    xs = np.where(green.any(axis=0))[0]
    assert xs.size, "no mesh drawn at all"
    centre = xs.mean() / W
    assert 0.7 < centre < 0.9, \
        f"landmark at x=0.20 drew at {centre:.2f}, expected ~0.80"
    print(f"ok  mesh follows: landmark x=0.20 drawn at x={centre:.2f}")


# ── 3. the text does not ─────────────────────────────────────────────────
def test_overlay_bars_stay_put_and_read_forwards():
    """The bars are drawn after the flip at fixed coordinates, so they must be
    identical whichever way round the incoming frame was."""
    ann = annotator(RemoteCameraTracker)
    normal = ann._annotate_frame(marked_frame(), None, 0.0, H, W)
    flipped_input = annotator(RemoteCameraTracker)._annotate_frame(
        cv2.flip(marked_frame(), 1), None, 0.0, H, W)

    for label, y0, y1 in (("top", 0, 30), ("bottom", H - 36, H)):
        a, b = normal[y0:y1], flipped_input[y0:y1]
        assert np.array_equal(a, b), \
            f"{label} overlay bar changed with the input orientation"
        # Text is left-aligned in both bars; mirrored text would sit right.
        ink = (cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) > 60)
        cols = np.where(ink.any(axis=0))[0]
        assert cols.size and cols.mean() < W * 0.55, \
            f"{label} overlay text drifted right — it looks mirrored"
    print("ok  overlay text: identical either way, still left-aligned")


def test_no_face_label_is_still_drawn():
    out = annotator(RemoteCameraTracker)._annotate_frame(marked_frame(), None, 0.0, H, W)
    top = cv2.cvtColor(out[0:30], cv2.COLOR_BGR2GRAY)
    assert (top > 60).sum() > 40, "the NO FACE DETECTED banner vanished"
    print("ok  labels intact: the no-face banner still renders after the flip")


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
