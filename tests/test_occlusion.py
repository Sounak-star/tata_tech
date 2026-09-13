"""
tests/test_occlusion.py — telling a visible face from one that is merely fitted.

MediaPipe keeps reporting a face through sunglasses and a mask, so `has_face` is
useless as an occlusion signal. These build synthetic faces — skin-coloured,
with and without dark lenses and a cloth mask — and check that the skin-fraction
test sees what a person would.

The distinction the whole feature turns on: EYES covered is dangerous (PERCLOS
becomes fiction), MOUTH covered is not (PERCLOS still works).

Run:  python -m tests.test_occlusion
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.occlusion import (                                     # noqa: E402
    EYE_COVERED_RATIO, OCCLUSION_RUN, Occlusion, OcclusionTracker,
    assess_occlusion, skin_fraction,
)

W, H = 640, 480
SKIN = (150, 175, 215)          # BGR, a mid brown-pink that lands in the Cr/Cb box


def face_image(*, sunglasses=False, mask=False, skin=SKIN) -> np.ndarray:
    """A crude but colour-correct face: skin oval, optional lenses and mask."""
    img = np.full((H, W, 3), 30, np.uint8)              # dark cab background
    cv2.ellipse(img, (320, 240), (150, 200), 0, 0, 360, skin, -1)
    # a little texture so Laplacian-style measures are not degenerate
    noise = np.random.default_rng(0).integers(-8, 8, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if sunglasses:
        cv2.rectangle(img, (215, 175), (425, 225), (18, 18, 20), -1)
    if mask:
        cv2.rectangle(img, (230, 285), (410, 400), (200, 205, 205), -1)
    return img


import cv2  # noqa: E402  (after face_image so the helper reads top-down)


def landmarks():
    """Mesh points matching face_image()'s geometry, in normalised coords."""
    pts = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]

    def put(i, x, y):
        pts[i] = SimpleNamespace(x=x / W, y=y / H, z=0.0)

    for i, (x, y) in zip([362, 385, 387, 263, 373, 380],
                         [(360, 200), (375, 190), (395, 190),
                          (410, 200), (395, 210), (375, 210)]):
        put(i, x, y)
    for i, (x, y) in zip([33, 160, 158, 133, 153, 144],
                         [(230, 200), (245, 190), (265, 190),
                          (280, 200), (265, 210), (245, 210)]):
        put(i, x, y)
    for i, (x, y) in zip([61, 291, 0, 17, 13, 14],
                         [(270, 340), (370, 340), (320, 320),
                          (320, 365), (320, 335), (320, 345)]):
        put(i, x, y)
    put(50, 250, 265)          # cheeks — the skin reference
    put(280, 390, 265)
    return pts


# ── 1. the measurement itself ────────────────────────────────────────────
def test_skin_fraction_separates_skin_from_lenses_and_cloth():
    skin_patch = np.full((40, 40, 3), SKIN, np.uint8)
    lens = np.full((40, 40, 3), (18, 18, 20), np.uint8)
    cloth = np.full((40, 40, 3), (200, 205, 205), np.uint8)

    s, l, c = (skin_fraction(x) for x in (skin_patch, lens, cloth))
    assert s > 0.9, f"skin did not read as skin ({s})"
    assert l < 0.1 and c < 0.1, f"lens={l} cloth={c} read as skin"
    print(f"ok  measurement: skin {s:.2f}, dark lens {l:.2f}, cloth {c:.2f}")


def test_bare_face_reads_as_uncovered():
    occ = assess_occlusion(face_image(), landmarks())
    assert occ.valid, occ.note
    assert not occ.eyes_covered and not occ.mouth_covered, occ.as_dict()
    assert occ.eye_ratio > 0.8 and occ.mouth_ratio > 0.8
    print(f"ok  bare face: eye ratio {occ.eye_ratio:.2f}, "
          f"mouth ratio {occ.mouth_ratio:.2f}")


def test_sunglasses_are_detected():
    occ = assess_occlusion(face_image(sunglasses=True), landmarks())
    assert occ.valid and occ.eyes_covered, occ.as_dict()
    assert not occ.mouth_covered, "sunglasses should not implicate the mouth"
    print(f"ok  sunglasses: eye ratio {occ.eye_ratio:.2f} "
          f"(threshold {EYE_COVERED_RATIO})")


def test_mask_is_detected_but_the_eyes_stay_clear():
    """The screenshot case. The mouth is gone; PERCLOS is untouched."""
    occ = assess_occlusion(face_image(mask=True), landmarks())
    assert occ.valid and occ.mouth_covered, occ.as_dict()
    assert not occ.eyes_covered, \
        "a dust mask must not be read as the eyes being covered"
    print(f"ok  dust mask: mouth ratio {occ.mouth_ratio:.2f}, "
          f"eyes still clear at {occ.eye_ratio:.2f}")


def test_both_covered():
    occ = assess_occlusion(face_image(sunglasses=True, mask=True), landmarks())
    assert occ.eyes_covered and occ.mouth_covered, occ.as_dict()
    print("ok  sunglasses + mask: both regions reported covered")


def test_ratio_is_robust_to_skin_tone_and_lighting():
    """Absolute skin thresholds fail across people and cab lighting; a ratio
    against the same face's own cheek should not."""
    for label, tone in (("lighter", (175, 200, 235)), ("darker", (95, 115, 150))):
        bare = assess_occlusion(face_image(skin=tone), landmarks())
        shaded = assess_occlusion(face_image(sunglasses=True, skin=tone), landmarks())
        assert bare.valid, f"{label}: no reference skin ({bare.note})"
        assert not bare.eyes_covered, f"{label} bare face read as covered"
        assert shaded.eyes_covered, f"{label} sunglasses missed"
    print("ok  robustness: verdict holds across lighter and darker skin tones")


def test_unusable_reference_reports_unknown_rather_than_guessing():
    dark = np.full((H, W, 3), 12, np.uint8)          # no skin anywhere
    occ = assess_occlusion(dark, landmarks())
    assert not occ.valid and "reference" in occ.note
    assert not occ.eyes_covered, "guessed a verdict with no reference"
    print(f"ok  no reference: reports unknown — \"{occ.note}\"")


def test_no_face_is_not_an_occlusion_verdict():
    occ = assess_occlusion(face_image(), None)
    assert not occ.valid and not occ.eyes_covered
    print("ok  no landmarks: no verdict, no false 'covered'")


# ── 2. stability ─────────────────────────────────────────────────────────
def test_verdict_needs_agreement_across_frames():
    """A hand passing the face, or one frame of glare, must not flip the
    fatigue source."""
    trk = OcclusionTracker()
    clear, covered = face_image(), face_image(sunglasses=True)
    lms = landmarks()

    for _ in range(OCCLUSION_RUN + 2):
        trk.update(clear, lms, ear=0.30)
    assert not trk.eyes_covered

    trk.update(covered, lms, ear=0.30)               # a single bad frame
    assert not trk.eyes_covered, "one frame flipped the verdict"

    for _ in range(OCCLUSION_RUN):
        trk.update(covered, lms, ear=0.30)
    assert trk.eyes_covered, "sustained occlusion never registered"
    assert "covered" in trk.reason()

    for _ in range(OCCLUSION_RUN):
        trk.update(clear, lms, ear=0.30)
    assert not trk.eyes_covered, "never recovered when the glasses came off"
    print(f"ok  stability: {OCCLUSION_RUN} frames of agreement each way, "
          f"single frames ignored")


def test_a_frozen_eye_signal_counts_as_covered():
    """Catches lenses the colour test misses: if EAR never moves, the mesh is
    reporting a model prior, not an eyelid."""
    trk = OcclusionTracker()
    clear, lms = face_image(), landmarks()

    for _ in range(trk.EAR_WINDOW + OCCLUSION_RUN):
        trk.update(clear, lms, ear=0.2999)           # pinned, never blinks
    assert trk.ear_flat, "a perfectly flat EAR was not noticed"
    assert trk.eyes_covered, "frozen eye signal was still trusted"
    assert "not being observed" in trk.reason()
    print(f"ok  frozen signal: flat EAR over {trk.EAR_WINDOW} windows "
          f"-> \"{trk.reason()}\"")


def test_a_blinking_operator_is_not_flagged():
    """The false positive that would matter most: someone sitting still, with
    their eyes genuinely visible, must not be pushed into posture mode."""
    trk = OcclusionTracker()
    clear, lms = face_image(), landmarks()
    rng = np.random.default_rng(7)

    for i in range(trk.EAR_WINDOW * 2):
        ear = 0.12 if i % 17 == 0 else 0.30 + 0.01 * rng.standard_normal()
        trk.update(clear, lms, ear=ear)
    assert not trk.ear_flat, "a blinking operator read as a frozen signal"
    assert not trk.eyes_covered
    print("ok  no false positive: a still-but-blinking operator stays in face mode")


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
