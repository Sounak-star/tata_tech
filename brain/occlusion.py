"""
brain/occlusion.py — is the face actually visible, or just being guessed at?

MediaPipe's face landmarker is stubborn. Put on sunglasses and a dust mask and
it will still happily fit a 478-point mesh, report EAR 0.30 and "EYES OPEN", and
hand the fatigue model twelve confident features derived from geometry it cannot
actually see.  `has_face` stays True, so a naive "no face -> use posture" switch
never fires — and the model gets confident wrong numbers, which is worse than
getting none.

So we ask a different question: IS THERE SKIN WHERE THE EYES SHOULD BE.

Skin fraction is the right measurement because it answers that question
directly. An uncovered eye socket is mostly skin; a sunglass lens is not. It is
measured as a RATIO against the operator's own cheek in the SAME frame, which
cancels out skin tone, cab lighting, exposure and white balance — all the things
an absolute threshold would get wrong.

One distinction matters more than any threshold here:

    EYES covered  -> EAR, PERCLOS and blink rate are fiction. This is the
                     dangerous case, and it must fall back to posture.
    MOUTH covered -> only MAR and yawn detection are lost. PERCLOS still works,
                     and PERCLOS is the strongest drowsiness signal we have.
                     Falling back to posture here would be a DOWNGRADE.

A dust mask alone is therefore not a reason to switch. Reported, not acted on.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Mesh indices. Eyes reuse the sets extract_features.py already uses for EAR.
_LEFT_EYE = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]
_MOUTH = [61, 291, 0, 17, 13, 14]
# Reference skin: the cheeks. Close enough to the eyes and mouth to share their
# lighting, far enough that neither sunglasses nor a mask normally reaches them.
_CHEEKS = [50, 280]

# Skin fraction in a region, relative to the cheek reference, below which we call
# it covered. TUNE ON SITE: dark lenses, tinted visors and light dust masks all
# sit at different points, and a cab windscreen changes the colour cast.
EYE_COVERED_RATIO = 0.45
MOUTH_COVERED_RATIO = 0.45
MIN_REFERENCE_SKIN = 0.25     # below this the cheek is unreliable; report unknown

# Frames of agreement before flipping the verdict. A single frame of glare or a
# hand passing the face must not change the fatigue source.
OCCLUSION_RUN = 6


@dataclass
class Occlusion:
    """Per-region visibility, with the numbers behind the verdict."""
    valid: bool = False           # was the reference skin usable at all
    eyes_covered: bool = False
    mouth_covered: bool = False
    eye_ratio: float = 1.0        # eye skin fraction / cheek skin fraction
    mouth_ratio: float = 1.0
    reference_skin: float = 0.0
    note: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("eye_ratio", "mouth_ratio", "reference_skin"):
            d[k] = round(float(d[k]), 3)
        return d


def _bbox(landmarks, idxs: Sequence[int], w: int, h: int,
          pad: float = 0.35) -> Tuple[int, int, int, int]:
    xs = [landmarks[i].x * w for i in idxs]
    ys = [landmarks[i].y * h for i in idxs]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    px, py = max((x1 - x0) * pad, 4.0), max((y1 - y0) * pad, 4.0)
    return (int(max(0, x0 - px)), int(max(0, y0 - py)),
            int(min(w, x1 + px)), int(min(h, y1 + py)))


def skin_fraction(bgr_region: np.ndarray) -> float:
    """Share of pixels that look like skin, in YCrCb.

    The classic Cr/Cb box. It is crude in absolute terms — which is exactly why
    we only ever use it as a ratio against the same face's cheek in the same
    frame, where the crudeness cancels.
    """
    if bgr_region is None or bgr_region.size == 0:
        return 0.0
    ycrcb = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    mask = (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 130)
    return float(mask.mean())


def assess_occlusion(bgr: np.ndarray, landmarks) -> Occlusion:
    """Decide, for this frame, whether the eyes and mouth are actually visible."""
    if bgr is None or landmarks is None or len(landmarks) < 468:
        return Occlusion(note="no face")

    h, w = bgr.shape[:2]

    def region(idxs, pad=0.35) -> np.ndarray:
        x0, y0, x1, y1 = _bbox(landmarks, idxs, w, h, pad)
        return bgr[y0:y1, x0:x1]

    # The reference must be SKIN, so sample tight patches centred on each cheek
    # rather than a padded box across both: a wide box spills off the face onto
    # the cab behind, which dilutes the reference and makes every ratio drift.
    span = abs(landmarks[_CHEEKS[1]].x - landmarks[_CHEEKS[0]].x) * w
    r = int(max(6, span * 0.16))
    patches = []
    for idx in _CHEEKS:
        cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        patch = bgr[max(0, cy - r):min(h, cy + r), max(0, cx - r):min(w, cx + r)]
        if patch.size:
            patches.append(skin_fraction(patch))
    ref = float(np.mean(patches)) if patches else 0.0
    if ref < MIN_REFERENCE_SKIN:
        # We cannot see enough skin anywhere to calibrate against. Could be heavy
        # shadow, a full-face covering, or a camera fault — either way, refusing
        # to guess is the honest answer.
        return Occlusion(valid=False, reference_skin=ref,
                         note="no reliable skin reference on this face")

    eye = (skin_fraction(region(_LEFT_EYE)) + skin_fraction(region(_RIGHT_EYE))) / 2.0
    mouth = skin_fraction(region(_MOUTH))

    eye_ratio = eye / ref
    mouth_ratio = mouth / ref
    return Occlusion(
        valid=True,
        eyes_covered=eye_ratio < EYE_COVERED_RATIO,
        mouth_covered=mouth_ratio < MOUTH_COVERED_RATIO,
        eye_ratio=eye_ratio,
        mouth_ratio=mouth_ratio,
        reference_skin=ref,
    )


class OcclusionTracker:
    """Smooths per-frame verdicts into a stable one.

    Also watches whether the EAR signal is ALIVE. A hallucinated eye mesh tends
    to sit still: if EAR has not moved and no blink has been seen for a long
    stretch, the landmarks are reporting a model prior rather than an eyelid.
    That corroborates the pixel test and catches lenses the colour test misses.
    """

    EAR_WINDOW = 60          # ~12 s of windows at the browser's rate
    EAR_FLAT_RANGE = 0.015   # EAR range below this over a full window is not a
                             # living eye — real ones jitter and blink

    def __init__(self) -> None:
        self.state = Occlusion()
        self._eyes_run = 0
        self._clear_run = 0
        self._ears: Deque[float] = deque(maxlen=self.EAR_WINDOW)
        self.eyes_covered = False
        self.ear_flat = False

    def reset(self) -> None:
        self._eyes_run = self._clear_run = 0
        self._ears.clear()
        self.eyes_covered = False
        self.ear_flat = False
        self.state = Occlusion()

    def update(self, bgr: np.ndarray, landmarks, *, ear: float = 0.0) -> Occlusion:
        occ = assess_occlusion(bgr, landmarks)
        self.state = occ

        if ear:
            self._ears.append(float(ear))
        self.ear_flat = (len(self._ears) == self._ears.maxlen
                         and (max(self._ears) - min(self._ears)) < self.EAR_FLAT_RANGE)

        covered_now = (occ.valid and occ.eyes_covered) or self.ear_flat
        if covered_now:
            self._eyes_run += 1
            self._clear_run = 0
        else:
            self._clear_run += 1
            self._eyes_run = 0

        if self._eyes_run >= OCCLUSION_RUN:
            self.eyes_covered = True
        elif self._clear_run >= OCCLUSION_RUN:
            self.eyes_covered = False
        return occ

    def status(self) -> dict:
        return {
            **self.state.as_dict(),
            # The stable verdict, as opposed to this frame's raw reading.
            "eyes_covered_stable": self.eyes_covered,
            "ear_flat": self.ear_flat,
            "ear_samples": len(self._ears),
        }

    def reason(self) -> str:
        if not self.eyes_covered:
            return ""
        if self.ear_flat and not self.state.eyes_covered:
            return "eye signal has not moved — eyes are not being observed"
        return "eyes appear covered (sunglasses or visor)"
