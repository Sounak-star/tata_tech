"""
brain/pose.py — upper-body pose, for when the face is not visible.

Sunglasses and a dust mask defeat EAR/MAR/PERCLOS entirely.  What they do not
defeat is posture: MediaPipe Pose still returns the nose, both ears and both
shoulders, and those are enough to see a head drop, a shoulder slump, or the
side-to-side sway of someone losing control of their trunk.

Two ideas carry this module.

  1. PRESENCE IS NOT THE SAME AS A FACE.  The single most valuable thing pose
     gives us is the ability to tell "the operator is here but I cannot see
     their face" from "the seat is empty".  Both look identical to the face
     pipeline, and getting them confused is dangerous in both directions:
     alarming at an empty cab teaches operators to ignore alerts, and trusting
     the last good face reading during an occlusion means reporting ALERT while
     somebody falls asleep.

  2. EVERY MEASUREMENT IS NORMALISED BY SHOULDER WIDTH.  Raw pixel geometry is
     worthless here — a tall operator, a raised seat, or a camera remounted two
     inches lower all change it.  Inter-shoulder distance is the natural ruler:
     it is in the frame, it belongs to the operator, and it scales with them.

This module measures.  It does not decide anything about fatigue — that is
posture_fatigue.py, which compares these numbers to the operator's own baseline.
"""

from __future__ import annotations

import os
import threading
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from .paths import PROJECT_ROOT

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:                                   # pragma: no cover
    HAS_MEDIAPIPE = False

# Same lite/float16 family as the face landmarker the project already ships.
POSE_MODEL_URL = os.environ.get(
    "SAARTHI_POSE_MODEL_URL",
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
)
POSE_MODEL_PATH = Path(os.environ.get(
    "SAARTHI_POSE_MODEL_PATH", str(PROJECT_ROOT / "pose_landmarker_lite.task")))

# MediaPipe Pose landmark indices (33-point topology).
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12

MIN_VISIBILITY = 0.5        # per-landmark confidence to count as seen
PRESENCE_FRAMES = 3         # consecutive frames of shoulders before "present"
ABSENCE_FRAMES = 8          # consecutive frames without shoulders before "empty"


@dataclass
class PoseMetrics:
    """One frame of posture, all lengths in shoulder-widths (dimensionless).

    Dimensionless is the point: these numbers are comparable across operators,
    seat heights and camera mountings, which raw pixels are not.
    """
    present: bool = False
    shoulder_width: float = 0.0     # normalised by frame width — a size proxy
    head_height: float = 0.0        # ear-midpoint above shoulder line
    head_forward: float = 0.0       # ear midpoint ahead of shoulder midpoint (x)
    shoulder_y: float = 0.0         # shoulder-line height in the frame (0=top)
    shoulder_roll: float = 0.0      # degrees; one shoulder dropped vs the other
    centre_x: float = 0.0           # shoulder midpoint x, for sway
    visibility: float = 0.0         # mean visibility of the five landmarks used

    def as_dict(self) -> dict:
        return {k: (v if isinstance(v, bool) else round(float(v), 4))
                for k, v in asdict(self).items()}


def _pt(lms, idx: int) -> np.ndarray:
    lm = lms[idx]
    return np.array([lm.x, lm.y], dtype=np.float32)


def _vis(lms, idx: int) -> float:
    return float(getattr(lms[idx], "visibility", 1.0))


def metrics_from_landmarks(lms) -> PoseMetrics:
    """Turn 33 pose landmarks into normalised posture measurements.

    Returns present=False when the shoulders are not confidently visible: without
    them there is no ruler, so every other number would be meaningless.
    """
    if lms is None or len(lms) <= R_SHOULDER:
        return PoseMetrics()

    vis = [_vis(lms, i) for i in (NOSE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER)]
    if _vis(lms, L_SHOULDER) < MIN_VISIBILITY or _vis(lms, R_SHOULDER) < MIN_VISIBILITY:
        return PoseMetrics(visibility=float(np.mean(vis)))

    ls, rs = _pt(lms, L_SHOULDER), _pt(lms, R_SHOULDER)
    width = float(np.linalg.norm(ls - rs))
    if width < 1e-4:
        return PoseMetrics(visibility=float(np.mean(vis)))

    shoulder_mid = (ls + rs) / 2.0

    # Prefer the ears (unaffected by sunglasses or a mask); fall back to the nose.
    ears = [i for i in (L_EAR, R_EAR) if _vis(lms, i) >= MIN_VISIBILITY]
    if ears:
        head = np.mean([_pt(lms, i) for i in ears], axis=0)
    elif _vis(lms, NOSE) >= MIN_VISIBILITY:
        head = _pt(lms, NOSE)
    else:
        head = shoulder_mid.copy()          # head lost — reads as fully dropped

    # y grows downward in normalised image coords, so "above" is a negative dy.
    head_height = float((shoulder_mid[1] - head[1]) / width)
    head_forward = float(abs(head[0] - shoulder_mid[0]) / width)

    dx, dy = float(rs[0] - ls[0]), float(rs[1] - ls[1])
    roll = float(np.degrees(np.arctan2(dy, dx)))
    if roll > 90:
        roll -= 180
    elif roll < -90:
        roll += 180

    return PoseMetrics(
        present=True,
        shoulder_width=width,
        head_height=head_height,
        head_forward=head_forward,
        shoulder_y=float(shoulder_mid[1]),
        shoulder_roll=roll,
        centre_x=float(shoulder_mid[0]),
        visibility=float(np.mean(vis)),
    )


class PoseTracker:
    """MediaPipe PoseLandmarker over frames the face pipeline already decoded.

    Runs at a low duty cycle: posture changes over seconds, and a full-rate pose
    graph on a cab CPU would cost more than the fatigue pipeline it is backing up.
    """

    def __init__(self, *, every_n: int = 2, auto_download: bool = True) -> None:
        self.available = False
        self.backend = "unavailable"
        self.last_error: Optional[str] = None
        self.every_n = max(1, int(every_n))

        self._landmarker = None
        self._lock = threading.Lock()
        self._frame = 0
        self._ts_ms = 0
        self.latest = PoseMetrics()

        self._present_run = 0
        self._absent_run = 0
        self._present = False

        if not HAS_MEDIAPIPE:
            self.last_error = "mediapipe not installed"
            return

        if not POSE_MODEL_PATH.exists():
            if not auto_download or not download_pose_model():
                self.last_error = (f"pose model missing at {POSE_MODEL_PATH}; "
                                   f"run `python -m brain.pose --download`")
                print(f"[Pose] {self.last_error}", flush=True)
                return
        try:
            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            self._landmarker = PoseLandmarker.create_from_options(
                PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
                    running_mode=VisionRunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_tracking_confidence=0.5))
            self.available = True
            self.backend = "mediapipe-pose-lite"
            print("[Pose] PoseLandmarker ready.", flush=True)
        except Exception as exc:            # noqa: BLE001 — never take the cab down
            self.last_error = f"pose init failed: {exc}"
            print(f"[Pose] {self.last_error}", flush=True)

    # ── presence ────────────────────────────────────────────────────────
    @property
    def present(self) -> bool:
        """Somebody is in the seat, whether or not their face is readable."""
        return self._present

    def _update_presence(self, seen: bool) -> None:
        # Asymmetric run lengths: quick to notice someone, slow to declare the
        # seat empty. Declaring "empty" wrongly silences monitoring altogether.
        if seen:
            self._present_run += 1
            self._absent_run = 0
            if self._present_run >= PRESENCE_FRAMES:
                self._present = True
        else:
            self._absent_run += 1
            self._present_run = 0
            if self._absent_run >= ABSENCE_FRAMES:
                self._present = False

    # ── per-frame ───────────────────────────────────────────────────────
    def observe(self, bgr) -> Optional[PoseMetrics]:
        """Offer one BGR frame. Returns metrics on the frames it actually runs."""
        if not self.available or bgr is None:
            return None

        self._frame += 1
        if self._frame % self.every_n:
            return None

        try:
            import cv2
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._ts_ms += 40 * self.every_n          # monotonic, VIDEO mode needs it
            with self._lock:
                result = self._landmarker.detect_for_video(image, self._ts_ms)
        except Exception as exc:                      # noqa: BLE001
            self.last_error = f"pose inference failed: {exc}"
            return None

        lms = (result.pose_landmarks[0]
               if getattr(result, "pose_landmarks", None) else None)
        metrics = metrics_from_landmarks(lms)
        self._update_presence(metrics.present)
        self.latest = metrics
        return metrics

    def reset(self) -> None:
        self._present_run = self._absent_run = 0
        self._present = False
        self.latest = PoseMetrics()

    def status(self) -> dict:
        return {"available": self.available, "backend": self.backend,
                "present": self._present, "error": self.last_error,
                "metrics": self.latest.as_dict()}


def download_pose_model(path: Path = POSE_MODEL_PATH, url: str = POSE_MODEL_URL) -> bool:
    """Fetch the pose landmarker task file (~6 MB). Returns success."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        print(f"[Pose] downloading pose model from {url} ...", flush=True)
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
        print(f"[Pose] saved to {path}", flush=True)
        return True
    except Exception as exc:                          # noqa: BLE001
        print(f"[Pose] download failed: {exc}", flush=True)
        return False


if __name__ == "__main__":
    import sys
    if "--download" in sys.argv:
        raise SystemExit(0 if download_pose_model() else 1)
    print(__doc__)
