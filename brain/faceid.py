"""
brain/faceid.py — Q1: "Who is sitting in the seat?" (face signatures, not photos).

Design rules, in priority order:

  1. No image ever reaches disk.  Frames are decoded, aligned, embedded and
     dropped inside the calling request.  The only durable artefact is a set of
     512-d float vectors.
  2. The template is a SET, not a mean.  Enrolment deliberately captures pose
     variation (straight / left / right / chin-down / no glasses); averaging
     those lands on a centroid that matches none of them well.  We score with
     max-cosine over the set and keep the centroid only as a tiebreaker.
  3. Templates are stored through a fixed random ORTHOGONAL projection Q.
     Because <Qa, Qb> == <a, b>, cosine distance is preserved exactly — matching
     is unaffected — but a leaked template cannot be replayed against a public
     face model, and it is revocable: rotate Q, re-enrol, the old dump is dead.
     (Cancellable biometrics.  Q lives in the key file, never beside the data.)
  4. Mirror-invariance for free.  Browser webcam feeds are often horizontally
     flipped while the cab camera is not.  We embed the crop AND its mirror and
     sum them: e(x) + e(flip(x)) is identical whichever way round the input came,
     so an operator enrolled mirrored still matches unmirrored — and vice versa.

Everything degrades the way the rest of the codebase does: if onnxruntime or the
ArcFace model is missing, `FaceEmbedder.available` is False and the caller falls
back to manual (kiosk) operator selection rather than guessing.
"""

from __future__ import annotations

import os
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .paths import DATA, PROJECT_ROOT

# ── model ────────────────────────────────────────────────────────────────
EMBED_DIM = 512
CROP_SIZE = 112

# InsightFace buffalo_l recognition head (ArcFace w600k_r50), single-file ONNX.
# Override with SAARTHI_ARCFACE_URL / SAARTHI_ARCFACE_PATH for an air-gapped site.
ARCFACE_URL = os.environ.get(
    "SAARTHI_ARCFACE_URL",
    "https://huggingface.co/immich-app/buffalo_l/resolve/main/recognition/model.onnx",
)
ARCFACE_PATH = Path(os.environ.get("SAARTHI_ARCFACE_PATH", str(PROJECT_ROOT / "arcface_r50.onnx")))

# ── landmark → canonical 5-point alignment ───────────────────────────────
# ArcFace's canonical destination points for a 112x112 crop.
_ARCFACE_5PT = np.array([
    [38.2946, 51.6963],   # left eye   (image-left)
    [73.5318, 51.5014],   # right eye  (image-right)
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)

# MediaPipe FaceMesh indices.  Eye centres are averaged over the same six points
# extract_features.py already uses for EAR, so we do not depend on the iris
# refinement landmarks (468-477) being present in the model variant.
_MP_LEFT_EYE = [362, 385, 387, 263, 373, 380]    # subject's left eye
_MP_RIGHT_EYE = [33, 160, 158, 133, 153, 144]    # subject's right eye
_MP_NOSE_TIP = 1
_MP_MOUTH_L = 61     # outer corners — ArcFace's mouth points are outer, not inner
_MP_MOUTH_R = 291


def _pt(landmarks, idx: int, w: int, h: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def _centre(landmarks, idxs: Sequence[int], w: int, h: int) -> np.ndarray:
    return np.mean([_pt(landmarks, i, w, h) for i in idxs], axis=0).astype(np.float32)


def landmarks_to_5pt(landmarks, w: int, h: int) -> np.ndarray:
    """Five canonical points in pixel coords, ordered by image position.

    Ordering by x rather than by anatomy keeps alignment consistent whether or
    not the incoming feed is mirrored (see rule 4 in the module docstring).
    """
    eye_a = _centre(landmarks, _MP_LEFT_EYE, w, h)
    eye_b = _centre(landmarks, _MP_RIGHT_EYE, w, h)
    mouth_a = _pt(landmarks, _MP_MOUTH_L, w, h)
    mouth_b = _pt(landmarks, _MP_MOUTH_R, w, h)

    eye_l, eye_r = (eye_a, eye_b) if eye_a[0] <= eye_b[0] else (eye_b, eye_a)
    mou_l, mou_r = (mouth_a, mouth_b) if mouth_a[0] <= mouth_b[0] else (mouth_b, mouth_a)

    return np.stack([eye_l, eye_r, _pt(landmarks, _MP_NOSE_TIP, w, h), mou_l, mou_r])


def align_face(bgr: np.ndarray, landmarks) -> Optional[np.ndarray]:
    """Warp the face to ArcFace's canonical 112x112 crop. None if unsolvable."""
    h, w = bgr.shape[:2]
    src = landmarks_to_5pt(landmarks, w, h)
    matrix, _ = cv2.estimateAffinePartial2D(src, _ARCFACE_5PT, method=cv2.LMEDS)
    if matrix is None:
        return None
    return cv2.warpAffine(bgr, matrix, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_LINEAR)


# ── quality gating ───────────────────────────────────────────────────────
@dataclass
class Quality:
    """Why a frame was accepted or rejected — surfaced live to the supervisor."""
    ok: bool
    reasons: List[str] = field(default_factory=list)
    blur: float = 0.0
    face_px: float = 0.0
    brightness: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0

    def as_dict(self) -> dict:
        return {"ok": self.ok, "reasons": self.reasons, "blur": round(self.blur, 1),
                "face_px": round(self.face_px, 1), "brightness": round(self.brightness, 1),
                "yaw": round(self.yaw, 1), "pitch": round(self.pitch, 1)}


# Enrolment gates are strict (a supervisor is standing there to fix it);
# recognition gates are looser (a working cab is dim, dusty and vibrating).
MIN_BLUR_ENROL, MIN_BLUR_PROBE = 45.0, 18.0
MIN_FACE_PX_ENROL, MIN_FACE_PX_PROBE = 110.0, 70.0
BRIGHTNESS_RANGE = (35.0, 235.0)


def assess_quality(bgr: np.ndarray, landmarks, *, strict: bool = True,
                   yaw: float = 0.0, pitch: float = 0.0) -> Quality:
    h, w = bgr.shape[:2]
    pts = landmarks_to_5pt(landmarks, w, h)

    # Eye-to-mouth span is a stabler size proxy than a bounding box.
    face_px = float(np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[4] - pts[1])) / 2.0 * 2.2
    lo = np.clip(pts.min(axis=0) - 20, [0, 0], [w, h]).astype(int)
    hi = np.clip(pts.max(axis=0) + 20, [0, 0], [w, h]).astype(int)
    roi = bgr[lo[1]:hi[1], lo[0]:hi[0]]
    if roi.size == 0:
        return Quality(False, ["face outside frame"])

    grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    brightness = float(grey.mean())

    min_blur = MIN_BLUR_ENROL if strict else MIN_BLUR_PROBE
    min_px = MIN_FACE_PX_ENROL if strict else MIN_FACE_PX_PROBE

    reasons: List[str] = []
    if blur < min_blur:
        reasons.append("too blurry — hold still")
    if face_px < min_px:
        reasons.append("move closer to the camera")
    if brightness < BRIGHTNESS_RANGE[0]:
        reasons.append("too dark — turn on the cab light")
    elif brightness > BRIGHTNESS_RANGE[1]:
        reasons.append("overexposed — avoid direct backlight")

    return Quality(not reasons, reasons, blur, face_px, brightness, yaw, pitch)


# ── embedder ─────────────────────────────────────────────────────────────
def l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


class FaceEmbedder:
    """ArcFace ONNX embedder with mirror-invariant flip TTA.

    `session` may be injected for tests: any callable crop_nchw -> (1, 512).
    """

    def __init__(self, model_path: Optional[Path] = None, *, session=None,
                 auto_download: bool = False) -> None:
        self.available = False
        self.backend = "unavailable"
        self.last_error: Optional[str] = None
        self._session = session
        self._input_name: Optional[str] = None
        self._lock = threading.Lock()

        if session is not None:
            self.available = True
            self.backend = "injected"
            return

        path = Path(model_path or ARCFACE_PATH)
        try:
            import onnxruntime as ort
        except ImportError:
            self.last_error = ("onnxruntime not installed — face ID disabled, "
                               "falling back to manual operator selection")
            print(f"[FaceID] {self.last_error}", flush=True)
            return

        if not path.exists():
            if not auto_download or not download_model(path):
                self.last_error = (f"ArcFace model not found at {path}. Fetch it once with "
                                   f"`python -m brain.faceid --download`, or set SAARTHI_ARCFACE_PATH.")
                print(f"[FaceID] {self.last_error}", flush=True)
                return

        try:
            self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
            self.available = True
            self.backend = "arcface-onnx"
            print(f"[FaceID] ArcFace ready ({path.name}, CPU).", flush=True)
        except Exception as exc:            # noqa: BLE001 — never take the cab down
            self.last_error = f"ONNX load failed: {exc}"
            print(f"[FaceID] {self.last_error}", flush=True)

    @staticmethod
    def _preprocess(crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 127.5
        return np.transpose(rgb, (2, 0, 1))[None, ...]     # NCHW

    def _run(self, nchw: np.ndarray) -> np.ndarray:
        if self._input_name is None:                        # injected test stub
            return np.asarray(self._session(nchw), dtype=np.float32).reshape(-1)
        out = self._session.run(None, {self._input_name: nchw})[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def embed(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Unit-norm 512-d signature, invariant to horizontal mirroring."""
        if not self.available:
            return None
        try:
            with self._lock:                # ORT sessions are not thread-safe
                a = self._run(self._preprocess(crop_bgr))
                b = self._run(self._preprocess(cv2.flip(crop_bgr, 1)))
        except Exception as exc:            # noqa: BLE001
            self.last_error = f"inference failed: {exc}"
            return None
        return l2(l2(a) + l2(b))

    def embed_frame(self, bgr: np.ndarray, landmarks) -> Optional[np.ndarray]:
        """Align then embed, in one step. None if the face cannot be warped."""
        crop = align_face(bgr, landmarks)
        return None if crop is None else self.embed(crop)


def download_model(path: Path = ARCFACE_PATH, url: str = ARCFACE_URL) -> bool:
    """One-off fetch of the ArcFace ONNX (~170 MB). Returns success."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        print(f"[FaceID] downloading ArcFace model from {url} ...", flush=True)
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
        print(f"[FaceID] saved to {path}", flush=True)
        return True
    except Exception as exc:                # noqa: BLE001
        print(f"[FaceID] download failed: {exc}", flush=True)
        return False


if __name__ == "__main__":
    import sys
    if "--download" in sys.argv:
        raise SystemExit(0 if download_model() else 1)
    print(__doc__)
