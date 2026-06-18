"""
brain/live_camera.py — Background thread for live webcam fatigue feature extraction.
"""

from __future__ import annotations

import os
import time
import urllib.request
import threading
from typing import Dict, List, Optional
from collections import deque
import cv2
import numpy as np

# We ensure paths.py is imported to fix sys.path before trying to import offline pipeline
from . import paths

try:
    import mediapipe as mp
    from extract_features import (
        calculate_ear,
        calculate_mar,
        calculate_head_pose,
        aggregate_window,
        WINDOW_SEC,
        PERCLOS_LOOKBACK_SEC,
        FRAME_SAMPLE_RATE,
        EAR_CLOSED_THRESH,
        LEFT_EYE_INDICES,
        RIGHT_EYE_INDICES,
    )
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"


class BackgroundCameraTracker:
    """Runs cv2.VideoCapture and MediaPipe FaceLandmarker in a daemon thread."""

    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.cap = None
        
        self.latest_features: Optional[Dict[str, float]] = None
        self._lock = threading.Lock()

        # Preview state — written by the camera thread, read by /video_feed.
        # Completely separate from the feature pipeline; never affects fatigue math.
        self._preview_lock = threading.Lock()
        self._fatigue_overlay: dict = {"p_pct": 0, "decision": "—", "severity": "OK"}
        # Pre-bake a "starting" placeholder so /video_feed has something to yield
        # immediately, before the camera thread produces its first real frame.
        _ph = np.full((240, 320, 3), 40, dtype=np.uint8)
        cv2.putText(_ph, "Camera starting...", (28, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 1, cv2.LINE_AA)
        _, _ph_buf = cv2.imencode('.jpg', _ph)
        self._latest_annotated_frame: bytes = _ph_buf.tobytes()

        self._running = False
        self._thread = None
        
        # Make sure model exists
        if HAS_MEDIAPIPE and not os.path.exists(MODEL_PATH):
            print(f"Downloading MediaPipe model to {MODEL_PATH}...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    def _init_camera(self) -> bool:
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
        return self.cap.isOpened()

    def _calibrate_once(self, seconds: float) -> List[Dict]:
        """
        Single calibration pass: read `seconds` of frames, return only the windows
        where a face was detected (NaN windows are discarded).  May return [].
        """
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        frames_per_window = max(1, int(WINDOW_SEC * fps))
        frames_per_lookback = int(PERCLOS_LOOKBACK_SEC * fps)

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1)

        trailing_buffer = deque(maxlen=frames_per_lookback)
        window_frames = []
        calibration_rows = []
        window_idx = 0
        frame_idx = 0

        try:
            with FaceLandmarker.create_from_options(options) as landmarker:
                start_time = time.time()
                while time.time() - start_time < seconds:
                    ret, frame = self.cap.read()
                    if not ret:
                        time.sleep(0.01)
                        continue

                    frame_idx += 1
                    time_sec = frame_idx / fps
                    timestamp_ms = int(time_sec * 1000)

                    height, width, _ = frame.shape
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                    results = landmarker.detect_for_video(mp_image, timestamp_ms)

                    # ── Live preview during calibration ────────────────────────
                    try:
                        if results.face_landmarks and len(results.face_landmarks) > 0:
                            _lm  = results.face_landmarks[0]
                            _ear = calculate_ear(_lm, width, height)
                        else:
                            _lm  = None
                            _ear = 0.0
                        _ann = self._annotate_frame(frame, _lm, _ear, height, width)
                        
                        # Add a "CALIBRATING" overlay
                        cv2.putText(_ann, "CALIBRATING BASELINE...", (width // 2 - 100, height // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                                    
                        _ok, _buf = cv2.imencode('.jpg', _ann, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if _ok:
                            with self._preview_lock:
                                self._latest_annotated_frame = _buf.tobytes()
                    except Exception as _prev_exc:
                        print(f"[Camera] Preview encode error: {_prev_exc}", flush=True)
                    # ───────────────────────────────────────────────────────────

                    if frame_idx % FRAME_SAMPLE_RATE == 0:
                        frame_data = {
                            'time_sec': time_sec, 'has_face': False,
                            'ear': np.nan, 'mar': np.nan, 'pitch': np.nan,
                            'yaw': np.nan, 'roll': np.nan
                        }

                        if results.face_landmarks and len(results.face_landmarks) > 0:
                            landmarks = results.face_landmarks[0]
                            frame_data['has_face'] = True
                            frame_data['ear'] = calculate_ear(landmarks, width, height)
                            frame_data['mar'] = calculate_mar(landmarks, width, height)
                            pitch, yaw, roll = calculate_head_pose(landmarks, width, height)
                            frame_data['pitch'] = pitch
                            frame_data['yaw'] = yaw
                            frame_data['roll'] = roll

                        trailing_buffer.append(frame_data)
                        window_frames.append(frame_data)

                        if len(window_frames) >= frames_per_window / FRAME_SAMPLE_RATE:
                            row = aggregate_window(
                                window_frames, trailing_buffer,
                                fold=0, subject_id="live", video_name="live",
                                window_idx=window_idx, label=0
                            )
                            if not np.isnan(row.get("ear_mean", np.nan)):
                                calibration_rows.append(row)
                                print(f"  [Calibration win {window_idx}] EAR={row['ear_mean']:.3f} "
                                      f"PERCLOS={row['perclos']:.3f} valid=YES", flush=True)
                            else:
                                print(f"  [Calibration win {window_idx}] no face detected — skipped",
                                      flush=True)
                            window_idx += 1
                            window_frames = []
        except Exception as e:
            print(f"[LiveCamera] Error during calibration: {e}", flush=True)

        return calibration_rows

    def calibrate(self, seconds: float = 25.0) -> List[Dict]:
        """
        Phase-1 Calibration: captures an ALERT baseline from the operator.

        Retries indefinitely until at least one valid face window is collected.
        A synthetic fallback is NEVER silently accepted — if the face is not
        detected the operator is prompted to adjust their position and the
        calibration window is repeated.  Ctrl-C aborts.

        Returns a non-empty list of feature-window dicts for FatigueEngine.
        Returns [] only when MediaPipe / camera is unavailable (caller's problem).
        """
        if not HAS_MEDIAPIPE:
            print("[LiveCamera] MediaPipe not available. Skipping calibration.")
            return []

        if not self._init_camera():
            print("[LiveCamera] Failed to open webcam. Skipping calibration.")
            return []

        attempt = 0
        while True:
            attempt += 1
            print(f"\n[Calibration] Attempt {attempt} — {seconds:.0f}s baseline capture. "
                  f"Look forward & stay alert!", flush=True)

            rows = self._calibrate_once(seconds)

            if rows:
                print(f"[Calibration] SUCCESS — {len(rows)} valid face windows collected.",
                      flush=True)
                return rows

            # No face detected in this pass — refuse synthetic, force a retry.
            _SEP = "!" * 64
            print(f"\n{_SEP}", flush=True)
            print("!  CALIBRATION FAILED — NO FACE DETECTED                          !", flush=True)
            print("!  A synthetic baseline has been REFUSED.  The model would         !", flush=True)
            print("!  misread your actual face and fatigue readings would be wrong.   !", flush=True)
            print("!                                                                   !", flush=True)
            print("!  To fix this:                                                     !", flush=True)
            print("!    1. Sit squarely in front of the camera                        !", flush=True)
            print("!    2. Ensure your face is well-lit (avoid strong backlight)      !", flush=True)
            print("!    3. Keep your eyes open and look toward the screen             !", flush=True)
            print("!    4. Check that no other app has the camera locked (Teams etc.) !", flush=True)
            print(f"{_SEP}\n", flush=True)
            print(f"[Calibration] Retrying in 3 s...  (Ctrl-C to abort)\n", flush=True)
            time.sleep(3)

    def start(self):
        if not HAS_MEDIAPIPE or not self._init_camera():
            print("[LiveCamera] Tracker cannot start.")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_latest_features(self) -> Optional[Dict[str, float]]:
        with self._lock:
            return self.latest_features

    # ── Preview API ─────────────────────────────────────────────────────────

    def set_fatigue_info(self, fatigue: dict) -> None:
        """Called each brain tick to keep the on-frame fatigue overlay current."""
        with self._preview_lock:
            self._fatigue_overlay = {
                "p_pct":    int(fatigue.get("p", 0) * 100),
                "decision": fatigue.get("decision", "—"),
                "severity": fatigue.get("severity", "OK"),
            }

    def get_latest_annotated_frame(self) -> Optional[bytes]:
        """Return the most recent JPEG bytes (annotated), or None if not ready."""
        with self._preview_lock:
            return self._latest_annotated_frame

    def _annotate_frame(
        self,
        frame: np.ndarray,
        landmarks,           # results.face_landmarks[0] or None
        ear: float,
        height: int,
        width: int,
    ) -> np.ndarray:
        """
        Draw face mesh + EYES status + fatigue overlay onto a BGR copy of frame.
        Display-only — never touches features, calibration, or fatigue math.
        """
        img = frame.copy()

        if landmarks is not None:
            eye_open  = ear >= EAR_CLOSED_THRESH
            mesh_col  = (0, 200, 0)
            eye_col   = (0, 200, 0) if eye_open else (30, 30, 220)
            eye_label = f"EAR {ear:.3f}  EYES {'OPEN' if eye_open else 'CLOSED'}"

            # Full face mesh — 1 px green dots
            for lm in landmarks:
                cv2.circle(img, (int(lm.x * width), int(lm.y * height)),
                           1, mesh_col, -1)
            # Eye landmarks — larger, coloured to show open/closed state
            for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                lm = landmarks[idx]
                cv2.circle(img, (int(lm.x * width), int(lm.y * height)),
                           4, eye_col, -1)
        else:
            eye_col   = (140, 140, 140)
            eye_label = "NO FACE DETECTED"

        # Read fatigue overlay atomically (brief lock)
        with self._preview_lock:
            ov = dict(self._fatigue_overlay)

        p_pct    = ov["p_pct"]
        decision = ov["decision"]
        severity = ov["severity"]
        fat_col  = ((30, 30, 220) if severity == "STRONG"
                    else (30, 130, 255) if decision == "AT-RISK"
                    else (30, 200, 30))

        # Top bar — eye status
        cv2.rectangle(img, (0, 0), (width, 30), (18, 18, 18), -1)
        cv2.putText(img, eye_label, (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, eye_col, 1, cv2.LINE_AA)

        # Bottom bar — fatigue summary
        cv2.rectangle(img, (0, height - 36), (width, height), (18, 18, 18), -1)
        cv2.putText(img, f"FATIGUE {p_pct}%  {decision}  {severity}",
                    (8, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, fat_col, 1, cv2.LINE_AA)

        return img

    # ────────────────────────────────────────────────────────────────────────

    def _run_loop(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        frames_per_window = max(1, int(WINDOW_SEC * fps))
        frames_per_lookback = int(PERCLOS_LOOKBACK_SEC * fps)

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1)

        trailing_buffer = deque(maxlen=frames_per_lookback)
        window_frames = []
        window_idx = 0
        frame_idx = 0

        with FaceLandmarker.create_from_options(options) as landmarker:
            while self._running:
                try:
                    ret, frame = self.cap.read()
                    if not ret:
                        time.sleep(0.01)
                        continue

                    frame_idx += 1
                    time_sec = frame_idx / fps
                    timestamp_ms = int(time_sec * 1000)

                    height, width, _ = frame.shape
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                    results = landmarker.detect_for_video(mp_image, timestamp_ms)
                except Exception as _exc:
                    print(f"[Camera] Frame error (camera may be disconnected): {_exc}", flush=True)
                    time.sleep(0.1)
                    continue

                # ── Live preview (display-only) ──────────────────────────────
                # Isolated in its own try/except so any annotation or encode error
                # is logged and skipped — it must never crash the feature thread.
                try:
                    if results.face_landmarks and len(results.face_landmarks) > 0:
                        _lm  = results.face_landmarks[0]
                        _ear = calculate_ear(_lm, width, height)
                    else:
                        _lm  = None
                        _ear = 0.0
                    _ann = self._annotate_frame(frame, _lm, _ear, height, width)
                    _ok, _buf = cv2.imencode('.jpg', _ann, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if _ok:
                        with self._preview_lock:
                            self._latest_annotated_frame = _buf.tobytes()
                except Exception as _prev_exc:
                    print(f"[Camera] Preview encode error: {_prev_exc}", flush=True)
                # ─────────────────────────────────────────────────────────────

                if frame_idx % FRAME_SAMPLE_RATE == 0:
                    frame_data = {
                        'time_sec': time_sec, 'has_face': False,
                        'ear': np.nan, 'mar': np.nan, 'pitch': np.nan,
                        'yaw': np.nan, 'roll': np.nan
                    }

                    if results.face_landmarks and len(results.face_landmarks) > 0:
                        landmarks = results.face_landmarks[0]
                        frame_data['has_face'] = True
                        frame_data['ear'] = calculate_ear(landmarks, width, height)
                        frame_data['mar'] = calculate_mar(landmarks, width, height)
                        pitch, yaw, roll = calculate_head_pose(landmarks, width, height)
                        frame_data['pitch'] = pitch
                        frame_data['yaw'] = yaw
                        frame_data['roll'] = roll

                    trailing_buffer.append(frame_data)
                    window_frames.append(frame_data)

                    if len(window_frames) >= frames_per_window / FRAME_SAMPLE_RATE:
                        row = aggregate_window(
                            window_frames, trailing_buffer,
                            fold=0, subject_id="live", video_name="live",
                            window_idx=window_idx, label=0
                        )
                        feats_valid = not np.isnan(row.get("ear_mean", np.nan))
                        if feats_valid:
                            candidate = {
                                "ear_mean":   row["ear_mean"],
                                "ear_min":    row["ear_min"],
                                "ear_std":    row["ear_std"],
                                "perclos":    row["perclos"],
                                "blink_rate": row["blink_rate"],
                                "mar_mean":   row["mar_mean"],
                                "mar_max":    row["mar_max"],
                                "is_yawn":    row["is_yawn"],
                                "pitch_mean": row["pitch_mean"],
                                "yaw_mean":   row["yaw_mean"],
                                "roll_mean":  row["roll_mean"],
                                "pitch_std":  row["pitch_std"],
                            }
                            with self._lock:
                                self.latest_features = candidate
                            print(
                                f"[Camera win {window_idx}] VALID "
                                f"EAR={row['ear_mean']:.3f} "
                                f"PERCLOS={row['perclos']:.3f} "
                                f"MAR={row['mar_mean']:.3f} "
                                f"pitch={row['pitch_mean']:.1f}",
                                flush=True
                            )
                        else:
                            # Face not detected — keep last valid features; log the miss.
                            print(
                                f"[Camera win {window_idx}] NO FACE — keeping previous features",
                                flush=True
                            )
                        window_idx += 1
                        window_frames = []
