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

# A face window is only valid for so long. Without this, a face lost to
# sunglasses or a dust mask leaves `latest_features` frozen at the last good
# reading and the pipeline keeps reporting it forever — the cab confidently
# shows ALERT/OK while the operator falls asleep. Failing loudly beats that.
FEATURES_MAX_AGE_SEC = 4.0   # ~2 windows at the browser's 5 fps

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"


class FaceObserverMixin:
    """Lets face-ID / enrolment reuse the frame the tracker has already decoded
    and landmarked, instead of standing up a second camera pipeline.

    The observer is called on the tracker thread, so it must be quick and it
    must never raise — an exception here would take the fatigue pipeline down
    with it, which is a safety regression for a cosmetic feature.
    """

    face_observer = None

    def set_face_observer(self, fn) -> None:
        self.face_observer = fn

    def _notify_face(self, frame, landmarks, has_face: bool,
                     ear: float = 0.0, yaw: float = 0.0, pitch: float = 0.0) -> None:
        fn = self.face_observer
        if fn is None:
            return
        try:
            fn(frame, landmarks, has_face=has_face, ear=float(ear or 0.0),
               yaw=float(yaw or 0.0), pitch=float(pitch or 0.0))
        except Exception as exc:                      # noqa: BLE001
            print(f"[Camera] face observer error (ignored): {exc}", flush=True)


class BackgroundCameraTracker(FaceObserverMixin):
    """Runs cv2.VideoCapture and MediaPipe FaceLandmarker in a daemon thread."""

    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.cap = None
        
        self.latest_features: Optional[Dict[str, float]] = None
        self._features_ts = 0.0          # when latest_features was last written
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
                # Sanity-check the baseline before accepting it.
                # These limits detect a drowsy/eyes-closed calibration state.
                _pc_mean  = sum(r.get("perclos", 0.0) for r in rows) / len(rows)
                _ear_mean = sum(r.get("ear_mean", 0.0) for r in rows) / len(rows)
                _br_mean  = sum(r.get("blink_rate", 0.0) for r in rows) / len(rows)

                print(f"[Calibration] Sanity check — "
                      f"perclos_mean={_pc_mean:.3f}  ear_mean={_ear_mean:.3f}  blink_rate_mean={_br_mean:.1f}",
                      flush=True)

                _bad = []
                if _pc_mean > 0.20:
                    _bad.append(f"perclos mean={_pc_mean:.3f} > 0.20 (eyes were mostly closed!")
                if _ear_mean < 0.20:
                    _bad.append(f"ear_mean mean={_ear_mean:.3f} < 0.20 (eyes too closed)")

                if not _bad:
                    print(f"[Calibration] SANITY OK — {len(rows)} valid alert windows accepted.",
                          flush=True)
                    return rows

                # Bad baseline — show exactly what was wrong and retry.
                _SEP2 = "!" * 64
                print(f"\n{_SEP2}", flush=True)
                print("!  CALIBRATION REJECTED — BASELINE LOOKS DROWSY              !", flush=True)
                for _b in _bad:
                    print(f"!    >> {_b}", flush=True)
                print("!                                                               !", flush=True)
                print("!  During calibration you must look RELAXED but AWAKE:          !", flush=True)
                print("!    • Keep eyes OPEN and blinking NORMALLY (not staring)       !", flush=True)
                print("!    • Face camera squarely, good light, no strong backlight    !", flush=True)
                print("!    • Do NOT hold your eyes extra-wide (causes low perclos     !", flush=True)
                print("!      but then live perclos looks high by comparison)          !", flush=True)
                print(f"{_SEP2}\n", flush=True)
                print("[Calibration] Retrying in 3 s...  (Ctrl-C to abort)\n", flush=True)
                time.sleep(3)
                # loop repeats

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
        """Latest face features, or None once they are too old to trust.

        Returning stale features is worse than returning nothing: the caller
        cannot tell the difference, so it reports an old healthy reading as if
        it were current.
        """
        with self._lock:
            if self.latest_features is None:
                return None
            if time.time() - self._features_ts > FEATURES_MAX_AGE_SEC:
                return None
            return self.latest_features

    def features_age(self) -> float:
        """Seconds since the last valid face window (inf if there never was one)."""
        with self._lock:
            if self.latest_features is None:
                return float("inf")
            return time.time() - self._features_ts

    def face_signal_ok(self) -> bool:
        return self.features_age() <= FEATURES_MAX_AGE_SEC

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
                        self._notify_face(frame, landmarks, True,
                                          frame_data['ear'], yaw, pitch)
                    else:
                        self._notify_face(frame, None, False)

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
                                "is_warming_up": float(window_idx < 3),
                            }
                            with self._lock:
                                self.latest_features = candidate
                                self._features_ts = time.time()
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


class RemoteCameraTracker(FaceObserverMixin):
    """
    Processes JPEG frames sent from the browser via WebSocket.
    Same interface as BackgroundCameraTracker but no local webcam needed.
    Frames are fed in via feed_frame(jpeg_bytes).
    """

    def __init__(self) -> None:
        self.latest_features: Optional[Dict[str, float]] = None
        self._features_ts = 0.0          # when latest_features was last written
        self._last_frame_ts = 0.0
        self._lock = threading.Lock()

        self._preview_lock = threading.Lock()
        self._fatigue_overlay: dict = {"p_pct": 0, "decision": "—", "severity": "OK"}

        # Placeholder frame
        _ph = np.full((240, 320, 3), 40, dtype=np.uint8)
        cv2.putText(_ph, "Waiting for browser cam...", (18, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
        _, _ph_buf = cv2.imencode('.jpg', _ph)
        self._ph_bytes = _ph_buf.tobytes()
        self._latest_annotated_frame: bytes = self._ph_bytes

        self.last_error = None
        self._landmarker = None
        self._frame_idx = 0
        self._window_idx = 0
        self._fps = 5.0  # approximate incoming frame rate from browser

        self._frames_per_window = max(1, int(WINDOW_SEC * self._fps))
        self._frames_per_lookback = int(PERCLOS_LOOKBACK_SEC * self._fps)
        self._trailing_buffer: deque = deque(maxlen=self._frames_per_lookback)
        self._window_frames: list = []

        self.is_available = False

        if HAS_MEDIAPIPE:
            try:
                if not os.path.exists(MODEL_PATH):
                    print(f"[RemoteCamera] Downloading MediaPipe model to {MODEL_PATH}...")
                    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

                BaseOptions = mp.tasks.BaseOptions
                FaceLandmarker = mp.tasks.vision.FaceLandmarker
                FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode

                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=MODEL_PATH),
                    running_mode=VisionRunningMode.VIDEO,
                    num_faces=1)

                self._landmarker = FaceLandmarker.create_from_options(options)
                self.is_available = True
                self.last_error = "Init successful"
                print("[RemoteCamera] MediaPipe ready for browser frames.", flush=True)
            except Exception as e:
                self.last_error = f"Init error: {e}"
                print(f"[RemoteCamera] Failed to init MediaPipe: {e}", flush=True)

    def feed_frame(self, jpeg_bytes: bytes) -> None:
        """Decode a JPEG from the browser, run MediaPipe, extract features."""
        if not self.is_available or self._landmarker is None:
            return

        try:
            self._last_frame_ts = time.time()
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return

            self._frame_idx += 1
            time_sec = self._frame_idx / self._fps
            timestamp_ms = int(time_sec * 1000)

            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            results = self._landmarker.detect_for_video(mp_image, timestamp_ms)

            # ── Annotated preview ──
            if results.face_landmarks and len(results.face_landmarks) > 0:
                _lm = results.face_landmarks[0]
                _ear = calculate_ear(_lm, width, height)
            else:
                _lm = None
                _ear = 0.0

            ann = self._annotate_frame(frame, _lm, _ear, height, width)
            _ok, _buf = cv2.imencode('.jpg', ann, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if _ok:
                with self._preview_lock:
                    self._latest_annotated_frame = _buf.tobytes()

            # ── Feature extraction (every frame since browser sends ~5 fps) ──
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
                self._notify_face(frame, landmarks, True,
                                  frame_data['ear'], yaw, pitch)
            else:
                self._notify_face(frame, None, False)

            self._trailing_buffer.append(frame_data)
            self._window_frames.append(frame_data)

            if len(self._window_frames) >= self._frames_per_window / max(1, FRAME_SAMPLE_RATE):
                row = aggregate_window(
                    self._window_frames, self._trailing_buffer,
                    fold=0, subject_id="remote", video_name="browser",
                    window_idx=self._window_idx, label=0
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
                        "is_warming_up": float(self._window_idx < 3),
                    }
                    with self._lock:
                        self.latest_features = candidate
                        self._features_ts = time.time()
                    print(
                        f"[RemoteCamera win {self._window_idx}] VALID "
                        f"EAR={row['ear_mean']:.3f} "
                        f"PERCLOS={row['perclos']:.3f} "
                        f"MAR={row['mar_mean']:.3f}",
                        flush=True
                    )
                self._window_idx += 1
                self._window_frames = []
                self.last_error = "Frames processing fine"

        except Exception as e:
            self.last_error = f"Frame error: {e}"
            print(f"[RemoteCamera] Frame processing error: {e}", flush=True)

    def get_latest_features(self) -> Optional[Dict[str, float]]:
        """Latest face features, or None once they are too old to trust.

        Returning stale features is worse than returning nothing: the caller
        cannot tell the difference, so it reports an old healthy reading as if
        it were current.
        """
        with self._lock:
            if self.latest_features is None:
                return None
            if time.time() - self._features_ts > FEATURES_MAX_AGE_SEC:
                return None
            return self.latest_features

    def features_age(self) -> float:
        """Seconds since the last valid face window (inf if there never was one)."""
        with self._lock:
            if self.latest_features is None:
                return float("inf")
            return time.time() - self._features_ts

    def face_signal_ok(self) -> bool:
        return self.features_age() <= FEATURES_MAX_AGE_SEC

    def get_latest_annotated_frame(self) -> Optional[bytes]:
        if time.time() - self._last_frame_ts > 2.0:
            return self._ph_bytes
        with self._preview_lock:
            return self._latest_annotated_frame

    def set_fatigue_info(self, fatigue: dict) -> None:
        with self._preview_lock:
            self._fatigue_overlay = {
                "p_pct":    int(fatigue.get("p", 0) * 100),
                "decision": fatigue.get("decision", "—"),
                "severity": fatigue.get("severity", "OK"),
            }

    def _annotate_frame(self, frame, landmarks, ear, height, width):
        """Draw face mesh + overlay onto a BGR copy of frame."""
        img = frame.copy()

        if landmarks is not None:
            eye_open = ear >= EAR_CLOSED_THRESH
            mesh_col = (0, 200, 0)
            eye_col = (0, 200, 0) if eye_open else (30, 30, 220)
            eye_label = f"EAR {ear:.3f}  EYES {'OPEN' if eye_open else 'CLOSED'}"

            for lm in landmarks:
                cv2.circle(img, (int(lm.x * width), int(lm.y * height)),
                           1, mesh_col, -1)
            for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                lm = landmarks[idx]
                cv2.circle(img, (int(lm.x * width), int(lm.y * height)),
                           4, eye_col, -1)
        else:
            eye_col = (140, 140, 140)
            eye_label = "NO FACE DETECTED"

        with self._preview_lock:
            ov = dict(self._fatigue_overlay)

        p_pct = ov["p_pct"]
        decision = ov["decision"]
        severity = ov["severity"]
        fat_col = ((30, 30, 220) if severity == "STRONG"
                   else (30, 130, 255) if decision == "AT-RISK"
                   else (30, 200, 30))

        cv2.rectangle(img, (0, 0), (width, 30), (18, 18, 18), -1)
        cv2.putText(img, eye_label, (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, eye_col, 1, cv2.LINE_AA)

        cv2.rectangle(img, (0, height - 36), (width, height), (18, 18, 18), -1)
        cv2.putText(img, f"FATIGUE {p_pct}%  {decision}  {severity}",
                    (8, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, fat_col, 1, cv2.LINE_AA)

        return img

    def stop(self):
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None

