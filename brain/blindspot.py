"""
brain/blindspot.py — YOLO11-nano blind-spot detection running on a looped video clip.
"""

from __future__ import annotations

import os
import time
import threading
from typing import Optional
import cv2
import numpy as np

# Thresholds for the person box height / frame height
GREEN_MAX = 0.25
AMBER_MAX = 0.50

COLOR_GREEN = (0, 200, 0)
COLOR_AMBER = (0, 180, 255)
COLOR_RED   = (0, 0, 255)

CLIP_PATH = "data/blindspot_clip.mp4"
MODEL_NAME = "yolo11n.pt"

class BackgroundBlindspotTracker:
    def __init__(self, clip_path: str = CLIP_PATH):
        self.clip_path = clip_path
        self._lock = threading.Lock()
        
        self.latest_zone = 0  # 0: GREEN, 1: AMBER, 2: RED
        self.is_reversing_active = False  # Set by trigger, cleared when clip ends
        
        self._preview_lock = threading.Lock()
        
        # Placeholder frame
        _ph = np.full((240, 320, 3), 40, dtype=np.uint8)
        cv2.putText(_ph, "YOLO loading...", (80, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 1, cv2.LINE_AA)
        _, _ph_buf = cv2.imencode('.jpg', _ph)
        self._latest_annotated_frame: bytes = _ph_buf.tobytes()

        self._running = False
        self._thread = None
        self._model = None
        self._cap = None
        
        # We will use this flag to signal the loop to reset the clip
        self._trigger_restart = False
        self._last_zone_was_red_or_amber = False
        self.is_available = True

    def start(self):
        if not os.path.exists(self.clip_path):
            print(f"[BlindSpot] Clip not found: {self.clip_path}. Falling back to simulated zone.")
            self.is_available = False
            return
            
        try:
            from ultralytics import YOLO
            self._model = YOLO(MODEL_NAME)
        except Exception as e:
            print(f"[BlindSpot] Failed to load YOLO: {e}. Falling back to simulated zone.")
            self.is_available = False
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()

    def trigger_blindspot(self):
        """Restarts the clip from frame 0 and engages reversing."""
        with self._lock:
            self._trigger_restart = True

    def get_latest_zone(self) -> int:
        with self._lock:
            return self.latest_zone

    def get_is_reversing(self) -> bool:
        with self._lock:
            return self.is_reversing_active

    def get_latest_annotated_frame(self) -> Optional[bytes]:
        with self._preview_lock:
            return self._latest_annotated_frame

    def _run_loop(self):
        self._cap = cv2.VideoCapture(self.clip_path)
        if not self._cap.isOpened():
            self.is_available = False
            return

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
            
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        clip_start_time = time.time()
        
        while self._running:
            # Check for manual trigger
            with self._lock:
                if self._trigger_restart:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    clip_start_time = time.time()
                    self.is_reversing_active = True
                    self._trigger_restart = False
                    self._last_zone_was_red_or_amber = False
                    print("[BlindSpot] Clip restarted. Reversing=YES.", flush=True)

            # Keep playback smooth to real-time by skipping frames if YOLO is slow
            expected_frame = int((time.time() - clip_start_time) * fps)
            
            if expected_frame >= total_frames:
                # Clip ended, loop it
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                clip_start_time = time.time()
                expected_frame = 0
                with self._lock:
                    if self.is_reversing_active:
                        print("[BlindSpot] Clip ended. Reversing=NO.", flush=True)
                        self.is_reversing_active = False
                        self._last_zone_was_red_or_amber = False
                        
            current_frame = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame < expected_frame:
                # We are behind real-time, skip ahead
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, expected_frame)
            
            ret, frame = self._cap.read()
            if not ret:
                # Sometimes set() goes past the end of video
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                clip_start_time = time.time()
                with self._lock:
                    if self.is_reversing_active:
                        self.is_reversing_active = False
                continue

            h, w = frame.shape[:2]
            
            # Run YOLO
            results = self._model(frame, classes=[0], verbose=False)[0]
            
            max_box_h = 0.0
            best_box = None
            
            for box in results.boxes:
                if int(box.cls) == 0:
                    x1, y1, x2, y2 = box.xyxy[0]
                    box_h = float((y2 - y1) / h)
                    if box_h > max_box_h:
                        max_box_h = box_h
                        best_box = (int(x1), int(y1), int(x2), int(y2))

            # Map to zone
            if max_box_h > AMBER_MAX:
                zone, color = 2, COLOR_RED
                label = "RED"
            elif max_box_h > GREEN_MAX:
                zone, color = 1, COLOR_AMBER
                label = "AMBER"
            else:
                zone, color = 0, COLOR_GREEN
                label = "GREEN"
                
            # Auto-clear reversing if we return to GREEN after being in RED/AMBER
            with self._lock:
                self.latest_zone = zone
                if self.is_reversing_active:
                    if zone >= 1:
                        self._last_zone_was_red_or_amber = True
                    elif zone == 0 and self._last_zone_was_red_or_amber:
                        print("[BlindSpot] Returned to GREEN. Auto-clearing reversing.", flush=True)
                        self.is_reversing_active = False
                        self._last_zone_was_red_or_amber = False

            # Annotate frame
            ann_frame = frame.copy()
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                cv2.rectangle(ann_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(ann_frame, f"person {max_box_h:.2f} [{label}]", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            else:
                cv2.putText(ann_frame, "NO PERSON DETECTED", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GREEN, 2, cv2.LINE_AA)
                            
            if self.is_reversing_active:
                cv2.putText(ann_frame, "REVERSING", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # Resize to save bandwidth on MJPEG
            ann_frame_small = cv2.resize(ann_frame, (640, 360))
            _ok, _buf = cv2.imencode('.jpg', ann_frame_small, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if _ok:
                with self._preview_lock:
                    self._latest_annotated_frame = _buf.tobytes()

            # Small sleep to prevent tight loop if processing is faster than real-time
            process_time = time.time() - clip_start_time - (expected_frame / fps)
            sleep_time = max(0.01, (1.0 / fps) - process_time)  # At least 10ms to yield CPU
            time.sleep(sleep_time)
