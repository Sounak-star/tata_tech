"""
test_webcam.py  -  SAARTHI Live Inference Loop
==============================================
Live camera inference using exactly the same feature extraction logic
as the offline training (extract_features.py) to prevent train/serve skew.

Phase 1: 20-second calibration to capture baseline.
Phase 2: Live monitoring with cv2 UI overlays.
"""

import os
import cv2
import time
import urllib.request
import numpy as np
from collections import deque
import mediapipe as mp

# REUSE: Import exact feature logic from the offline extractor
from extract_features import (
    calculate_ear,
    calculate_mar,
    calculate_head_pose,
    aggregate_window,
    WINDOW_SEC,
    PERCLOS_LOOKBACK_SEC,
    FRAME_SAMPLE_RATE
)

from fatigue_monitor import calibrate, FatigueMonitor, LABEL_AT_RISK
from reason_card import init as init_reason_card, get_alert_reasons

# ==============================================================================
# CONFIG
# ==============================================================================
CALIBRATION_SEC = 25.0

# IMPORTANT: The 30-60s smoothing buffer was tuned for the offline metric.
# In a live demo, waiting 15s for the banner to turn red feels broken.
# Set this SHORT for the responsive demo (~3 windows = 4.5s of closed eyes).
# Set LONG for stable deployment.
SMOOTH_WINDOWS = 3 

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"

# ==============================================================================
# MAIN LIVE LOOP
# ==============================================================================
def main():
    # 1. Ensure MediaPipe model exists
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe face landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Attempt to get real FPS, fallback to 30.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    frames_per_window = max(1, int(WINDOW_SEC * fps))
    frames_per_lookback = int(PERCLOS_LOOKBACK_SEC * fps)

    # Initialize MediaPipe FaceLandmarker
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1)

    # State for exact feature extraction recreation
    trailing_buffer = deque(maxlen=frames_per_lookback)
    window_frames = []
    window_idx = 0
    frame_idx = 0

    # Application state
    calibration_rows = []
    is_calibrating = True
    calibration_start_time = None
    monitor = None
    baseline = None

    # UI state
    current_decision = "ALERT"
    current_severity = "OK"
    current_p = 0.0
    current_reasons = []

    last_time = time.time()
    
    print("Starting SAARTHI Live Loop...")
    print(f"Extraction settings: Window={WINDOW_SEC}s, SampleRate={FRAME_SAMPLE_RATE}")
    
    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            now = time.time()
            live_fps = 1.0 / (now - last_time) if now - last_time > 0 else fps
            last_time = now
            
            if calibration_start_time is None:
                calibration_start_time = time.time()

            # Mirror frame for intuitive UI
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            
            # The time_sec calculation mirrors extract_features exactly
            time_sec = frame_idx / fps
            timestamp_ms = int(time_sec * 1000)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Process frames exactly as offline pipeline
            if frame_idx % FRAME_SAMPLE_RATE == 0:
                frame_data = {
                    'time_sec': time_sec,
                    'has_face': False,
                    'ear': np.nan,
                    'mar': np.nan,
                    'pitch': np.nan,
                    'yaw': np.nan,
                    'roll': np.nan
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
                    # Aggregate using exact offline logic
                    row = aggregate_window(
                        window_frames, trailing_buffer, 
                        fold=0, subject_id="live", video_name="live", 
                        window_idx=window_idx, label=0
                    )
                    window_idx += 1
                    window_frames = []
                    
                    if is_calibrating:
                        calibration_rows.append(row)
                    else:
                        # 1.5s window ready -> Feed the model
                        res = monitor.update(row)
                        current_decision = res["decision"]
                        current_severity = res["severity"]
                        current_p = res["p_at_risk"]
                        
                        # Generate UI string from SHAP
                        if current_decision == LABEL_AT_RISK:
                            current_reasons = get_alert_reasons(row, baseline, top_n=1)
                        else:
                            current_reasons = []
                            
                        # Terminal Logging Gate Check
                        reason_str = current_reasons[0] if current_reasons else "Normal"
                        print(f"Win {window_idx:04d} | State: {current_decision:7s} | "
                              f"Risk: {current_p:5.1%} | Severity: {current_severity:6s} | "
                              f"Reason: {reason_str}")
            
            # ==============================================================================
            # UI OVERLAYS
            # ==============================================================================
            overlay = frame.copy()
            elapsed_calib = time.time() - calibration_start_time
            
            if is_calibrating:
                rem = max(0, int(CALIBRATION_SEC - elapsed_calib))
                if elapsed_calib >= CALIBRATION_SEC and len(calibration_rows) > 5:
                    is_calibrating = False
                    print(f"\n[Calibration] Complete. Captured {len(calibration_rows)} alert windows.")
                    baseline = calibrate(calibration_rows)
                    monitor = FatigueMonitor(baseline, smooth_window=SMOOTH_WINDOWS)
                    init_reason_card(monitor)
                    print("[Live] Starting active monitoring...")
                else:
                    msg = f"CALIBRATING ({rem}s) - Look forward & stay alert"
                    cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                    cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                # Banner color mapping (BGR)
                color = (0, 200, 0) # OK = green
                if current_severity == "MILD":
                    color = (0, 165, 255) # MILD = amber
                elif current_severity == "STRONG":
                    color = (0, 0, 255) # STRONG = red
                
                cv2.rectangle(overlay, (0, 0), (width, 100), color, -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                # Primary status text
                cv2.putText(frame, f"State: {current_decision} ({current_severity})", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                
                # Probability
                cv2.putText(frame, f"Risk: {current_p:.0%}", 
                            (width - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                
                # Reason card below banner
                if current_reasons:
                    cv2.putText(frame, f"Why: {current_reasons[0]}", 
                                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
            # Live FPS in corner
            cv2.putText(frame, f"FPS: {live_fps:.1f}", 
                        (width - 150, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("SAARTHI Live Inference", frame)
            
            if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
