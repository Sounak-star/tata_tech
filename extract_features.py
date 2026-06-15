import os
import cv2
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
from collections import deque

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATASET_ROOT = "c:/Users/noelb/tata1"
OUTPUT_CSV = "raw_features_full.csv"

WINDOW_SEC = 1.5
PERCLOS_LOOKBACK_SEC = 60.0

EAR_CLOSED_THRESH = 0.2
BLINK_MIN_SEC = 0.07
BLINK_MAX_SEC = 0.40

MAR_YAWN_THRESH = 0.5
YAWN_MIN_SEC = 1.0

MIN_FACE_COVERAGE = 0.5
FRAME_SAMPLE_RATE = 2 # Process every Nth frame. 1 means all frames.

# ==============================================================================
# MEDIAPIPE INDICES
# ==============================================================================
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
INNER_LIP_INDICES = [78, 13, 308, 14] # Left corner, top, right corner, bottom

# 3D Model Points for head pose estimation (in generic units)
# 1 = nose tip, 152 = chin, 263 = left eye left corner, 33 = right eye right corner, 291 = left mouth corner, 61 = right mouth corner
FACE_3D_MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],          # Nose tip (1)
    [0.0, -330.0, -65.0],     # Chin (152)
    [-225.0, 170.0, -135.0],  # Left eye left corner (263)
    [225.0, 170.0, -135.0],   # Right eye right corner (33)
    [-150.0, -150.0, -125.0], # Left mouth corner (291)
    [150.0, -150.0, -125.0]   # Right mouth corner (61)
], dtype=np.float64)

HEAD_POSE_INDICES = [1, 152, 263, 33, 291, 61]

# ==============================================================================
# GEOMETRY FUNCTIONS
# ==============================================================================
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(landmarks, width, height):
    def eye_aspect_ratio(eye_indices):
        p1 = (landmarks[eye_indices[0]].x * width, landmarks[eye_indices[0]].y * height)
        p2 = (landmarks[eye_indices[1]].x * width, landmarks[eye_indices[1]].y * height)
        p3 = (landmarks[eye_indices[2]].x * width, landmarks[eye_indices[2]].y * height)
        p4 = (landmarks[eye_indices[3]].x * width, landmarks[eye_indices[3]].y * height)
        p5 = (landmarks[eye_indices[4]].x * width, landmarks[eye_indices[4]].y * height)
        p6 = (landmarks[eye_indices[5]].x * width, landmarks[eye_indices[5]].y * height)
        
        vertical_1 = distance(p2, p6)
        vertical_2 = distance(p3, p5)
        horizontal = distance(p1, p4)
        
        if horizontal == 0:
            return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    left_ear = eye_aspect_ratio(LEFT_EYE_INDICES)
    right_ear = eye_aspect_ratio(RIGHT_EYE_INDICES)
    return (left_ear + right_ear) / 2.0

def calculate_mar(landmarks, width, height):
    # INNER_LIP_INDICES: 78 (left), 13 (top), 308 (right), 14 (bottom)
    p_left = (landmarks[78].x * width, landmarks[78].y * height)
    p_right = (landmarks[308].x * width, landmarks[308].y * height)
    p_top = (landmarks[13].x * width, landmarks[13].y * height)
    p_bottom = (landmarks[14].x * width, landmarks[14].y * height)
    
    horizontal = distance(p_left, p_right)
    vertical = distance(p_top, p_bottom)
    
    if horizontal == 0:
        return 0.0
    return vertical / horizontal

def calculate_head_pose(landmarks, width, height):
    # Get 2D image points
    image_points = []
    for idx in HEAD_POSE_INDICES:
        image_points.append([landmarks[idx].x * width, landmarks[idx].y * height])
    image_points = np.array(image_points, dtype=np.float64)
    
    # Camera internals
    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        FACE_3D_MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    if not success:
        return 0.0, 0.0, 0.0
        
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Extract Euler angles from rotation matrix
    sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0
        
    pitch = math.degrees(x)
    yaw = math.degrees(y)
    roll = math.degrees(z)
    
    return pitch, yaw, roll

# ==============================================================================
# FEATURE EXTRACTION & AGGREGATION
# ==============================================================================
def process_video(video_path, fold, subject_id, label, existing_df=None):
    video_name = os.path.basename(video_path)
    
    # Check if already processed
    if existing_df is not None:
        mask = (existing_df['fold'] == fold) & (existing_df['subject_id'] == subject_id) & (existing_df['video_name'] == video_name)
        if mask.any():
            return None # Skip

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 30.0 # Fallback

    frames_per_window = max(1, int(WINDOW_SEC * fps))
    frames_per_lookback = int(PERCLOS_LOOKBACK_SEC * fps)

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='face_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1)

    trailing_buffer = deque(maxlen=frames_per_lookback)
    window_frames = []
    
    rows = []
    window_idx = 0
    frame_idx = 0

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            if frame_idx % FRAME_SAMPLE_RATE != 0:
                continue
                
            # The frame_idx in terms of processed frames
            time_sec = frame_idx / fps
            timestamp_ms = int(time_sec * 1000)
            
            height, width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
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
                # Aggregate window
                row = aggregate_window(window_frames, trailing_buffer, fold, subject_id, video_name, window_idx, label)
                rows.append(row)
                window_idx += 1
                window_frames = []
                
        # Process remaining frames if any
        if len(window_frames) > 0:
            row = aggregate_window(window_frames, trailing_buffer, fold, subject_id, video_name, window_idx, label)
            rows.append(row)

    cap.release()
    
    return pd.DataFrame(rows) if rows else None

def aggregate_window(window_frames, trailing_buffer, fold, subject_id, video_name, window_idx, label):
    df_win = pd.DataFrame(window_frames)
    df_trail = pd.DataFrame(list(trailing_buffer))
    
    frames_total = len(df_win)
    frames_no_face = frames_total - df_win['has_face'].sum()
    low_quality = (frames_total - frames_no_face) / frames_total < MIN_FACE_COVERAGE if frames_total > 0 else True
    
    window_start_sec = df_win['time_sec'].iloc[0] if frames_total > 0 else 0.0
    
    # Filter for frames with faces
    df_win_face = df_win[df_win['has_face']]
    
    if len(df_win_face) > 0:
        ear_mean = df_win_face['ear'].mean()
        ear_min = df_win_face['ear'].min()
        ear_std = df_win_face['ear'].std() if len(df_win_face) > 1 else 0.0
        
        mar_mean = df_win_face['mar'].mean()
        mar_max = df_win_face['mar'].max()
        
        pitch_mean = df_win_face['pitch'].mean()
        yaw_mean = df_win_face['yaw'].mean()
        roll_mean = df_win_face['roll'].mean()
        pitch_std = df_win_face['pitch'].std() if len(df_win_face) > 1 else 0.0
    else:
        ear_mean = ear_min = ear_std = np.nan
        mar_mean = mar_max = np.nan
        pitch_mean = yaw_mean = roll_mean = pitch_std = np.nan
        
    # PERCLOS and blink rate over trailing buffer
    df_trail_face = df_trail[df_trail['has_face']]
    if len(df_trail_face) > 0:
        perclos = (df_trail_face['ear'] < EAR_CLOSED_THRESH).mean()
        
        # Blink rate calculation
        is_closed = df_trail_face['ear'] < EAR_CLOSED_THRESH
        closed_groups = (is_closed != is_closed.shift()).cumsum()
        closed_segments = df_trail_face[is_closed].groupby(closed_groups)
        
        blink_count = 0
        for _, segment in closed_segments:
            duration = segment['time_sec'].iloc[-1] - segment['time_sec'].iloc[0]
            if BLINK_MIN_SEC <= duration <= BLINK_MAX_SEC:
                blink_count += 1
                
        # rate per minute
        trail_duration = df_trail['time_sec'].iloc[-1] - df_trail['time_sec'].iloc[0]
        if trail_duration > 0:
            blink_rate = blink_count / (trail_duration / 60.0)
        else:
            blink_rate = 0.0
            
        # Is yawn over trailing buffer
        is_yawning = df_trail_face['mar'] > MAR_YAWN_THRESH
        yawn_groups = (is_yawning != is_yawning.shift()).cumsum()
        yawn_segments = df_trail_face[is_yawning].groupby(yawn_groups)
        
        is_yawn = 0
        for _, segment in yawn_segments:
            duration = segment['time_sec'].iloc[-1] - segment['time_sec'].iloc[0]
            if duration >= YAWN_MIN_SEC:
                is_yawn = 1
                break
    else:
        perclos = np.nan
        blink_rate = np.nan
        is_yawn = 0
        
    # Fill standard output row
    row = {
        'fold': fold,
        'subject_id': subject_id,
        'video_name': video_name,
        'window_idx': window_idx,
        'window_start_sec': window_start_sec,
        'ear_mean': ear_mean,
        'ear_min': ear_min,
        'ear_std': ear_std,
        'perclos': perclos,
        'blink_rate': blink_rate,
        'mar_mean': mar_mean,
        'mar_max': mar_max,
        'is_yawn': is_yawn,
        'pitch_mean': pitch_mean,
        'yaw_mean': yaw_mean,
        'roll_mean': roll_mean,
        'pitch_std': pitch_std,
        'frames_total': frames_total,
        'frames_no_face': frames_no_face,
        'low_quality': int(low_quality),
        'drowsiness_level': label
    }
    return row

# ==============================================================================
# MAIN & DATASET WALKER
# ==============================================================================
def find_videos():
    video_files = []
    
    for root, dirs, files in os.walk(DATASET_ROOT):
        # find fold name in root path
        path_parts = os.path.normpath(root).split(os.sep)
        fold_name = None
        for part in path_parts:
            if part.lower().startswith("fold"):
                fold_name = part.split('_')[0]
                break
                
        if not fold_name:
            continue
            
        # check if this is a subject folder
        valid_labels = {'0': 0, '5': 1, '10': 2}
        found_videos = []
        for f in files:
            name, ext = os.path.splitext(f)
            if ext.lower() in ['.mov', '.mp4', '.avi'] and name in valid_labels:
                found_videos.append((os.path.join(root, f), fold_name, os.path.basename(root), valid_labels[name]))
                
        video_files.extend(found_videos)
        
    return video_files

def main(test_single_subject=False):
    print("Finding videos...")
    videos = find_videos()
    print(f"Found {len(videos)} videos total.")
    
    if len(videos) == 0:
        print("No videos found. Check DATASET_ROOT.")
        return

    if test_single_subject:
        print("Testing on 3 subject folders as requested.")
        # Get unique subject folders
        unique_subjects = list(dict.fromkeys(v[2] for v in videos))
        selected_subjects = unique_subjects[:3]
        videos = [v for v in videos if v[2] in selected_subjects]
        print(f"Running on {len(videos)} videos for subjects {selected_subjects}")

    existing_df = None
    if os.path.exists(OUTPUT_CSV):
        print(f"Loading existing features from {OUTPUT_CSV} for resumability...")
        existing_df = pd.read_csv(OUTPUT_CSV)

    # Make sure output CSV has header if not exists
    schema = [
        'fold', 'subject_id', 'video_name', 'window_idx', 'window_start_sec', 
        'ear_mean', 'ear_min', 'ear_std', 'perclos', 'blink_rate', 
        'mar_mean', 'mar_max', 'is_yawn', 'pitch_mean', 'yaw_mean', 
        'roll_mean', 'pitch_std', 'frames_total', 'frames_no_face', 
        'low_quality', 'drowsiness_level'
    ]

    for video_path, fold, subj_folder, label in tqdm(videos, desc="Processing videos"):
        subject_id = f"{fold}_{subj_folder}"
        video_name = os.path.basename(video_path)
        
        df_res = process_video(video_path, fold, subject_id, label, existing_df)
        
        if df_res is not None and len(df_res) > 0:
            df_res = df_res[schema] # Ensure column order
            df_res.to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
            print(f"\\nProcessed {video_name} for {subject_id}: {len(df_res)} rows.")

if __name__ == "__main__":
    import sys
    test_mode = len(sys.argv) > 1 and sys.argv[1] == '--test'
    main(test_single_subject=test_mode)
