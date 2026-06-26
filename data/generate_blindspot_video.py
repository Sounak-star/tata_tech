import cv2
import numpy as np
import os

def generate_video():
    os.makedirs('data', exist_ok=True)
    width, height = 640, 480
    fps = 15
    duration_sec = 10
    total_frames = fps * duration_sec
    
    # Define codec and create VideoWriter
    # Use MP4V codec which is standard and supported on Windows/FastAPI
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('data/blindspot_clip.mp4', fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: Could not open VideoWriter. Trying alternative codec...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('data/blindspot_clip.mp4', fourcc, fps, (width, height))
        
    if not out.isOpened():
        print("Fatal Error: VideoWriter failed to initialize.")
        return
        
    for frame_idx in range(total_frames):
        # Create background (brown/grey dirt ground)
        frame = np.full((height, width, 3), 45, dtype=np.uint8)
        # Add texture lines to represent ground grid
        for y in range(0, height, 40):
            cv2.line(frame, (0, y), (width, y), (55, 55, 55), 1)
        
        # Draw excavator rear bumper at the bottom (yellow and black hazard stripes)
        cv2.rectangle(frame, (0, 420), (width, 480), (30, 30, 30), -1)
        stripe_width = 30
        for x in range(-50, width + 50, stripe_width * 2):
            pts = np.array([
                [x, 420],
                [x + stripe_width, 420],
                [x + stripe_width + 20, 480],
                [x + 20, 480]
            ], np.int32)
            cv2.fillPoly(frame, [pts], (0, 210, 255)) # Yellow stripes
            
        # Draw bumper text
        cv2.putText(frame, "REAR CAMERA - KEEP CLEAR", (190, 455),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Draw two safety cones on the left and right sides
        # Left cone
        cv2.fillConvexPoly(frame, np.array([[80, 380], [100, 320], [120, 380]], np.int32), (0, 102, 255))
        cv2.fillConvexPoly(frame, np.array([[90, 350], [100, 320], [110, 350]], np.int32), (255, 255, 255))
        cv2.rectangle(frame, (70, 380), (130, 385), (0, 102, 255), -1)
        
        # Right cone
        cv2.fillConvexPoly(frame, np.array([[520, 380], [540, 320], [560, 380]], np.int32), (0, 102, 255))
        cv2.fillConvexPoly(frame, np.array([[530, 350], [540, 320], [550, 350]], np.int32), (255, 255, 255))
        cv2.rectangle(frame, (510, 380), (570, 385), (0, 102, 255), -1)
        
        # Time interpolation parameter t (0.0 to 1.0)
        t = frame_idx / total_frames
        
        # Worker position: walking from far away top (y=120) to close bottom (y=390)
        px = int(320 + 80 * np.sin(t * np.pi * 2))
        py = int(120 + 270 * t)
        
        # Size scale
        scale = 0.3 + 1.7 * t
        
        # Dimensions
        head_r = int(10 * scale)
        torso_w = int(24 * scale)
        torso_h = int(35 * scale)
        leg_w = int(8 * scale)
        leg_h = int(30 * scale)
        
        # Draw legs (Blue denim trousers)
        cv2.rectangle(frame, (px - torso_w//3 - leg_w//2, py + torso_h), (px - torso_w//3 + leg_w//2, py + torso_h + leg_h), (120, 50, 20), -1)
        cv2.rectangle(frame, (px + torso_w//3 - leg_w//2, py + torso_h), (px + torso_w//3 + leg_w//2, py + torso_h + leg_h), (120, 50, 20), -1)
        
        # Draw torso (Orange safety vest)
        cv2.rectangle(frame, (px - torso_w//2, py), (px + torso_w//2, py + torso_h), (0, 102, 255), -1)
        # Silver reflective stripes
        stripe_w = max(1, int(3 * scale))
        cv2.rectangle(frame, (px - torso_w//3, py), (px - torso_w//3 + stripe_w, py + torso_h), (240, 240, 240), -1)
        cv2.rectangle(frame, (px + torso_w//3 - stripe_w, py), (px + torso_w//3, py + torso_h), (240, 240, 240), -1)
        
        # Draw arms
        cv2.rectangle(frame, (px - torso_w//2 - int(5*scale), py), (px - torso_w//2, py + int(25*scale)), (200, 200, 200), -1)
        cv2.rectangle(frame, (px + torso_w//2, py), (px + torso_w//2 + int(5*scale), py + int(25*scale)), (200, 200, 200), -1)
        
        # Draw face (Skin tone)
        cv2.circle(frame, (px, py - head_r), head_r, (180, 220, 250), -1)
        
        # Draw Safety Helmet (Yellow)
        helmet_r = int(11 * scale)
        cv2.ellipse(frame, (px, py - head_r - int(2*scale)), (helmet_r, int(7*scale)), 0, 180, 360, (0, 210, 255), -1)
        cv2.rectangle(frame, (px - helmet_r - int(2*scale), py - head_r - int(2*scale)), (px + helmet_r + int(2*scale), py - head_r), (0, 210, 255), -1)
        
        out.write(frame)
        
    out.release()
    print("Successfully generated data/blindspot_clip.mp4")

if __name__ == '__main__':
    generate_video()
