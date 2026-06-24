"""
Step 3 — Standalone YOLO test on blindspot clip.
Run this BEFORE any dashboard integration to confirm:
  1. The environment works (ultralytics installed, weights auto-download).
  2. The clip loads and people are detected.
  3. You can see actual box_h values to pick GREEN/AMBER/RED thresholds.

Usage:
    python test_yolo_clip.py

Press 'q' to quit the preview window early.
"""

from ultralytics import YOLO
import cv2

# ── config ──────────────────────────────────────────────────────────────
CLIP_PATH = "data/blindspot_clip.mp4"
MODEL_NAME = "yolo11n.pt"  # YOLO11-nano, auto-downloads on first use

# Suggested zone thresholds (tune after watching the output):
#   box_h < 0.25  → GREEN  (far away)
#   0.25 ≤ box_h ≤ 0.50 → AMBER  (approaching)
#   box_h > 0.50  → RED    (close / danger)
GREEN_MAX = 0.25
AMBER_MAX = 0.50

# ── colours ─────────────────────────────────────────────────────────────
COLOR_GREEN = (0, 200, 0)
COLOR_AMBER = (0, 180, 255)
COLOR_RED   = (0, 0, 255)

# ── run ─────────────────────────────────────────────────────────────────
model = YOLO(MODEL_NAME)  # auto-downloads weights first time
cap = cv2.VideoCapture(CLIP_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open clip: {CLIP_PATH}")

print(f"[INFO] Opened clip: {CLIP_PATH}")
print(f"[INFO] Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"[INFO] FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}")
print(f"[INFO] Thresholds → GREEN < {GREEN_MAX:.2f} | AMBER {GREEN_MAX:.2f}–{AMBER_MAX:.2f} | RED > {AMBER_MAX:.2f}")
print("[INFO] Press 'q' to quit.\n")

frame_num = 0
while True:
    ok, frame = cap.read()
    if not ok:
        print("[INFO] End of clip.")
        break

    frame_num += 1
    results = model(frame, verbose=False)[0]
    h = frame.shape[0]

    for box in results.boxes:
        if int(box.cls) == 0:  # class 0 = person
            x1, y1, x2, y2 = box.xyxy[0]
            box_h = float((y2 - y1) / h)  # box height as fraction of frame

            # pick zone colour
            if box_h > AMBER_MAX:
                zone, color = "RED", COLOR_RED
            elif box_h > GREEN_MAX:
                zone, color = "AMBER", COLOR_AMBER
            else:
                zone, color = "GREEN", COLOR_GREEN

            # draw
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"person {box_h:.2f} [{zone}]"
            cv2.putText(frame, label, (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            print(f"  frame {frame_num:4d} | box_h={box_h:.3f} | zone={zone}")

    cv2.imshow("YOLO Blindspot Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("[INFO] Quit by user.")
        break

cap.release()
cv2.destroyAllWindows()
print("\n[DONE] Review the box_h values above to tune your GREEN/AMBER/RED thresholds.")
