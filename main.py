"""
Age & Gender Detection — Accurate Version
Uses InsightFace (dedicated age/gender model, much more accurate than DeepFace)

Install:  pip install insightface onnxruntime opencv-python numpy
Run:      python age_gender_live.py
Press Q to quit.
"""

import cv2
import numpy as np

# InsightFace — high accuracy face analysis using ONNX models
# Uses a dedicated genderage model trained specifically for this task
from insightface.app import FaceAnalysis

# ── Load Model ────────────────────────────────────────────────────
print("⏳ Loading model (downloads on first run ~200MB)...")

app = FaceAnalysis(
    allowed_modules=['detection', 'genderage'],  # Only load what we need
    providers=['CPUExecutionProvider']            # Use CPU (no GPU needed)
)

# det_size: higher = more accurate but slower. 640x640 is a good balance.
app.prepare(ctx_id=-1, det_size=(640, 640))

print("✅ Model ready! Opening webcam...")

# ── Webcam Setup ──────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not found. Try changing 0 to 1 in VideoCapture(0)")
    exit()

# ── Main Loop ─────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame.")
        break

    # ── Run Face Analysis ─────────────────────────────────────────
    # InsightFace returns a list of detected faces
    # Each face has: .bbox, .age, .gender (1=Male, 0=Female)
    faces = app.get(frame)

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2, cv2.LINE_AA)

    for face in faces:
        # Bounding box coordinates
        x1, y1, x2, y2 = face.bbox.astype(int)

        # Age (integer, e.g. 24)
        age = int(face.age)

        # Gender: InsightFace gives 1 for Male, 0 for Female
        gender = "Male" if face.gender == 1 else "Female"
        gender_color = (255, 180, 50) if face.gender == 1 else (220, 80, 200)

        label = f"{gender}  |  Age: {age}"

        # Draw face bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), gender_color, 2)

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 10, y1), gender_color, -1)

        # Draw label text
        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

    # ── Show Frame ────────────────────────────────────────────────
    cv2.imshow("Age & Gender Detection  —  Press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Done.")