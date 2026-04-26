import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------
# AUTO DOWNLOAD MODEL (if missing)
# -------------------------------
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded!")

# -------------------------------
# EYE LANDMARK INDICES
# -------------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# -------------------------------
# FUNCTION: EYE RATIO
# -------------------------------
def eye_ratio(eye_points):
    v1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    v2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    h = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    return (v1 + v2) / (2.0 * h)

# -------------------------------
# MEDIAPIPE SETUP
# -------------------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

# -------------------------------
# CAMERA START
# -------------------------------
cap = cv2.VideoCapture(0)

print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    state = "No Face"
    color = (255, 255, 255)

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        h, w, _ = frame.shape
        points = [(int(l.x * w), int(l.y * h)) for l in landmarks]

        # Extract eye points
        left_eye = [points[i] for i in LEFT_EYE]
        right_eye = [points[i] for i in RIGHT_EYE]

        # Compute ratios
        left_ratio = eye_ratio(left_eye)
        right_ratio = eye_ratio(right_eye)
        avg_ratio = (left_ratio + right_ratio) / 2

        # -------------------------------
        # ATTENTION LOGIC
        # -------------------------------
        if avg_ratio < 0.20:
            state = "SLEEPING 😴"
            color = (0, 0, 255)
        elif avg_ratio < 0.25:
            state = "DROWSY 😪"
            color = (0, 165, 255)
        else:
            state = "ATTENTIVE 😐"
            color = (0, 255, 0)

        # Draw eye landmarks
        for p in left_eye + right_eye:
            cv2.circle(frame, p, 2, (255, 0, 0), -1)

    cv2.putText(frame, state, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                color, 2)

    cv2.imshow("Attention Monitor", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# -------------------------------
# CLEANUP
# -------------------------------
cap.release()
cv2.destroyAllWindows()