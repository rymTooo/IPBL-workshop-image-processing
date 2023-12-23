import cv2
import mediapipe as mp
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import math
from MediapipeHandLandmark import MediapipeHandLandmark as HandLmk


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize pose estimation.
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)  # 0 typically refers to the default camera.

def print_z_coordinate(landmarks):
    print("--------------------------")
    print("nose = 0")
    print(landmarks.landmark[mp_pose.PoseLandmark.NOSE.value])
def analyze_posture(landmarks):
    """Analyze the landmarks and determine if the posture is 'good' or 'bad'."""
    try:
        # Get landmarks
        nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].y
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y

        # Calculate the midpoint of the shoulders' y-values
        shoulder_mid = (left_shoulder + right_shoulder) / 2

        # If the nose is significantly lower than the shoulder midpoint, classify as 'bad' posture
        if nose > shoulder_mid + 0.05:  # 0.05 is a threshold, adjust as needed.
            return "bad"
        else:
            return "good"
    except:
        return "error"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the BGR image to RGB.
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # Draw the pose landmarks on the image and analyze posture
    if results.pose_landmarks:
        posture_status = analyze_posture(results.pose_landmarks)
        annotated_image = frame.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            print_z_coordinate(results.pose_landmarks)
        cv2.putText(annotated_image, f"Posture: {posture_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Annotated Image', annotated_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()