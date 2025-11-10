import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    dot_product = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)
    if mag_ba == 0 or mag_bc == 0:
        return 0.0
    cosine_angle = dot_product / (mag_ba * mag_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# --- Configuration ---
camera_index = 0                   # change if your camera device has different index
save_output = False                # set True to save processed video to disk
output_video_path = 'webcam_output.mp4'
frame_width = 640                  # desired frame size (will try to set on the camera)
frame_height = 480
desired_fps = 30

# Open webcam
cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, desired_fps)

if save_output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, desired_fps, (frame_width, frame_height))
else:
    out = None

if not cap.isOpened():
    raise IOError(f"Cannot open camera index {camera_index}")

# Use Pose in a context manager for proper cleanup
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    prev_time = 0
    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame from camera.")
            break

        # Optional: flip horizontally for a mirror view
        frame = cv2.flip(frame, 1)

        # Prepare image for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        h, w = image_bgr.shape[:2]

        # Draw landmarks and compute angles if available
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Left leg points
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            left_angle = calculate_angle(l_hip, l_knee, l_ankle)

            # Right leg points
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            right_angle = calculate_angle(r_hip, r_knee, r_ankle)

            # Pixel coords for display
            l_knee_px = (int(l_knee[0] * w), int(l_knee[1] * h))
            r_knee_px = (int(r_knee[0] * w), int(r_knee[1] * h))

            # Draw angles as text
            cv2.putText(image_bgr, f"L: {left_angle:.1f} deg", (l_knee_px[0] - 30, l_knee_px[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image_bgr, f"R: {right_angle:.1f} deg", (r_knee_px[0] - 30, r_knee_px[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0.0
        prev_time = curr_time
        cv2.putText(image_bgr, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Webcam - Pose & Knee Angles (press q to quit)', image_bgr)

        # Optionally write to file
        if out is not None:
            out.write(image_bgr)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()