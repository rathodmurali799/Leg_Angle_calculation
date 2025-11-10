
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ----------------- Configuration -----------------
camera_index = 0            # change if your webcam index differs
frame_width = 640
frame_height = 480
desired_fps = 30
save_output_video = True    # set False to disable saving the processed video
output_video_path = "webcam_output_with_angles.mp4"
output_csv_path = "angle_log.csv"

# Smoothing (EMA) alpha between 0 (very smooth) and 1 (no smoothing)
ema_alpha = 0.3

# Optional short-term smoothing for display (median-like)
display_buffer_len = 5

# Rep counting thresholds (knee angle in degrees)
# Typical knee: straight ~ 160-180, bent ~ 60-120 depending on exercise.
flex_threshold = 100       # below this -> considered "flexed"
extend_threshold = 160     # above this -> considered "extended"
min_seconds_between_reps = 0.5  # debounce

# -------------------------------------------------

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

# Open webcam
cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, desired_fps)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera index {camera_index}")

# Setup video writer if saving
if save_output_video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Use values from camera if available
    fps = cap.get(cv2.CAP_PROP_FPS) or desired_fps
    out = cv2.VideoWriter(output_video_path, fourcc, float(fps), (frame_width, frame_height))
else:
    out = None

# Prepare CSV logging
csv_file = open(output_csv_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "timestamp", "frame_idx",
    "left_knee_raw", "left_knee_ema", "left_knee_display",
    "right_knee_raw", "right_knee_ema", "right_knee_display",
    "left_reps", "right_reps"
])

# For EMA and display smoothing
left_ema = None
right_ema = None
left_buffer = deque(maxlen=display_buffer_len)
right_buffer = deque(maxlen=display_buffer_len)

# Rep counting state machines
class RepCounter:
    def __init__(self, flex_th, extend_th, min_dt):
        self.flex_th = flex_th
        self.extend_th = extend_th
        self.state = "extended"  # start assuming extended (to require flex first)
        self.count = 0
        self.last_rep_time = 0.0
        self.min_dt = min_dt

    def update(self, angle, timestamp):
        # state machine: extended -> flexed -> extended counts 1 rep
        if self.state == "extended":
            if angle < self.flex_th:
                self.state = "flexed"
        elif self.state == "flexed":
            if angle > self.extend_th:
                # check debounce
                if timestamp - self.last_rep_time >= self.min_dt:
                    self.count += 1
                    self.last_rep_time = timestamp
                self.state = "extended"
        return self.count

left_counter = RepCounter(flex_threshold, extend_threshold, min_seconds_between_reps)
right_counter = RepCounter(flex_threshold, extend_threshold, min_seconds_between_reps)

# Matplotlib live plot setup
plt.ion()
fig, ax = plt.subplots(figsize=(6, 3))
ax.set_title("Knee Angles (smoothed)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Angle (deg)")
ax.set_ylim(0, 200)
line_left, = ax.plot([], [], label="Left (EMA)")
line_right, = ax.plot([], [], label="Right (EMA)")
ax.legend()
times = deque(maxlen=300)      # keep last N points
left_vals = deque(maxlen=300)
right_vals = deque(maxlen=300)
start_time = time.time()

# Main loop using MediaPipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    frame_idx = 0
    prev_time = time.time()
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Ignoring empty frame.")
                break

            frame_idx += 1
            # mirror view
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Prepare image for pose
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            timestamp = time.time() - start_time

            left_raw = 0.0
            right_raw = 0.0

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # LEFT leg
                l_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                l_knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                          lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                l_ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                           lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                left_raw = calculate_angle(l_hip, l_knee, l_ankle)

                # RIGHT leg
                r_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                r_knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                r_ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                right_raw = calculate_angle(r_hip, r_knee, r_ankle)

                # Update EMA
                if left_ema is None:
                    left_ema = left_raw
                else:
                    left_ema = (ema_alpha * left_raw) + (1 - ema_alpha) * left_ema

                if right_ema is None:
                    right_ema = right_raw
                else:
                    right_ema = (ema_alpha * right_raw) + (1 - ema_alpha) * right_ema

                # Display smoothing buffer (median-like)
                left_buffer.append(left_ema)
                right_buffer.append(right_ema)
                left_display = float(np.median(left_buffer))
                right_display = float(np.median(right_buffer))

                # Update rep counters
                left_reps = left_counter.update(left_display, timestamp)
                right_reps = right_counter.update(right_display, timestamp)

                # Draw text near knees
                # Convert normalized coords to pixels for placing text
                try:
                    l_knee_px = (int(l_knee[0] * w), int(l_knee[1] * h))
                    r_knee_px = (int(r_knee[0] * w), int(r_knee[1] * h))
                except Exception:
                    l_knee_px = (50, 100)
                    r_knee_px = (w-150, 100)

                cv2.putText(image_bgr, f"L:{left_display:.1f}° R:{right_display:.1f}°",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image_bgr, f"Left reps: {left_reps}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2, cv2.LINE_AA)
                cv2.putText(image_bgr, f"Right reps: {right_reps}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2, cv2.LINE_AA)

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

            else:
                # No landmarks detected, keep previous EMA values for display
                left_display = float(np.median(left_buffer)) if left_buffer else 0.0
                right_display = float(np.median(right_buffer)) if right_buffer else 0.0
                left_reps = left_counter.count
                right_reps = right_counter.count

                cv2.putText(image_bgr, "No pose detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # Log to CSV
            csv_writer.writerow([
                time.time(), frame_idx,
                f"{left_raw:.3f}", f"{left_ema:.3f}" if left_ema is not None else "",
                f"{left_display:.3f}",
                f"{right_raw:.3f}", f"{right_ema:.3f}" if right_ema is not None else "",
                f"{right_display:.3f}",
                left_reps, right_reps
            ])

            # Write processed frame to output video if enabled
            if out is not None:
                # ensure writer frame size same as writer expectations
                frame_to_write = cv2.resize(image_bgr, (frame_width, frame_height))
                out.write(frame_to_write)

            # Show frame
            cv2.imshow("Webcam - Pose & Knee Angles (press 'q' to quit)", image_bgr)

            # Update matplotlib data and redraw (non-blocking)
            times.append(timestamp)
            left_vals.append(left_display)
            right_vals.append(right_display)
            # update plot data
            line_left.set_data(times, left_vals)
            line_right.set_data(times, right_vals)
            ax.relim()
            ax.autoscale_view(scalex=True, scaley=False)
            plt.pause(0.001)  # tiny pause to allow GUI update

            # Handle quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
        csv_file.close()
        cv2.destroyAllWindows()
        plt.close(fig)

        print("Finished.")
        if save_output_video:
            print(f"Saved processed video to: {output_video_path}")
        print(f"Saved CSV log to: {output_csv_path}")
        print(f"Left reps: {left_counter.count}, Right reps: {right_counter.count}")
