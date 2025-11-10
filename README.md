Webcam Leg Angle Calculation

Introduction:
This project uses a webcam to calcualtion leg angles in real-time. It leverages MediaPipe Pose for human pose estimation, OpenCV for video capture and display, and NumPy for numerical calculations. The system can optionally count repetitions of leg movements, smooth noisy measurements, log data to CSV, and save processed video.

Workflow Steps

Open webcam & configure frame size/FPS.

Initialize MediaPipe Pose.

Capture frames in a loop.

Detect pose landmarks.

Calculate knee angles using hip-knee-ankle coordinates.

Apply smoothing (EMA + median buffer, optional).

Count reps using state machine (optional).

Draw skeleton, angles, FPS, and rep counts.

Log to CSV & save video (optional).

Display video → quit on q.

Cleanup resources.

It is designed for applications such as:

Exercise tracking (e.g., squats, lunges)

Physical therapy monitoring

Real-time feedback for sports or fitness

What the Code Does

Webcam Capture

Opens your webcam and continuously captures frames.

Supports adjustable resolution and frame rate.

Pose Detection

Detects human body landmarks (hips, knees, ankles, etc.) using MediaPipe Pose.

Provides normalized 2D coordinates for each landmark.

Leg Angle Calculation

Calculates the angle at each knee using the hip, knee, and ankle coordinates.

Converts vectors into angles in degrees for easy interpretation.

Visualization

Displays the live webcam feed.

Overlays pose landmarks, Leg angles, and optional repetition counts on the video.

Shows frames per second (FPS) for performance monitoring.

Smoothing (Optional in Long Version)

Applies Exponential Moving Average (EMA) to reduce jitter in knee angle measurements.

Uses a short-term median buffer for smooth visual display.

Repetition Counting (Long Version)

Counts leg repetitions based on knee flexion and extension thresholds.

Implements a state machine to track movement and avoid false counts.

Logging & Output (Long Version)

Saves leg angle data and rep counts to a CSV file.

Optionally saves a video with landmarks and angles overlaid.

Live Plotting (Long Version)

Uses Matplotlib to visualize knee angles over time in real-time.

Summary

Short version: Simple real-time knee angle visualizer.

Long version: Full exercise tracker with smoothing, repetition counting, logging, and plotting.

The main concept: capture webcam frames → detect pose → calculate knee angles → display and optionally log or track repetitions.
