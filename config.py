"""Shared configuration for the pose-landmark extraction pipeline."""

# ---- Input / output ----
input_video_path = "materials/videos/squat.mp4"
output_folder = "output"

# ---- MediaPipe Pose Landmarker model ----
# "heavy" is the most accurate of the three BlazePose variants
# (lite/full/heavy) at the cost of slower inference -- fine here since
# this is offline export, not realtime. Swap the filename/url pair
# below to switch variants; the other two are:
#   lite:  pose_landmarker_lite.task
#   full:  pose_landmarker_full.task
model_dir = "models"
model_filename = "pose_landmarker_heavy.task"
model_url = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)

min_detection_confidence = 0.5
min_tracking_confidence = 0.5
min_presence_confidence = 0.5
min_landmark_visibility = 0.5

# ---- Smoothing across frames ----
# 0 = off, higher = smoother but laggier. Only used by
# export_landmarks.py -- export_landmarks_rigged.py intentionally skips
# smoothing since Kalidokit does its own internally.
smoothing_alpha = 0.4