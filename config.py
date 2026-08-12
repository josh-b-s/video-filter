"""Shared configuration for the pose-landmark extraction pipeline."""

# ---- Input / output ----
input_video_path = "materials/videos/pullup.mp4"
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
# MediaPipe still estimates a 3D position for a landmark even when it's
# occluded (e.g. an arm hidden behind the torso), just with lower
# confidence -- it isn't just guessing blindly, it uses a learned body
# pose prior. Filtering those out at 0.5 was causing occluded limbs to
# disappear from the figure entirely instead of just being a bit less
# certain. Lowered so occluded-but-estimated points still render;
# raise this back up if you'd rather have limbs vanish than show a
# possibly-off position during heavy occlusion.
min_landmark_visibility = 0.15

# ---- Smoothing across frames ----
# 0 = off, higher = smoother but laggier. Only used by
# export_landmarks.py -- export_landmarks_rigged.py intentionally skips
# smoothing since Kalidokit does its own internally.
smoothing_alpha = 0.4
