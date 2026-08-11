"""
Exports raw MediaPipe pose landmarks (both normalized-image-space and
metric world-space) for every frame, formatted for Kalidokit's Pose
solver. Pairs with viewer_template_rigged.html, which loads a rigged
GLB/GLTF model, runs Kalidokit in-browser each frame, and drives the
model's skeleton from the resulting bone rotations.

Unlike export_landmarks.py, this does NOT smooth/filter/strip landmarks
or restrict to a bone subset -- Kalidokit wants the full raw 33-point
MediaPipe format (it does its own internal smoothing), and it needs
BOTH the normalized landmarks and the world landmarks to solve rotations
correctly.

Usage:
    python export_landmarks_rigged.py

Reads:  input_video_path (from pose_skeleton.py config)
Writes: output/<video-name>-[rigged-3d-viewer].html
"""

import json
import logging
import os
from pathlib import Path

import cv2 as cv
from tqdm import tqdm

from pose_skeleton import (
    SkeletonRenderer3D,
    input_video_path,
    make_sure_file_exists,
    output_folder,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def landmark_to_list(lm) -> list:
    """Kalidokit expects [x, y, z, visibility] per landmark."""
    return [lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)]


def extract_frames(input_path: str) -> tuple[list[dict], float]:
    make_sure_file_exists(input_path)

    video = cv.VideoCapture(input_path)
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = float(video.get(cv.CAP_PROP_FPS))
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    logger.info(f"fps: {fps}; total_frames: {total_frames}")

    # Reuse SkeletonRenderer3D purely to get an already-configured
    # landmarker + correct timestamp cadence; we bypass its smoothing.
    renderer = SkeletonRenderer3D(fps)
    frame_duration_ms = renderer._frame_duration_ms
    landmarker = renderer.landmarker

    from mediapipe import Image, ImageFormat

    frames_out = []
    frame_index = 0

    progress = tqdm(total=total_frames, desc="Extracting raw landmarks...")
    while True:
        success, frame = video.read()
        if not success:
            break

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(frame_index * frame_duration_ms)
        frame_index += 1

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        entry = {"landmarks": None, "worldLandmarks": None}
        if result.pose_landmarks:
            entry["landmarks"] = [landmark_to_list(lm) for lm in result.pose_landmarks[0]]
        if result.pose_world_landmarks:
            entry["worldLandmarks"] = [
                landmark_to_list(lm) for lm in result.pose_world_landmarks[0]
            ]

        frames_out.append(entry)
        progress.update(1)

    progress.close()
    renderer.close()
    video.release()
    return frames_out, fps


def build_html(frames: list[dict], fps: float, out_path: str) -> None:
    payload = {"frames": frames, "fps": fps}

    template_path = os.path.join(os.path.dirname(__file__), "viewer_template_rigged.html")
    with open(template_path, "r") as f:
        template = f.read()

    html = template.replace("__SKELETON_DATA__", json.dumps(payload))

    with open(out_path, "w") as f:
        f.write(html)

    logger.info(f"Viewer written to: {out_path}")


if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)
    frames, fps = extract_frames(input_video_path)
    filename_noext = Path(input_video_path).stem
    out_path = os.path.join(output_folder, f"{filename_noext}-[rigged-3d-viewer].html")
    build_html(frames, fps, out_path)
