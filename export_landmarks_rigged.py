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

Reads:  config.input_video_path
Writes: output/<video-name>-[rigged-3d-viewer].html
"""

import logging
import os
from pathlib import Path

import cv2 as cv
from tqdm import tqdm

import config
from pose_detector import PoseDetector
from utils import make_sure_file_exists
from viewer_writer import write_viewer_html

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

    detector = PoseDetector(fps)
    frames_out = []

    progress = tqdm(total=total_frames, desc="Extracting raw landmarks...")
    while True:
        success, frame = video.read()
        if not success:
            break

        result = detector.detect(frame)

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
    detector.close()
    video.release()
    return frames_out, fps


if __name__ == "__main__":
    os.makedirs(config.output_folder, exist_ok=True)
    frames, fps = extract_frames(config.input_video_path)
    filename_noext = Path(config.input_video_path).stem
    out_path = os.path.join(config.output_folder, f"{filename_noext}-[rigged-3d-viewer].html")
    template_path = os.path.join(os.path.dirname(__file__), "viewer_template_rigged.html")
    write_viewer_html({"frames": frames, "fps": fps}, template_path, out_path)
