"""
Exports 3D world-landmark points into a single self-contained HTML file
(viewer_template.html) you can open in any browser and freely rotate /
zoom / scrub through frames, rendered as a volumetric 3D stick figure.

Usage:
    python export_landmarks.py

Reads:  config.input_video_path
Writes: output/<video-name>-[skeleton-3d-viewer].html
"""

import logging
import os
from pathlib import Path

import cv2 as cv
from tqdm import tqdm

import config
import landmarks
from landmark_smoothing import LandmarkSmoother
from pose_detector import PoseDetector
from utils import make_sure_file_exists
from viewer_writer import write_viewer_html

# Set to True to include ALL 33 landmarks (including the face: nose,
# eyes, ears, mouth) instead of the face-stripped set. Only meant for
# your own local viewing -- the volumetric viewer ignores individual
# face points either way and just uses the nose as a head anchor.
SHOW_ALL_LANDMARKS = True

ACTIVE_LANDMARKS = landmarks.ALL_LANDMARKS if SHOW_ALL_LANDMARKS else landmarks.BODY_LANDMARKS
ACTIVE_CONNECTIONS = landmarks.ALL_CONNECTIONS if SHOW_ALL_LANDMARKS else landmarks.BODY_CONNECTIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_frames(input_path: str) -> tuple[list[dict], list[dict]]:
    make_sure_file_exists(input_path)

    video = cv.VideoCapture(input_path)
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = float(video.get(cv.CAP_PROP_FPS))
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    logger.info(f"fps: {fps}; total_frames: {total_frames}")

    detector = PoseDetector(fps)
    smoother = LandmarkSmoother()
    frames_out = []
    raw_frames_out = []

    progress = tqdm(total=total_frames, desc="Extracting 3D landmarks...")
    while True:
        success, frame = video.read()
        if not success:
            break

        result = detector.detect(frame)

        points = landmarks.extract_points(result, ACTIVE_LANDMARKS, smoother)
        if not SHOW_ALL_LANDMARKS:
            landmarks.add_synthetic_head(points)
        frames_out.append({str(k): list(v) for k, v in points.items()})

        raw_points = landmarks.extract_raw_points(result)
        raw_frames_out.append({str(k): list(v) for k, v in raw_points.items()})

        progress.update(1)

    progress.close()
    detector.close()
    video.release()
    return frames_out, raw_frames_out


def build_payload(frames: list[dict], raw_frames: list[dict]) -> dict:
    return {
        "frames": frames,
        "bones": ACTIVE_CONNECTIONS,
        "hasHead": not SHOW_ALL_LANDMARKS,
        "rawFrames": raw_frames,
        "rawBones": landmarks.ALL_CONNECTIONS,
    }


if __name__ == "__main__":
    os.makedirs(config.output_folder, exist_ok=True)
    frames, raw_frames = extract_frames(config.input_video_path)
    filename_noext = Path(config.input_video_path).stem
    out_path = os.path.join(config.output_folder, f"{filename_noext}-[skeleton-3d-viewer].html")
    template_path = os.path.join(os.path.dirname(__file__), "viewer_template.html")
    write_viewer_html(build_payload(frames, raw_frames), template_path, out_path)
