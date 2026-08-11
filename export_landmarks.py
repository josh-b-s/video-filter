"""
Exports the same 3D world-landmark points that pose_skeleton.py renders,
but instead of baking them into fixed-camera matplotlib videos, dumps them
into a single self-contained HTML file you can open in any browser and
freely rotate / zoom / scrub through frames (via Plotly.js).

Usage:
    python export_landmarks.py

Reads:  input_video_path (from pose_skeleton.py config)
Writes: output/<video-name>-[skeleton-3d-viewer].html
"""

import json
import logging
import os
from pathlib import Path

import cv2 as cv
from tqdm import tqdm

from mediapipe.tasks.python.vision import PoseLandmarksConnections

from pose_skeleton import (
    BONE_CONNECTIONS,
    SkeletonRenderer3D,
    input_video_path,
    make_sure_file_exists,
    min_landmark_visibility,
    output_folder,
)

# Set to True to include ALL 33 landmarks (including the face: nose,
# eyes, ears, mouth) instead of the face-stripped set used by
# pose_skeleton.py. Only meant for your own local viewing.
SHOW_ALL_LANDMARKS = True

ALL_CONNECTIONS = [
    (c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS
]
ALL_LANDMARKS = sorted({i for pair in ALL_CONNECTIONS for i in pair})

ACTIVE_CONNECTIONS = ALL_CONNECTIONS if SHOW_ALL_LANDMARKS else BONE_CONNECTIONS
ACTIVE_LANDMARKS = ALL_LANDMARKS if SHOW_ALL_LANDMARKS else None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_all_points(renderer: SkeletonRenderer3D, frame) -> dict:
    """Like SkeletonRenderer3D._compute_points, but keeps every landmark
    in ACTIVE_LANDMARKS instead of the face-stripped BODY_LANDMARKS."""
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    from mediapipe import Image, ImageFormat
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

    timestamp_ms = int(renderer._frame_index * renderer._frame_duration_ms)
    renderer._frame_index += 1

    result = renderer.landmarker.detect_for_video(mp_image, timestamp_ms)

    points: dict = {}
    if not result.pose_world_landmarks:
        return points

    landmarks = result.pose_world_landmarks[0]
    for idx in ACTIVE_LANDMARKS:
        lm = landmarks[idx]
        visibility = getattr(lm, "visibility", 1.0)
        if visibility is not None and visibility < min_landmark_visibility:
            continue
        points[idx] = renderer._smooth_point(idx, lm.x, -lm.z, -lm.y)

    return points


def extract_frames(input_path: str) -> list[dict]:
    make_sure_file_exists(input_path)

    video = cv.VideoCapture(input_path)
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = float(video.get(cv.CAP_PROP_FPS))
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    logger.info(f"fps: {fps}; total_frames: {total_frames}")

    renderer = SkeletonRenderer3D(fps)
    frames_out = []

    progress = tqdm(total=total_frames, desc="Extracting 3D landmarks...")
    while True:
        success, frame = video.read()
        if not success:
            break

        points = (
            compute_all_points(renderer, frame)
            if SHOW_ALL_LANDMARKS
            else renderer._compute_points(frame)
        )
        # JSON needs string keys; landmark indices become strings, plus
        # the synthetic "neck"/"head" keys already are strings.
        frames_out.append({str(k): list(v) for k, v in points.items()})

        progress.update(1)

    progress.close()
    renderer.close()
    video.release()
    return frames_out


def build_html(frames: list[dict], out_path: str) -> None:
    payload = {
        "frames": frames,
        "bones": ACTIVE_CONNECTIONS,
        "hasHead": not SHOW_ALL_LANDMARKS,
    }

    template_path = os.path.join(os.path.dirname(__file__), "viewer_template.html")
    with open(template_path, "r") as f:
        template = f.read()

    html = template.replace("__SKELETON_DATA__", json.dumps(payload))

    with open(out_path, "w") as f:
        f.write(html)

    logger.info(f"Viewer written to: {out_path}")


if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)
    frames = extract_frames(input_video_path)
    filename_noext = Path(input_video_path).stem
    out_path = os.path.join(output_folder, f"{filename_noext}-[skeleton-3d-viewer].html")
    build_html(frames, out_path)