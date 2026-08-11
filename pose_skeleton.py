import logging
import os
import urllib.request
from pathlib import Path

import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
    RunningMode,
)
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================

input_video_path = "materials/videos/test.mp4"
output_folder = "output"

# Output render resolution (independent of source video resolution)
render_width = 960
render_height = 960

bg_color_rgb = (20 / 255, 23 / 255, 31 / 255)          # #14171F
line_color_rgb = (155 / 255, 111 / 255, 224 / 255)     # mobility purple
joint_color_rgb = (230 / 255, 220 / 255, 250 / 255)    # bright highlight

bone_linewidth = 6
joint_size = 90

# Fixed camera views instead of an orbit. azim is degrees around the
# vertical axis; 0/90/180/270 are 90 degrees apart from each other.
# "front" is set to roughly match how the person faced the source
# camera. If front/back or left/right come out swapped or mirrored for
# your footage, just adjust the degree values below and re-run -- it's
# a matter of which way the subject happened to be facing on camera.
camera_elevation = 12
camera_distance = 2.2  # axes half-range in meters, roughly torso-to-edge
VIEW_AZIMUTHS = {
    "front": -90,
    "right": 0,
    "back": 90,
    "left": 180,
}

# Synthetic head marker (no face landmarks exist since we strip 0-10).
# Placed above the shoulder midpoint by this many meters.
head_offset_up = 0.28
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

model_dir = "models"
model_filename = "pose_landmarker_lite.task"
model_url = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

min_detection_confidence = 0.5
min_tracking_confidence = 0.5
min_presence_confidence = 0.5
min_landmark_visibility = 0.5

# Smoothing across frames (reduces jitter). 0 = off, higher = smoother/laggier.
smoothing_alpha = 0.4

# MediaPipe's 33 pose landmarks, indexed 0-32. We exclude the face
# (indices 0-10) entirely so nothing resembling a face is ever drawn.
FACE_LANDMARKS = set(range(0, 11))

_RAW_CONNECTIONS = [
    (c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS
]
BONE_CONNECTIONS = [
    (a, b) for (a, b) in _RAW_CONNECTIONS
    if a not in FACE_LANDMARKS and b not in FACE_LANDMARKS
]

BODY_LANDMARKS = sorted({i for pair in BONE_CONNECTIONS for i in pair})


# ============================================================
# HELPERS
# ============================================================

def make_sure_file_exists(file_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")


def ensure_model_downloaded() -> str:
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_filename)

    if not os.path.exists(model_path):
        logger.info(f"Downloading pose landmarker model to: {model_path}")
        urllib.request.urlretrieve(model_url, model_path)
        logger.info("Model download complete.")

    return model_path


# ============================================================
# 3D SKELETON PIPELINE
# ============================================================

class SkeletonRenderer3D:
    """Runs MediaPipe Pose Landmarker on frames, reads the metric 3D
    world_landmarks (hip-centered, in meters), and renders a smoothed,
    face-free 3D stick-figure. render_all_views() draws the same pose
    from several fixed camera angles (front/left/right/back) so each
    can be written to its own output video."""

    def __init__(self, fps: float):
        model_path = ensure_model_downloaded()

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

        self._frame_index = 0
        self._frame_duration_ms = 1000.0 / fps if fps > 0 else 33.0

        self._smoothed: dict[int, tuple[float, float, float]] = {}

        # Persistent figure/axes reused every frame for speed.
        self.fig = plt.figure(figsize=(render_width / 100, render_height / 100), dpi=100)
        self.fig.patch.set_facecolor(bg_color_rgb)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def close(self) -> None:
        self.landmarker.close()
        plt.close(self.fig)

    def _smooth_point(self, idx: int, x: float, y: float, z: float) -> tuple[float, float, float]:
        if smoothing_alpha <= 0 or idx not in self._smoothed:
            self._smoothed[idx] = (x, y, z)
        else:
            px, py, pz = self._smoothed[idx]
            self._smoothed[idx] = (
                px + smoothing_alpha * (x - px),
                py + smoothing_alpha * (y - py),
                pz + smoothing_alpha * (z - pz),
            )
        return self._smoothed[idx]

    def _compute_points(self, frame: np.ndarray) -> dict:
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(self._frame_index * self._frame_duration_ms)
        self._frame_index += 1

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        points: dict[int, tuple[float, float, float]] = {}
        if not result.pose_world_landmarks:
            return points

        landmarks = result.pose_world_landmarks[0]
        for idx in BODY_LANDMARKS:
            lm = landmarks[idx]
            visibility = getattr(lm, "visibility", 1.0)
            if visibility is not None and visibility < min_landmark_visibility:
                continue
            # MediaPipe world space: x right, y down, z toward camera.
            # matplotlib's 3rd coordinate is the vertical (up) axis on
            # screen, so height (-y) must go last; depth (-z) goes into
            # the axis the camera orbits/views around.
            points[idx] = self._smooth_point(idx, lm.x, -lm.z, -lm.y)

        # Synthetic head marker + neck bone, since face landmarks
        # (0-10) are excluded entirely and there's nothing else to
        # anchor a head to.
        if LEFT_SHOULDER in points and RIGHT_SHOULDER in points:
            lx, ly, lz = points[LEFT_SHOULDER]
            rx, ry, rz = points[RIGHT_SHOULDER]
            neck = ((lx + rx) / 2, (ly + ry) / 2, (lz + rz) / 2)
            head = (neck[0], neck[1], neck[2] + head_offset_up)
            points["neck"] = neck
            points["head"] = head

        return points

    def _draw(self, points: dict, azimuth: float) -> np.ndarray:
        ax = self.ax
        ax.clear()
        ax.set_facecolor(bg_color_rgb)
        d = camera_distance
        ax.set_xlim(-d, d)
        ax.set_ylim(-d, d)
        ax.set_zlim(-d, d)
        ax.set_box_aspect((1, 1, 1))
        ax.axis("off")
        ax.view_init(elev=camera_elevation, azim=azimuth)

        if points:
            if "head" in points:
                nx, ny, nz = points["neck"]
                hx, hy, hz = points["head"]
                ax.plot(
                    [nx, hx], [ny, hy], [nz, hz],
                    color=line_color_rgb, linewidth=bone_linewidth,
                    solid_capstyle="round",
                )

            for (a, b) in BONE_CONNECTIONS:
                if a in points and b in points:
                    xa, ya, za = points[a]
                    xb, yb, zb = points[b]
                    ax.plot(
                        [xa, xb], [ya, yb], [za, zb],
                        color=line_color_rgb, linewidth=bone_linewidth,
                        solid_capstyle="round",
                    )

            body_points = [v for k, v in points.items() if k != "neck" and k != "head"]
            if body_points:
                xs, ys, zs = zip(*body_points)
                ax.scatter(xs, ys, zs, color=joint_color_rgb, s=joint_size, depthshade=False)
            if "head" in points:
                hx, hy, hz = points["head"]
                ax.scatter([hx], [hy], [hz], color=joint_color_rgb, s=joint_size * 4, depthshade=False)

        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        rgb_out = buf[:, :, :3]
        return cv.cvtColor(rgb_out, cv.COLOR_RGB2BGR)

    def render_all_views(self, frame: np.ndarray) -> dict:
        """Returns {view_name: rendered_bgr_frame} for every entry in
        VIEW_AZIMUTHS, all from the same detected pose for this frame."""
        points = self._compute_points(frame)
        return {
            name: self._draw(points, azim)
            for name, azim in VIEW_AZIMUTHS.items()
        }


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(input_path: str, output_folder_path: str) -> None:
    make_sure_file_exists(input_path)
    os.makedirs(output_folder_path, exist_ok=True)

    video = cv.VideoCapture(input_path)
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = float(video.get(cv.CAP_PROP_FPS))
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))

    logger.info(
        f"fps: {fps}; total_frames: {total_frames}; "
        f"render size: {render_width}x{render_height}; views: {list(VIEW_AZIMUTHS)}."
    )

    filename_noext = Path(input_path).stem
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writers = {}
    for view_name in VIEW_AZIMUTHS:
        out_path = os.path.join(
            output_folder_path, f"{filename_noext}-[skeleton-3d-{view_name}].mp4"
        )
        writer = cv.VideoWriter(out_path, fourcc, fps, (render_width, render_height), True)
        if not writer.isOpened():
            video.release()
            for w in writers.values():
                w.release()
            raise RuntimeError(f"Could not create output video: {out_path}")
        writers[view_name] = (writer, out_path)

    renderer = SkeletonRenderer3D(fps)
    progress = tqdm(total=total_frames, desc="Rendering 3D frames (4 views)...")

    while True:
        success, frame = video.read()
        if not success:
            break

        view_frames = renderer.render_all_views(frame)
        for view_name, (writer, _) in writers.items():
            writer.write(view_frames[view_name])
        progress.update(1)

    progress.close()
    renderer.close()
    for writer, out_path in writers.values():
        writer.release()
        logger.info(f"Video written to: {out_path}")
    video.release()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    process_video(input_video_path, output_folder)