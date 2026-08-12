"""Thin wrapper around MediaPipe's PoseLandmarker: handles model
download, landmarker setup, and per-frame VIDEO-mode timestamp
bookkeeping. This is the only module that talks to MediaPipe directly."""

import logging
import os
import urllib.request

import cv2 as cv
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

import config

logger = logging.getLogger(__name__)


def ensure_model_downloaded() -> str:
    os.makedirs(config.model_dir, exist_ok=True)
    model_path = os.path.join(config.model_dir, config.model_filename)

    if not os.path.exists(model_path):
        logger.info(f"Downloading pose landmarker model to: {model_path}")
        urllib.request.urlretrieve(config.model_url, model_path)
        logger.info("Model download complete.")

    return model_path


class PoseDetector:
    """Runs MediaPipe Pose Landmarker over a video, frame by frame,
    handling the model lifecycle and VIDEO-mode timestamp math so
    callers just hand it BGR frames in order."""

    def __init__(self, fps: float):
        model_path = ensure_model_downloaded()
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=config.min_detection_confidence,
            min_pose_presence_confidence=config.min_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._frame_index = 0
        self._frame_duration_ms = 1000.0 / fps if fps > 0 else 33.0

    def detect(self, frame_bgr):
        """Runs detection on a single BGR video frame and returns the
        MediaPipe PoseLandmarkerResult for it."""
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(self._frame_index * self._frame_duration_ms)
        self._frame_index += 1
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def close(self) -> None:
        self._landmarker.close()
