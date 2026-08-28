"""
Responsibility: video-level file I/O that isn't specific to SAM2 or
edge detection - FPS resampling and stitching processed chunks back
into one file.
"""

import logging
import os
from pathlib import Path

import cv2 as cv
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)


def resample_video_to_fps(input_path: str, target_fps: float, output_dir: str) -> str:
    """
    Writes a frame-dropped copy of input_path at target_fps and
    returns its path. Frames are dropped using an accumulator so drops
    spread evenly rather than drifting. Returns input_path unchanged if
    target_fps >= the source fps.
    """
    video = cv.VideoCapture(input_path)
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(video.get(cv.CAP_PROP_FPS))
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))

    if target_fps >= source_fps:
        video.release()
        logger.info(
            f"target_fps ({target_fps}) >= source fps ({source_fps}) - "
            f"skipping FPS resampling, using original video as-is."
        )
        return input_path

    os.makedirs(output_dir, exist_ok=True)
    filename_noext = Path(input_path).stem
    output_path = os.path.join(output_dir, f"{filename_noext}-{target_fps:g}fps.mp4")

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(output_path, fourcc, target_fps, (width, height), True)
    if not writer.isOpened():
        video.release()
        raise RuntimeError(f"Could not create resampled video: {output_path}")

    accumulator = 0.0
    written = 0

    progress = tqdm(total=total_frames, desc="Resampling FPS...")
    while True:
        success, frame = video.read()
        if not success:
            break
        accumulator += target_fps
        if accumulator >= source_fps:
            accumulator -= source_fps
            writer.write(frame)
            written += 1
        progress.update(1)
    progress.close()

    writer.release()
    video.release()

    logger.info(f"Resampled {total_frames} frames at {source_fps:.2f}fps -> {written} frames at {target_fps:g}fps -> {output_path}")
    return output_path


def concatenate_chunks(chunk_paths: list[str], output_path: str, fps: float, width: int, height: int) -> None:
    """Re-encodes a sequence of chunk video files, in order, into one file."""
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(output_path, fourcc, fps, (width, height), True)

    for chunk_path in chunk_paths:
        chunk_video = cv.VideoCapture(chunk_path)
        while True:
            success, frame = chunk_video.read()
            if not success:
                break
            writer.write(frame)
        chunk_video.release()

    writer.release()


def get_video_info(path: str) -> dict:
    video = cv.VideoCapture(path)
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    info = dict(
        width=int(video.get(cv.CAP_PROP_FRAME_WIDTH)),
        height=int(video.get(cv.CAP_PROP_FRAME_HEIGHT)),
        fps=float(video.get(cv.CAP_PROP_FPS)),
        total_frames=int(video.get(cv.CAP_PROP_FRAME_COUNT)),
    )
    video.release()
    return info


def read_first_frame(path: str):
    video = cv.VideoCapture(path)
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    success, frame = video.read()
    video.release()
    if not success:
        raise RuntimeError(f"Could not read frame 0 from video: {path}")
    return frame
