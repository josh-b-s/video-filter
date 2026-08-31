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


def trim_and_crop_video(
    input_path: str,
    trim_start_frame: int,
    trim_end_frame: int | None,
    crop_rect: tuple[int, int, int, int] | None,
    output_dir: str,
) -> str:
    """
    Writes a trimmed-and-cropped copy of input_path and returns its
    path. trim_start_frame/trim_end_frame are frame indices into the
    ORIGINAL video (before any FPS resampling) - this must run BEFORE
    resample_video_to_fps for that reason, since resampling changes
    which source frame ends up at which index. trim_end_frame=None
    means "to the end of the clip". crop_rect is (x1, y1, x2, y2) in
    the original video's pixel coordinates; None means no cropping.

    Returns input_path unchanged if there's nothing to do (full clip,
    no crop), to avoid a pointless re-encode pass - same convention as
    resample_video_to_fps.
    """
    video = cv.VideoCapture(input_path)
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = float(video.get(cv.CAP_PROP_FPS))
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))

    start = max(0, trim_start_frame)
    end = total_frames if trim_end_frame is None else min(trim_end_frame, total_frames)

    no_trim = (start == 0 and end == total_frames)
    no_crop = crop_rect is None

    if no_trim and no_crop:
        video.release()
        logger.info("No trim/crop configured - using original video as-is.")
        return input_path

    if end <= start:
        video.release()
        raise ValueError(f"trim_end_frame ({end}) must be greater than trim_start_frame ({start}).")

    if crop_rect is not None:
        x1, y1, x2, y2 = crop_rect
        x1, x2 = sorted((max(0, x1), min(width, x2)))
        y1, y2 = sorted((max(0, y1), min(height, y2)))
        if x2 <= x1 or y2 <= y1:
            video.release()
            raise ValueError(f"crop_rect {crop_rect} is empty after clamping to frame size {width}x{height}.")
        out_width, out_height = x2 - x1, y2 - y1
    else:
        x1 = y1 = 0
        out_width, out_height = width, height

    os.makedirs(output_dir, exist_ok=True)
    filename_noext = Path(input_path).stem
    output_path = os.path.join(output_dir, f"{filename_noext}-trimmed.mp4")

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(output_path, fourcc, fps, (out_width, out_height), True)
    if not writer.isOpened():
        video.release()
        raise RuntimeError(f"Could not create trimmed/cropped video: {output_path}")

    video.set(cv.CAP_PROP_POS_FRAMES, start)
    written = 0
    progress = tqdm(total=end - start, desc="Trimming/cropping...")
    for _ in range(end - start):
        success, frame = video.read()
        if not success:
            break
        if crop_rect is not None:
            frame = frame[y1:y2, x1:x2]
        writer.write(frame)
        written += 1
        progress.update(1)
    progress.close()

    writer.release()
    video.release()

    logger.info(
        f"Trimmed to frames [{start}, {end}) ({written} written)"
        + (f", cropped to {out_width}x{out_height} at ({x1},{y1})" if crop_rect is not None else "")
        + f" -> {output_path}"
    )
    return output_path


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
