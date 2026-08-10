import logging
import os
from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================

input_video_path = "materials/videos/ikun.mp4"
output_folder = "output"

# Preset 3 = balanced
bg_color_rgb = [237, 244, 247]
line_color_rgb = [155, 111, 224]

use_clahe = True
clahe_clip_limit = 1.5
clahe_tile_grid_size = (8, 8)

gaussian_ksize = (5, 5)

canny_low_threshold = 40
canny_high_threshold = 120

sobel_threshold = 60
use_sobel = True

open_ksize = (2, 2)
close_ksize = (3, 3)
dilation_ksize = (2, 2)


# ============================================================
# HELPERS
# ============================================================

def make_sure_file_exists(file_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")


def bgr_from_rgb(rgb: list[int]) -> np.ndarray:
    return np.array(rgb[::-1], dtype=np.uint8)


def create_kernel(size: tuple[int, int]) -> np.ndarray:
    return np.ones(size, dtype=np.uint8)


# ============================================================
# EDGE PIPELINE
# ============================================================

def edge_detect(image: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    if use_clahe:
        clahe = cv.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid_size
        )
        gray = clahe.apply(gray)

    blurred = cv.GaussianBlur(gray, gaussian_ksize, 0)

    canny_edges = cv.Canny(
        blurred,
        canny_low_threshold,
        canny_high_threshold
    )

    combined = canny_edges

    if use_sobel:
        sobel_x = cv.Sobel(blurred, cv.CV_32F, 1, 0, ksize=3)
        sobel_y = cv.Sobel(blurred, cv.CV_32F, 0, 1, ksize=3)

        magnitude = cv.magnitude(sobel_x, sobel_y)
        magnitude = cv.convertScaleAbs(magnitude)

        _, sobel_edges = cv.threshold(
            magnitude,
            sobel_threshold,
            255,
            cv.THRESH_BINARY
        )

        combined = cv.bitwise_or(canny_edges, sobel_edges)

    open_kernel = create_kernel(open_ksize)
    combined = cv.morphologyEx(combined, cv.MORPH_OPEN, open_kernel)

    close_kernel = create_kernel(close_ksize)
    combined = cv.morphologyEx(combined, cv.MORPH_CLOSE, close_kernel)

    dilate_kernel = create_kernel(dilation_ksize)
    combined = cv.dilate(combined, dilate_kernel, iterations=1)

    return combined


def recolor(edge_image: np.ndarray) -> np.ndarray:
    height, width = edge_image.shape
    result = np.empty((height, width, 3), dtype=np.uint8)

    bg_bgr = bgr_from_rgb(bg_color_rgb)
    line_bgr = bgr_from_rgb(line_color_rgb)

    result[:] = bg_bgr
    result[edge_image > 0] = line_bgr

    return result


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(input_path: str, output_path: str) -> None:
    make_sure_file_exists(input_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    video = cv.VideoCapture(input_path)
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = float(video.get(cv.CAP_PROP_FPS))
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))

    logger.info(
        f"width: {width}; height: {height}; fps: {fps}; total_frames: {total_frames}."
    )

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(output_path, fourcc, fps, (width, height), True)

    if not writer.isOpened():
        video.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    progress = tqdm(total=total_frames, desc="Processing frame...")

    while True:
        success, frame = video.read()
        if not success:
            break

        edges = edge_detect(frame)
        output_frame = recolor(edges)
        writer.write(output_frame)
        progress.update(1)

    progress.close()
    writer.release()
    video.release()

    logger.info(f"Video written to: {output_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    filename_noext = Path(input_video_path).stem
    output_video_path = os.path.join(
        output_folder,
        f"{filename_noext}-[balanced].mp4"
    )

    process_video(input_video_path, output_video_path)