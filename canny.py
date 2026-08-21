import logging
import os
from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm import tqdm

import config
from sam2_segment import Sam2Segmenter, build_config_from_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

def edge_detect(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """
    mask, if provided, is a (H, W) bool array. Edge detection still runs
    on the full frame (so gradients at the mask boundary are accurate),
    but the result is zeroed outside the mask afterward.
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    if config.use_clahe:
        clahe = cv.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=config.clahe_tile_grid_size
        )
        gray = clahe.apply(gray)

    blurred = cv.GaussianBlur(gray, config.gaussian_ksize, 0)

    canny_edges = cv.Canny(
        blurred,
        config.canny_low_threshold,
        config.canny_high_threshold
    )

    combined = canny_edges

    if config.use_sobel:
        sobel_x = cv.Sobel(blurred, cv.CV_32F, 1, 0, ksize=3)
        sobel_y = cv.Sobel(blurred, cv.CV_32F, 0, 1, ksize=3)

        magnitude = cv.magnitude(sobel_x, sobel_y)
        magnitude = cv.convertScaleAbs(magnitude)

        _, sobel_edges = cv.threshold(
            magnitude,
            config.sobel_threshold,
            255,
            cv.THRESH_BINARY
        )

        combined = cv.bitwise_or(canny_edges, sobel_edges)

    open_kernel = create_kernel(config.open_ksize)
    combined = cv.morphologyEx(combined, cv.MORPH_OPEN, open_kernel)

    close_kernel = create_kernel(config.close_ksize)
    combined = cv.morphologyEx(combined, cv.MORPH_CLOSE, close_kernel)

    dilate_kernel = create_kernel(config.dilation_ksize)
    combined = cv.dilate(combined, dilate_kernel, iterations=1)

    if config.edge_min_component_area > 0:
        combined = remove_small_components(combined, config.edge_min_component_area)

    if mask is not None:
        mask_u8 = (mask.astype(np.uint8)) * 255
        if config.sam2_mask_dilate_ksize is not None:
            dilate_kernel = create_kernel(config.sam2_mask_dilate_ksize)
            mask_u8 = cv.dilate(mask_u8, dilate_kernel, iterations=1)
        combined = cv.bitwise_and(combined, mask_u8)

    return combined


def remove_small_components(binary_image: np.ndarray, min_area: int) -> np.ndarray:
    """
    Drops connected white blobs smaller than min_area pixels from a
    binary (0/255) image. Used to strip speckle noise from edge maps
    while leaving real, larger contour lines untouched.
    """
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(binary_image, connectivity=8)
    cleaned = np.zeros_like(binary_image)
    for label in range(1, num_labels):  # label 0 is background
        if stats[label, cv.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def recolor(edge_image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """
    mask, if provided, controls how the SAM2 subject silhouette is
    rendered, per config.sam2_style:

    "filled": subject filled with line_color as a flat base (clean
        outer contour, no background noise), with internal edges
        (limb crossover etc.) drawn in bg_color on top so limbs stay
        visible when crossing in front of the torso or each other.

    "outline": inverted - background stays bg_color everywhere inside
        AND outside the subject. Only the silhouette's outline stroke
        and internal edges are drawn, both in line_color, giving pure
        line art constrained to the subject's shape.

    Without a mask, behaves as before: background everywhere, edges
    drawn in line_color.
    """
    height, width = edge_image.shape
    result = np.empty((height, width, 3), dtype=np.uint8)

    bg_bgr = bgr_from_rgb(config.bg_color_rgb)
    line_bgr = bgr_from_rgb(config.line_color_rgb)

    result[:] = bg_bgr

    if mask is not None:
        outline_mask = None
        if config.sam2_outline_thickness > 0:
            mask_u8 = (mask.astype(np.uint8)) * 255
            contours, _ = cv.findContours(mask_u8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            outline_mask = np.zeros((height, width), dtype=np.uint8)
            cv.drawContours(outline_mask, contours, -1, 255, thickness=config.sam2_outline_thickness)

        if config.sam2_style == "outline":
            # Background stays everywhere; only draw the outline stroke
            # and internal edges, both in line_color.
            if outline_mask is not None:
                result[outline_mask > 0] = line_bgr
            result[edge_image > 0] = line_bgr
        else:
            # "filled" (default): flat silhouette fill first, so the
            # whole subject reads as one clean shape even where
            # internal edges are faint or missed.
            result[mask] = line_bgr
            # Internal edges drawn in the background color, so they
            # read as "cut lines" separating limbs/torso within the
            # filled shape, rather than adding more purple on purple.
            result[edge_image > 0] = bg_bgr
            if outline_mask is not None:
                result[outline_mask > 0] = bg_bgr
    else:
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

    # --- SAM2 segmentation runs first, as its own pass over the whole
    # video, and is completely decoupled from the edge-detection loop
    # below. mask_iter is None when disabled, so the frame loop doesn't
    # need to branch on use_sam2 at all.
    mask_iter = None
    if config.use_sam2:
        logger.info("Running SAM2 segmentation pass...")
        segmenter = Sam2Segmenter(build_config_from_settings())
        segmenter.prepare_frames(input_path)
        if segmenter._num_frames != total_frames:
            raise RuntimeError(
                f"SAM2 extracted {segmenter._num_frames} frames but the video has "
                f"{total_frames} frames according to VideoCapture - these should always "
                f"match. Try deleting {config.sam2_frame_extraction_dir} and rerunning."
            )
        mask_iter = segmenter.propagate_masks()

    progress = tqdm(total=total_frames, desc="Processing frame...")

    while True:
        success, frame = video.read()
        if not success:
            break

        mask = next(mask_iter) if mask_iter is not None else None

        edges = edge_detect(frame, mask=mask)
        output_frame = recolor(edges, mask=mask)
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
    filename_noext = Path(config.input_video_path).stem
    output_video_path = os.path.join(
        config.output_folder,
        f"{filename_noext}-[balanced].mp4"
    )

    process_video(config.input_video_path, output_video_path)
