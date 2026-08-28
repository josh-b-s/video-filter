"""
Responsibility: turn a frame (+ optional SAM2 mask) into the final
recolored edge-art frame. Knows nothing about video I/O, SAM2, or
face detection specifics - it calls face_flatten.flatten_face_region
as one step, with the enabled flag passed in explicitly by the caller.
"""

import numpy as np
import cv2 as cv

import config
from core import face_flatten


def bgr_from_rgb(rgb: list[int]) -> np.ndarray:
    return np.array(rgb[::-1], dtype=np.uint8)


def create_kernel(size: tuple[int, int]) -> np.ndarray:
    return np.ones(size, dtype=np.uint8)


def remove_small_components(binary_image: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(binary_image, connectivity=8)
    cleaned = np.zeros_like(binary_image)
    for label in range(1, num_labels):
        if stats[label, cv.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def edge_detect(image: np.ndarray, mask: np.ndarray | None = None, face_removal_enabled: bool = True) -> np.ndarray:
    """
    mask, if provided, is a (H, W) bool array. Edge detection still runs
    on the full frame (so gradients at the mask boundary are accurate),
    but the result is zeroed outside the mask afterward.
    """
    image = face_flatten.flatten_face_region(image, enabled=face_removal_enabled, sam2_mask=mask)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    if config.use_clahe:
        clahe = cv.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=config.clahe_tile_grid_size
        )
        gray = clahe.apply(gray)

    blurred = cv.GaussianBlur(gray, config.gaussian_ksize, 0)

    canny_edges = cv.Canny(blurred, config.canny_low_threshold, config.canny_high_threshold)
    combined = canny_edges

    if config.use_sobel:
        sobel_x = cv.Sobel(blurred, cv.CV_32F, 1, 0, ksize=3)
        sobel_y = cv.Sobel(blurred, cv.CV_32F, 0, 1, ksize=3)
        magnitude = cv.magnitude(sobel_x, sobel_y)
        magnitude = cv.convertScaleAbs(magnitude)
        _, sobel_edges = cv.threshold(magnitude, config.sobel_threshold, 255, cv.THRESH_BINARY)
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


def recolor(edge_image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
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
            if outline_mask is not None:
                result[outline_mask > 0] = line_bgr
            result[edge_image > 0] = line_bgr
        else:
            result[mask] = line_bgr
            result[edge_image > 0] = bg_bgr
            if outline_mask is not None:
                result[outline_mask > 0] = bg_bgr
    else:
        result[edge_image > 0] = line_bgr

    return result
