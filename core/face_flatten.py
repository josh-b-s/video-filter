"""
Responsibility: find the subject's face in a frame and flatten it to a
single flat color before edge detection ever sees it, so there is
guaranteed zero internal gradient there.

Moved out of canny.py unchanged in behavior. See config.py for the
tunable settings this module reads (face_detection_confidence,
face_search_tile_grid, use_face_tracking, etc).

Every function here takes an explicit `enabled: bool` (the per-video
face-removal toggle from job_store) rather than reading a single
global switch, so multiple videos with different toggles can be
processed in the same run without stepping on each other's config.
"""

import logging
import os

import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
    FaceLandmarksConnections,
)
from mediapipe.tasks.python.core.base_options import BaseOptions

import config

logger = logging.getLogger(__name__)

_FACE_OVAL_INDICES = sorted({
    idx
    for connection in FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL
    for idx in (connection.start, connection.end)
})

_face_landmarker = None

_face_detect_attempts = 0
_face_detect_hits = 0

_face_tracker = None
_face_tracker_miss_streak = 0


def reset_tracker_state() -> None:
    """Call at the start of each new video/chunk so a stale tracker
    box from a previous clip is never reused."""
    global _face_tracker, _face_tracker_miss_streak, _face_detect_attempts, _face_detect_hits
    _face_tracker = None
    _face_tracker_miss_streak = 0
    _face_detect_attempts = 0
    _face_detect_hits = 0


def _create_csrt_tracker():
    attempts = [
        ("cv2.TrackerCSRT.create()", lambda: cv.TrackerCSRT.create()),
        ("cv2.TrackerCSRT_create()", lambda: cv.TrackerCSRT_create()),
        ("cv2.legacy.TrackerCSRT.create()", lambda: cv.legacy.TrackerCSRT.create()),
        ("cv2.legacy.TrackerCSRT_create()", lambda: cv.legacy.TrackerCSRT_create()),
    ]
    errors = []
    for name, factory in attempts:
        try:
            return factory()
        except AttributeError as e:
            errors.append(f"{name}: {e}")
            continue
    raise RuntimeError(
        "Could not create a CSRT tracker with any known OpenCV API shape. "
        "This requires opencv-contrib-python (not plain opencv-python). "
        "Tried:\n" + "\n".join(errors)
    )


def _get_face_landmarker():
    global _face_landmarker
    if _face_landmarker is None:
        if not os.path.exists(config.face_landmarker_model_path):
            raise FileNotFoundError(
                f"Face landmarker model not found at "
                f"'{config.face_landmarker_model_path}'. Download it from "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task and point "
                "config.face_landmarker_model_path at it, or untick face "
                "removal for this video."
            )
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.face_landmarker_model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=config.face_detection_confidence,
            min_face_presence_confidence=config.face_detection_confidence,
        )
        _face_landmarker = FaceLandmarker.create_from_options(options)
    return _face_landmarker


def detect_face_oval_points(image_bgr: np.ndarray, count_attempt: bool = True) -> np.ndarray | None:
    global _face_detect_attempts, _face_detect_hits

    landmarker = _get_face_landmarker()
    height, width = image_bgr.shape[:2]

    image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)

    if count_attempt:
        _face_detect_attempts += 1

    if not result.face_landmarks:
        return None

    if count_attempt:
        _face_detect_hits += 1

    landmarks = result.face_landmarks[0]
    return np.array([
        (landmarks[i].x * width, landmarks[i].y * height)
        for i in _FACE_OVAL_INDICES
    ], dtype=np.float32)


def _make_person_tiles(
    person_bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    x0, y0, x1, y1 = person_bbox
    frame_h, frame_w = frame_shape

    box_w, box_h = x1 - x0, y1 - y0
    rows, cols = config.face_search_tile_grid
    overlap = config.face_search_tile_overlap

    tile_w = box_w / cols
    tile_h = box_h / rows
    pad_w = tile_w * overlap
    pad_h = tile_h * overlap

    tiles = []
    for row in range(rows):
        for col in range(cols):
            tx0 = x0 + col * tile_w - pad_w
            tx1 = x0 + (col + 1) * tile_w + pad_w
            ty0 = y0 + row * tile_h - pad_h
            ty1 = y0 + (row + 1) * tile_h + pad_h

            tx0 = max(0, int(tx0))
            ty0 = max(0, int(ty0))
            tx1 = min(frame_w, int(tx1))
            ty1 = min(frame_h, int(ty1))

            if tx1 > tx0 and ty1 > ty0:
                tiles.append((tx0, ty0, tx1, ty1))
    return tiles


def get_head_crop(image_bgr: np.ndarray, sam2_mask: np.ndarray):
    ys, xs = np.nonzero(sam2_mask)
    if len(xs) == 0:
        return None

    height, width = image_bgr.shape[:2]
    person_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    tiles = _make_person_tiles(person_bbox, (height, width))

    for tx0, ty0, tx1, ty1 in tiles:
        crop = image_bgr[ty0:ty1, tx0:tx1]
        crop_h, crop_w = crop.shape[:2]
        if crop_h == 0 or crop_w == 0:
            continue

        scale = config.head_crop_upscale_target / max(crop_h, crop_w)
        scale = max(scale, 1.0)
        new_size = (int(round(crop_w * scale)), int(round(crop_h * scale)))
        crop_upscaled = cv.resize(crop, new_size, interpolation=cv.INTER_CUBIC)

        local_points = detect_face_oval_points(crop_upscaled, count_attempt=False)
        if local_points is not None:
            return local_points, tx0, ty0, scale, scale
    return None


def _get_tracked_crop(image_bgr: np.ndarray):
    global _face_tracker, _face_tracker_miss_streak

    if _face_tracker is None:
        return None

    height, width = image_bgr.shape[:2]
    success, box = _face_tracker.update(image_bgr)
    if not success:
        _face_tracker = None
        _face_tracker_miss_streak = 0
        return None

    bx, by, bw, bh = box
    pad_x, pad_y = bw * 0.3, bh * 0.3
    tx0 = max(0, int(bx - pad_x))
    ty0 = max(0, int(by - pad_y))
    tx1 = min(width, int(bx + bw + pad_x))
    ty1 = min(height, int(by + bh + pad_y))

    if tx1 <= tx0 or ty1 <= ty0:
        _face_tracker = None
        _face_tracker_miss_streak = 0
        return None

    crop = image_bgr[ty0:ty1, tx0:tx1]
    crop_h, crop_w = crop.shape[:2]
    scale = config.head_crop_upscale_target / max(crop_h, crop_w)
    scale = max(scale, 1.0)
    new_size = (int(round(crop_w * scale)), int(round(crop_h * scale)))
    crop_upscaled = cv.resize(crop, new_size, interpolation=cv.INTER_CUBIC)

    local_points = detect_face_oval_points(crop_upscaled, count_attempt=False)
    if local_points is None:
        _face_tracker_miss_streak += 1
        if _face_tracker_miss_streak >= config.face_tracker_max_miss_streak:
            _face_tracker = None
            _face_tracker_miss_streak = 0
        return None

    _face_tracker_miss_streak = 0
    return local_points, tx0, ty0, scale, scale


def _seed_tracker(image_bgr: np.ndarray, oval_points: np.ndarray) -> None:
    global _face_tracker, _face_tracker_miss_streak

    x0, y0 = oval_points.min(axis=0)
    x1, y1 = oval_points.max(axis=0)
    pad_x, pad_y = (x1 - x0) * 0.2, (y1 - y0) * 0.2
    box = (
        int(max(0, x0 - pad_x)),
        int(max(0, y0 - pad_y)),
        int((x1 - x0) + 2 * pad_x),
        int((y1 - y0) + 2 * pad_y),
    )
    try:
        tracker = _create_csrt_tracker()
        tracker.init(image_bgr, box)
        _face_tracker = tracker
        _face_tracker_miss_streak = 0
    except RuntimeError as e:
        if config.use_face_tracking:
            logger.warning(
                "CSRT tracker unavailable (%s) - falling back to full "
                "tile-grid search every frame.", e,
            )


def get_face_oval_mask(image_bgr: np.ndarray, sam2_mask: np.ndarray | None = None) -> np.ndarray | None:
    global _face_detect_attempts, _face_detect_hits

    height, width = image_bgr.shape[:2]

    crop_info = None
    used_tracker = False
    if config.use_face_tracking:
        crop_info = _get_tracked_crop(image_bgr)
        used_tracker = crop_info is not None

    if crop_info is None and sam2_mask is not None:
        crop_info = get_head_crop(image_bgr, sam2_mask)

    if crop_info is not None:
        local_points, crop_x0, crop_y0, scale_x, scale_y = crop_info
        oval_points = np.empty_like(local_points)
        oval_points[:, 0] = crop_x0 + local_points[:, 0] / scale_x
        oval_points[:, 1] = crop_y0 + local_points[:, 1] / scale_y
        _face_detect_attempts += 1
        _face_detect_hits += 1

        if config.use_face_tracking and not used_tracker:
            _seed_tracker(image_bgr, oval_points)
    else:
        oval_points = detect_face_oval_points(image_bgr)
        if oval_points is None:
            return None

    hull = cv.convexHull(oval_points).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv.fillConvexPoly(mask, hull, 255)
    return mask


def log_face_detection_summary(video_label: str = "") -> None:
    if _face_detect_attempts == 0:
        return
    logger.info(
        f"Face detection{f' ({video_label})' if video_label else ''}: "
        f"{_face_detect_hits}/{_face_detect_attempts} frames had a face "
        f"detected (confidence >= {config.face_detection_confidence})."
    )
    if _face_detect_hits == 0:
        logger.warning(
            "Face was never detected in any frame - face removal had no "
            "effect on this video. Try lowering config.face_detection_confidence "
            "(currently %.2f) or check the face is reasonably visible.",
            config.face_detection_confidence,
        )


def flatten_face_region(
    image_bgr: np.ndarray, enabled: bool, sam2_mask: np.ndarray | None = None
) -> np.ndarray:
    """Returns image_bgr unchanged if `enabled` is False or no face is found."""
    if not enabled:
        return image_bgr

    mask = get_face_oval_mask(image_bgr, sam2_mask)
    if mask is None:
        return image_bgr

    mean_color = cv.mean(image_bgr, mask=mask)[:3]
    flat_fill = np.full_like(image_bgr, mean_color, dtype=np.uint8)

    feather_ksize = config.face_blur_feather_ksize
    mask_feathered = cv.GaussianBlur(mask, feather_ksize, 0).astype(np.float32) / 255.0
    mask_feathered = mask_feathered[:, :, np.newaxis]

    blended = (
        flat_fill.astype(np.float32) * mask_feathered
        + image_bgr.astype(np.float32) * (1.0 - mask_feathered)
    )
    return blended.astype(np.uint8)
