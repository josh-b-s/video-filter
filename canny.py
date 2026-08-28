import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm import tqdm

import config
# NOTE: sam2_segment (and therefore torch/CUDA) is intentionally NOT
# imported at module level - it's imported lazily, only inside the
# one code path that needs it (the SAM2 pass in process_video, which
# always runs in the main process). Worker processes spawned for the
# chunked edge-detection pass re-import this whole module from
# scratch (multiprocessing on Windows uses 'spawn'), so a top-level
# import here would mean every worker also loads torch's full CUDA
# DLL stack for no reason - workers never call anything SAM2-related.
# With enough workers running simultaneously, that's enough
# concurrent CUDA DLL loading to exhaust the system paging file
# (WinError 1455) even though none of them actually use the GPU.

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
# FACE REGION FLATTENING
# ============================================================
# Replaces the actual face-contour region (not a rough ellipse) with
# a single flat color, sampled from the face's own average skin tone,
# BEFORE edge detection runs - so there is exactly zero internal
# gradient there for Canny/Sobel to find, guaranteed, rather than
# relying on a strong-enough blur to suppress detail. The head's
# OUTER silhouette (hairline, jaw, chin, ears) is completely
# untouched, since that comes from the SAM2 mask in recolor(), not
# from this step - only the frame content inside the face oval is
# replaced before Canny/Sobel ever see it.
#
# Uses MediaPipe's FaceLandmarker (the modern "Tasks" API - the older
# "Solutions" API this project might have referenced before has been
# dropped from recent mediapipe pip builds). FaceLandmarker returns a
# full 478-point face mesh per detected face, from which we take the
# predefined face-oval landmark subset to build an actual face-shaped
# mask via convex hull, rather than approximating with an ellipse.

import mediapipe as mp
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
    FaceLandmarksConnections,
)
from mediapipe.tasks.python.core.base_options import BaseOptions

_FACE_OVAL_INDICES = sorted({
    idx
    for connection in FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL
    for idx in (connection.start, connection.end)
})

_face_landmarker = None

# Running counts so a full-video run reports how often detection
# actually succeeded - if this comes out as 0/N, the model isn't
# finding a face at all (angle/motion blur/confidence too strict),
# which looks identical to "the effect did nothing" from the output
# video alone, so it's worth being able to tell the difference.
_face_detect_attempts = 0
_face_detect_hits = 0

# CSRT tracker state - once the (expensive, pose-robust) tile-grid
# search finds a face, its bounding box is handed to a cheap
# correlation tracker so subsequent frames can skip re-running the
# full tile search every time. Reset to None whenever tracking is
# lost/stale, which triggers a fresh tile-grid search to re-acquire.
_face_tracker = None
_face_tracker_miss_streak = 0


def _create_csrt_tracker():
    """
    OpenCV's tracker API surface differs across versions/packages
    (cv2.TrackerCSRT_create() in older opencv-contrib-python, the
    class-factory cv2.TrackerCSRT.create() in OpenCV 5, or under
    cv2.legacy in some in-between versions) - try each known shape
    rather than assuming one, since guessing wrong burns a whole
    round-trip to find out. Raises RuntimeError with a clear message
    listing what was tried if none of them exist on this install.
    """
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
        "This requires opencv-contrib-python (not plain opencv-python) - "
        "CSRT is a contrib-only tracker regardless of OpenCV version. "
        "Tried:\n" + "\n".join(errors)
    )


def _get_face_landmarker():
    """
    Lazily builds the MediaPipe FaceLandmarker. Requires the model
    bundle at config.face_landmarker_model_path - see that setting's
    comment in config.py for the download link.
    """
    global _face_landmarker

    if _face_landmarker is None:
        if not os.path.exists(config.face_landmarker_model_path):
            raise FileNotFoundError(
                f"Face landmarker model not found at "
                f"'{config.face_landmarker_model_path}'. Download it from "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task and point "
                "config.face_landmarker_model_path at it, or set "
                "config.use_face_blur = False to disable this feature."
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.face_landmarker_model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            # Lowered from the 0.5 default - workout footage tends to
            # have motion blur, extreme up-angle camera positions, and
            # partially-out-of-frame faces, all of which make the
            # default threshold too strict and silently skip every
            # frame with no error or warning.
            min_face_detection_confidence=config.face_detection_confidence,
            min_face_presence_confidence=config.face_detection_confidence,
        )
        _face_landmarker = FaceLandmarker.create_from_options(options)

    return _face_landmarker


def detect_face_oval_points(image_bgr: np.ndarray, count_attempt: bool = True) -> np.ndarray | None:
    """
    Runs the face landmarker on whatever image is passed in (this may
    be a small cropped-and-upscaled tile, not the full frame - see
    get_head_crop) and returns the face-oval landmark points in that
    image's OWN local pixel coordinates.

    count_attempt controls whether this call counts toward the
    per-frame hit-rate summary log. When searching multiple tiles per
    frame, only the overall per-frame result should count once -
    intermediate tile tries pass count_attempt=False so the summary
    reflects "frames with a face found", not "tiles checked".

    Returns None if no face is found - callers should treat that as
    "nothing detected", not an error.
    """
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

    # Assume the first/only detected face is the subject.
    landmarks = result.face_landmarks[0]
    return np.array([
        (landmarks[i].x * width, landmarks[i].y * height)
        for i in _FACE_OVAL_INDICES
    ], dtype=np.float32)


def _make_person_tiles(
    person_bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """
    Splits the person's bounding box into an overlapping grid of tiles
    covering the WHOLE body - not just an assumed head region. This is
    what makes head-finding robust to any pose (hanging, lying flat,
    sideways, arms overhead, upside-down) without assuming where the
    head is: every part of the body gets an equal, independently-
    upscaled chance at detection, rather than betting on a fixed
    fraction/position heuristic that only holds for one pose family.

    Returns a list of (x0, y0, x1, y1) tile boxes, ordered top-to-
    bottom, left-to-right (an arbitrary but consistent search order -
    see get_head_crop for why this order is a reasonable default).
    """
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


def get_head_crop(
    image_bgr: np.ndarray, sam2_mask: np.ndarray
) -> tuple[np.ndarray, int, int, float, float] | None:
    """
    Searches the WHOLE person region (not an assumed head position)
    for a face, by splitting the SAM2 mask's bounding box into an
    overlapping grid of tiles, upscaling each one independently, and
    trying face detection on each in turn until one succeeds. This
    both solves the small/distant-face resolution problem (each tile
    is upscaled the same way a single head-crop would be) AND makes no
    assumption about where the head is - unlike a fixed "top X% of the
    bbox" heuristic, which breaks down for arms-overhead poses (where
    the topmost mask pixel is a hand, not the head) or lying-down
    exercises (where "top" has no relationship to head position at
    all).

    Returns (local_oval_points, tile_x0, tile_y0, scale_x, scale_y)
    for whichever tile a face was found in - local_oval_points are
    already the detected face-oval landmarks, in that tile's local
    pixel coordinates, so the caller doesn't need to re-run detection.
    Map them to full-frame coordinates via:
        full_x = tile_x0 + local_x / scale_x
        full_y = tile_y0 + local_y / scale_y

    Returns None if the mask is empty, or if no tile yields a
    detected face.
    """
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

        # Upscale so the tile's longer side reaches the target size,
        # giving the detector a "zoomed in" view of a face that would
        # otherwise be too small/low-resolution in the full frame.
        scale = config.head_crop_upscale_target / max(crop_h, crop_w)
        scale = max(scale, 1.0)  # never downscale - only helps small tiles
        new_size = (int(round(crop_w * scale)), int(round(crop_h * scale)))
        crop_upscaled = cv.resize(crop, new_size, interpolation=cv.INTER_CUBIC)

        # Each tile try doesn't count toward the per-frame hit-rate
        # log on its own - only the overall per-frame outcome does,
        # logged once by the caller (get_face_oval_mask).
        local_points = detect_face_oval_points(crop_upscaled, count_attempt=False)
        if local_points is not None:
            return local_points, tx0, ty0, scale, scale

    return None


def _get_tracked_crop(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, int, int, float, float] | None:
    """
    If a CSRT tracker is active, asks it where the face box is this
    frame, crops+upscales around that box, and tries landmark
    detection there. This is the cheap path - skips the full tile-grid
    search entirely when tracking is working.

    Returns None (and resets tracker state) if there's no active
    tracker, the tracker itself reports failure, or landmark detection
    on the tracked region misses too many frames in a row (tracker
    drift/lost target) - in all these cases the caller should fall
    back to the full tile-grid search to re-acquire.
    """
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
    """
    (Re-)initializes the CSRT tracker on the bounding box of a
    just-detected face, so subsequent frames can use the cheap
    tracked path instead of the full tile-grid search.
    """
    global _face_tracker, _face_tracker_miss_streak

    x0, y0 = oval_points.min(axis=0)
    x1, y1 = oval_points.max(axis=0)
    # A little padding beyond the tight oval so the tracker has some
    # surrounding context to lock onto, not just the face interior.
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
        # No usable tracker API on this OpenCV install - log once and
        # keep going without tracking (tile-grid search still runs
        # every frame; just slower, not broken).
        if config.use_face_tracking:
            logger.warning(
                "CSRT tracker unavailable (%s) - falling back to "
                "full tile-grid search every frame. Set "
                "config.use_face_tracking = False to silence this.",
                e,
            )


def get_face_oval_mask(
    image_bgr: np.ndarray, sam2_mask: np.ndarray | None = None
) -> np.ndarray | None:
    """
    Detects a face and returns a filled (H, W) uint8 mask (255 inside
    the face contour, 0 outside), built from MediaPipe's actual
    face-oval landmark points via convex hull - not an approximated
    ellipse or fixed-fraction box.

    Detection tries three paths, cheapest first:
      1. CSRT tracker (if one is active from a previous frame) - crop
         around wherever the tracker says the face is, no tile search.
      2. Full tile-grid search over the whole SAM2-masked person
         region (see get_head_crop) - pose-robust, more expensive.
      3. Full-frame detection with no cropping at all, if sam2_mask
         wasn't provided.
    Whenever (2) succeeds, the tracker is (re-)seeded from that result
    so later frames can use the cheap path (1) again.

    Returns None if no face is found this frame (person facing away,
    steep angle, etc.) - callers should treat that as "nothing to
    flatten this frame", not an error.
    """
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
        # Both paths already ran detection with count_attempt=False -
        # count this frame as a single hit here instead.
        _face_detect_attempts += 1
        _face_detect_hits += 1

        if config.use_face_tracking and not used_tracker:
            # Tile search (re-)acquired a face - seed/refresh the
            # tracker so subsequent frames can use the cheap path.
            _seed_tracker(image_bgr, oval_points)
    else:
        oval_points = detect_face_oval_points(image_bgr)
        if oval_points is None:
            return None

    hull = cv.convexHull(oval_points).astype(np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv.fillConvexPoly(mask, hull, 255)
    return mask


def log_face_detection_summary() -> None:
    """
    Call once after processing a whole video to report how often face
    detection actually succeeded. If hits is 0 (or very low relative
    to attempts), the flattening effect isn't visibly doing anything
    not because the flattening logic is wrong, but because the model
    isn't finding a face in this footage at the configured confidence
    - that's a detection problem, not a rendering problem, and no
    per-frame warning would otherwise surface it.
    """
    if _face_detect_attempts == 0:
        return
    logger.info(
        f"Face detection: {_face_detect_hits}/{_face_detect_attempts} frames "
        f"had a face detected (confidence >= {config.face_detection_confidence})."
    )
    if _face_detect_hits == 0:
        logger.warning(
            "Face was never detected in any frame - face flattening had no "
            "effect on this video. Try lowering config.face_detection_confidence "
            "(currently %.2f), increasing config.head_crop_upscale_target, or "
            "check that the face is reasonably visible/front-facing in this "
            "footage.",
            config.face_detection_confidence,
        )


def flatten_face_region(image_bgr: np.ndarray, sam2_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Replaces the face-oval region of the frame with a single flat
    color (the mean color sampled from that same region), feathering
    the mask edge so the transition isn't a hard boundary. Operates
    on the source frame itself, before any edge detection runs, so
    there is zero internal gradient there - guaranteed, since a flat
    fill has no edges by construction, unlike relying on a blur being
    "strong enough".

    sam2_mask, if provided, is used to crop+upscale just the head
    region before running face detection (see get_head_crop) - this
    is what makes a small/distant face detectable at all on typical
    full-body exercise footage, rather than relying on a "long range"
    detection model.

    If no face is detected this frame, returns image_bgr unchanged.
    """
    mask = get_face_oval_mask(image_bgr, sam2_mask)
    if mask is None:
        return image_bgr

    mean_color = cv.mean(image_bgr, mask=mask)[:3]  # BGR mean over the masked region
    flat_fill = np.full_like(image_bgr, mean_color, dtype=np.uint8)

    # Feather the hard convex-hull mask edge with a wide Gaussian blur
    # on the mask itself, so the transition to the flat fill fades
    # smoothly rather than cutting off sharply at the hull boundary.
    feather_ksize = config.face_blur_feather_ksize
    mask_feathered = cv.GaussianBlur(mask, feather_ksize, 0).astype(np.float32) / 255.0
    mask_feathered = mask_feathered[:, :, np.newaxis]

    blended = (
        flat_fill.astype(np.float32) * mask_feathered
        + image_bgr.astype(np.float32) * (1.0 - mask_feathered)
    )
    return blended.astype(np.uint8)


# ============================================================
# EDGE PIPELINE
# ============================================================

def edge_detect(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """
    mask, if provided, is a (H, W) bool array. Edge detection still runs
    on the full frame (so gradients at the mask boundary are accurate),
    but the result is zeroed outside the mask afterward.
    """
    if config.use_face_blur:
        image = flatten_face_region(image, sam2_mask=mask)

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
# FPS RESAMPLING
# ============================================================

def resample_video_to_fps(input_path: str, target_fps: float, output_dir: str) -> str:
    """
    Writes a frame-dropped copy of input_path at target_fps and
    returns its path. Frames are dropped using an accumulator (not a
    naive "every Nth frame") so drops are spread evenly across the
    clip rather than drifting - matters when original_fps/target_fps
    isn't a whole number (e.g. 25 -> 15 is a 1.667x ratio).

    This becomes the actual input to the rest of the pipeline (SAM2 +
    edge detection both run on the reduced frame count), which is
    where the speedup comes from - not just fewer frames written at
    the end.

    If target_fps >= the source's actual fps, returns input_path
    unchanged (nothing to drop, avoid a pointless re-encode pass).
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

    # Bresenham-style accumulator: advance by target_fps each source
    # frame, emit a frame and subtract source_fps whenever the
    # accumulator has crossed a full source_fps "tick" - spreads
    # drops evenly rather than concentrating rounding error.
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

    logger.info(
        f"Resampled {total_frames} frames at {source_fps:.2f}fps down to "
        f"{written} frames at {target_fps:g}fps -> {output_path}"
    )
    return output_path


# ============================================================
# PROCESS VIDEO
# ============================================================

def _process_frame_range(
    video_path: str,
    start_frame: int,
    end_frame: int,
    masks: list[np.ndarray | None],
    output_path: str,
    width: int,
    height: int,
    fps: float,
) -> str:
    """
    Worker function: processes frames [start_frame, end_frame) of
    video_path (edge_detect + recolor, using the corresponding
    pre-computed SAM2 masks) and writes them to their own chunk video
    file at output_path. Returns output_path so the caller can
    concatenate chunks back together in order.

    Runs as its own process (see process_video) - each worker gets
    fresh module-level state (face landmarker, CSRT tracker, etc.)
    since Python's multiprocessing re-imports this module per process;
    nothing needs to be explicitly reset for that reason. This does
    mean each chunk pays its own face re-acquisition cost at the start
    (the tracker doesn't carry over from the previous chunk) - a small,
    one-time cost per chunk, not per frame.

    Must be a module-level function (not a nested closure/lambda) so
    it can be pickled for multiprocessing on Windows, which uses the
    'spawn' start method.
    """
    video = cv.VideoCapture(video_path)
    video.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(output_path, fourcc, fps, (width, height), True)

    for i in tqdm(range(end_frame - start_frame), desc=f"Frames {start_frame}-{end_frame}"):
        success, frame = video.read()
        if not success:
            break

        mask = masks[i]
        edges = edge_detect(frame, mask=mask)
        output_frame = recolor(edges, mask=mask)
        writer.write(output_frame)

    writer.release()
    video.release()
    return output_path


def _concatenate_chunks(chunk_paths: list[str], output_path: str, fps: float, width: int, height: int) -> None:
    """
    Re-encodes a sequence of chunk video files, in order, into a
    single final output file. Done via OpenCV (read-then-rewrite)
    rather than shelling out to ffmpeg, to avoid adding a new external
    binary dependency to the project.
    """
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


def process_video(input_path: str, output_path: str) -> None:
    make_sure_file_exists(input_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Optional FPS reduction. This becomes the actual input to
    # everything below - SAM2 and edge detection both process fewer
    # frames as a result, not just the final written video.
    working_input_path = input_path
    if config.target_fps is not None:
        working_input_path = resample_video_to_fps(
            input_path, config.target_fps, config.resampled_video_dir
        )

    video = cv.VideoCapture(working_input_path)
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {working_input_path}")

    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = float(video.get(cv.CAP_PROP_FPS))
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    video.release()

    logger.info(
        f"width: {width}; height: {height}; fps: {fps}; total_frames: {total_frames}."
    )

    # --- SAM2 segmentation runs first, as its own pass over the whole
    # video - this is an inherently sequential pass (each frame's mask
    # depends on the model's running memory of previous frames), so it
    # cannot itself be chunked/parallelized the way the edge-detection
    # step below can. Masks are fully materialized into a list (not
    # left as a generator) so they can be sliced per-chunk.
    masks: list[np.ndarray | None] = [None] * total_frames
    if config.use_sam2:
        # Imported here (not at module top) so worker processes never
        # load torch/CUDA - see the note at the top of this file.
        from sam2_segment import Sam2Segmenter, build_config_from_settings

        logger.info("Running SAM2 segmentation pass...")
        segmenter = Sam2Segmenter(build_config_from_settings())
        segmenter.prepare_frames(working_input_path)
        if segmenter._num_frames != total_frames:
            raise RuntimeError(
                f"SAM2 extracted {segmenter._num_frames} frames but the video has "
                f"{total_frames} frames according to VideoCapture - these should always "
                f"match. Try deleting {config.sam2_frame_extraction_dir} and rerunning."
            )
        masks = list(tqdm(segmenter.propagate_masks(), total=total_frames, desc="Propagating masks..."))

    if not config.use_multiprocessing or config.num_worker_processes == 1:
        # Single-process path - same as chunking with exactly one
        # chunk covering the whole video, but skips the pool/temp-file
        # machinery entirely. Useful for debugging face-detection
        # changes, where per-process log output interleaving from a
        # pool makes things harder to read.
        chunk_path = _process_frame_range(
            working_input_path, 0, total_frames, masks, output_path, width, height, fps
        )
    else:
        num_workers = config.num_worker_processes or min(os.cpu_count() or 1, 6)
        num_workers = min(num_workers, max(1, total_frames))

        chunk_boundaries = np.linspace(0, total_frames, num_workers + 1, dtype=int)
        os.makedirs(config.chunk_output_dir, exist_ok=True)

        chunk_args = []
        for i in range(num_workers):
            start, end = int(chunk_boundaries[i]), int(chunk_boundaries[i + 1])
            if end <= start:
                continue
            chunk_output_path = os.path.join(config.chunk_output_dir, f"chunk_{i:03d}.mp4")
            chunk_args.append((
                working_input_path, start, end, masks[start:end],
                chunk_output_path, width, height, fps,
            ))

        logger.info(f"Processing {total_frames} frames across {len(chunk_args)} worker processes...")

        with ProcessPoolExecutor(max_workers=len(chunk_args)) as executor:
            futures = [executor.submit(_process_frame_range, *args) for args in chunk_args]
            chunk_paths = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks..."):
                chunk_paths.append(future.result())

        # as_completed() doesn't preserve submission order - futures
        # already carry their own paths, so sort chunk_paths back into
        # frame order via the chunk index encoded in the filename
        # rather than trusting completion order.
        chunk_paths.sort()

        logger.info("Concatenating chunks into final output...")
        _concatenate_chunks(chunk_paths, output_path, fps, width, height)

        for chunk_path in chunk_paths:
            os.remove(chunk_path)

    if config.use_face_blur:
        log_face_detection_summary()

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
