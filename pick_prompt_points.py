"""
Interactive point picker for SAM2 prompts.

Shows frame 0 of your video in a window, with a LIVE SAM2 mask preview
(in red) that updates after every click - so you see exactly what
SAM2 would select before running it on the full video.

Controls:
    Left click  = foreground point (label 1)  — e.g. the person, the bar
    Right click = background point (label 0)  — e.g. something to exclude
    'u'         = undo last point
    'r'         = reset all points
    's'         = save points to prompt_points.json and print the
                  Sam2Config snippet to paste into test_sam2.py / canny.py
    'q' / ESC   = quit without saving

IMPORTANT - to get the WHOLE PERSON in the mask, not just one piece
of clothing: SAM2 segments the specific object under each click, not
"person" as a concept. A single click on a shirt only grabs the
shirt, since it's a visually distinct region from skin, hair, pants,
and shoes. Click once on EACH visually distinct region you want
included - e.g. face/skin, hair, top, leggings/shorts, shoes - and
they'll merge into one combined foreground mask (watch the red
preview grow as you add points). Same idea applies to equipment:
click once on the bar/chair/desk too if you want it kept.

Run this BEFORE test_sam2.py to figure out your prompt coordinates,
instead of guessing from a screenshot.

Uses the same checkpoint as config.py's Sam2 settings - edit
config.py to change model size, not this file.

Usage:
    python pick_prompt_points.py
"""

import ctypes
import json
import logging
import os

import cv2 as cv
import numpy as np

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

window_name = "Click EACH region: skin/hair/top/pants/shoes (L=fg, R=bg, u=undo, r=reset, s=save, q=quit)"

fg_color_bgr = (0, 255, 0)     # green - point markers
bg_color_bgr = (0, 0, 255)     # red - point markers
mask_color_bgr = (0, 0, 255)   # red - live mask preview
mask_alpha = 0.5
point_radius = 6


# ============================================================
# STATE
# ============================================================

points: list[tuple[int, int]] = []  # always stored in ORIGINAL frame coordinates
labels: list[int] = []
predictor = None  # SAM2ImagePredictor, built lazily
display_scale = 1.0  # set in main() based on screen size vs frame size


def get_screen_size() -> tuple[int, int]:
    """Returns (width, height) of the primary screen. Falls back to a
    conservative default if detection fails (e.g. non-Windows)."""
    try:
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1280, 800


def fit_to_screen(image: np.ndarray, max_width: int, max_height: int) -> tuple[np.ndarray, float]:
    """Scales image down (never up) to fit within max_width x max_height,
    preserving aspect ratio. Returns (scaled_image, scale_factor)."""
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale >= 1.0:
        return image, 1.0
    new_size = (int(w * scale), int(h * scale))
    return cv.resize(image, new_size, interpolation=cv.INTER_AREA), scale


def build_predictor():
    global predictor
    if predictor is not None:
        return predictor

    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading SAM2 image predictor on {device}...")
    sam2_model = build_sam2(config.sam2_model_config_path, config.sam2_checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor


def compute_mask(base_frame) -> np.ndarray | None:
    """Runs SAM2 on the current points and returns a bool mask, or None if no points yet."""
    if not points:
        return None

    pred = build_predictor()
    # set_image re-encodes the frame; only needed once per frame, but
    # cheap enough to call every time here since this is single-image
    # interactive use, not a video loop.
    pred.set_image(cv.cvtColor(base_frame, cv.COLOR_BGR2RGB))

    point_coords = np.array(points, dtype=np.float32)
    point_labels = np.array(labels, dtype=np.int32)

    masks, scores, _ = pred.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )
    return masks[0].astype(bool)


def redraw(base_frame):
    display = base_frame.copy()

    mask = compute_mask(base_frame)
    if mask is not None:
        overlay = display.copy()
        overlay[mask] = mask_color_bgr
        display = cv.addWeighted(overlay, mask_alpha, display, 1 - mask_alpha, 0)

    for (x, y), label in zip(points, labels):
        color = fg_color_bgr if label == 1 else bg_color_bgr
        cv.circle(display, (x, y), point_radius, color, -1)
        cv.circle(display, (x, y), point_radius, (255, 255, 255), 2)

    if display_scale < 1.0:
        new_size = (int(display.shape[1] * display_scale), int(display.shape[0] * display_scale))
        display = cv.resize(display, new_size, interpolation=cv.INTER_AREA)

    cv.imshow(window_name, display)


def on_mouse(event, x, y, flags, param):
    base_frame = param
    # Clicks arrive in DISPLAYED (possibly scaled-down) coordinates -
    # convert back to original frame coordinates before storing, since
    # SAM2 and the saved prompt_points.json must use true pixel coords.
    orig_x = int(round(x / display_scale))
    orig_y = int(round(y / display_scale))

    if event == cv.EVENT_LBUTTONDOWN:
        points.append((orig_x, orig_y))
        labels.append(1)
        logger.info(f"Added FOREGROUND point at ({orig_x}, {orig_y})")
        redraw(base_frame)
    elif event == cv.EVENT_RBUTTONDOWN:
        points.append((orig_x, orig_y))
        labels.append(0)
        logger.info(f"Added BACKGROUND point at ({orig_x}, {orig_y})")
        redraw(base_frame)


def save_points():
    if not points:
        logger.warning("No points to save.")
        return

    data = {"prompt_points": points, "prompt_labels": labels}
    with open(config.prompt_points_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(points)} points to {config.prompt_points_path}")

    # Print a ready-to-paste config snippet.
    print("\n--- Paste into your Sam2Config ---")
    print(f"    prompt_points={points},")
    print(f"    prompt_labels={labels},")
    print("-----------------------------------\n")


def main() -> None:
    global display_scale

    if not os.path.exists(config.input_video_path):
        raise FileNotFoundError(f"Video not found: {config.input_video_path}")

    video = cv.VideoCapture(config.input_video_path)
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {config.input_video_path}")

    success, frame = video.read()
    video.release()
    if not success:
        raise RuntimeError("Could not read frame 0 from video.")

    logger.info(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")

    screen_w, screen_h = get_screen_size()
    _, display_scale = fit_to_screen(frame, screen_w - 100, screen_h - 150)
    logger.info(f"Screen: {screen_w}x{screen_h} - display scale: {display_scale:.3f}")

    # Load the model up front so the first click doesn't stall the UI.
    build_predictor()

    logger.info(
        "Left click = foreground point, right click = background point. "
        "Press 'u' to undo, 'r' to reset, 's' to save, 'q' to quit."
    )

    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, on_mouse, param=frame)
    redraw(frame)

    while True:
        key = cv.waitKey(20) & 0xFF

        if key == ord("q") or key == 27:  # 27 = ESC
            logger.info("Quit without saving.")
            break
        elif key == ord("u"):
            if points:
                removed = points.pop()
                labels.pop()
                logger.info(f"Undid point at {removed}")
                redraw(frame)
        elif key == ord("r"):
            points.clear()
            labels.clear()
            logger.info("Reset all points.")
            redraw(frame)
        elif key == ord("s"):
            save_points()

        # Window closed via the [x] button.
        if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
