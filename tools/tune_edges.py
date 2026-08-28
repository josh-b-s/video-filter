"""
Interactive live tuner for edge detection parameters (advanced/optional
tool - the GUI in app.py covers the normal picking workflow; this is
for dialing in canny/sobel/morphology numbers by eye).

Shows a sample frame with the real edge_detect()/recolor() pipeline
applied, next to a sidebar of sliders. Press 's' to write values back
into config.py.

Run from the project root:
    python -m tools.tune_edges <video_filename_in_input_folder>

Points/face-toggle come from the job store (whatever was picked for
that video in the GUI) rather than a separate single-video file.

Controls:
    Sliders     - adjust values live
    'm'         - toggle SAM2 mask on/off in the preview
    'o'         - toggle sam2_style between "filled" and "outline"
    'c'         - toggle use_clahe
    'b'         - toggle use_sobel
    's'         - save current values back into config.py
    'q' / ESC   - quit without saving
"""

import logging
import os
import re
import sys

import cv2 as cv
import numpy as np

import config
from core import edges, job_store, video_io

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

preview_window = "Preview - m=mask, o=style, c=clahe, b=sobel, s=save, q=quit"
controls_window = "Controls"

SLIDERS = [
    ("canny_low_threshold", 0, 500, 1),
    ("canny_high_threshold", 0, 500, 1),
    ("sobel_threshold", 0, 255, 1),
    ("clahe_clip_limit", 0, 100, 0.1),
    ("gaussian_ksize", 1, 31, 1),
    ("open_ksize", 1, 15, 1),
    ("close_ksize", 1, 15, 1),
    ("dilation_ksize", 1, 15, 1),
    ("sam2_mask_dilate_ksize", 0, 31, 1),
    ("edge_min_component_area", 0, 1000, 1),
    ("sam2_outline_thickness", 0, 30, 1),
]

TOGGLE_KEYS = {ord("c"): "use_clahe", ord("b"): "use_sobel"}
SIDEBAR_WIDTH = 340
SIDEBAR_ROW_HEIGHT = 50
SIDEBAR_MARGIN = 20


def get_config_value_for_slider(attr: str, scale: float):
    val = getattr(config, attr)
    if isinstance(val, tuple):
        val = val[0]
    return int(round(val / scale))


def apply_slider_value(attr: str, trackbar_val: int, scale: float):
    real_val = trackbar_val * scale
    current = getattr(config, attr)
    if attr == "gaussian_ksize":
        n = max(1, int(round(real_val)))
        if n % 2 == 0:
            n += 1
        setattr(config, attr, (n, n))
    elif isinstance(current, tuple):
        n = max(1, int(round(real_val)))
        setattr(config, attr, (n, n))
    elif isinstance(current, int) and scale == int(scale):
        setattr(config, attr, int(real_val))
    else:
        setattr(config, attr, real_val)


def fit_to_screen(image, max_width, max_height):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale >= 1.0:
        return image, 1.0
    new_size = (int(w * scale), int(h * scale))
    return cv.resize(image, new_size, interpolation=cv.INTER_AREA), scale


def update_config_file(updates: dict) -> None:
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.py")
    with open(config_path) as f:
        text = f.read()

    for key, value in updates.items():
        new_val_str = repr(round(value, 3)) if isinstance(value, float) else repr(value)
        pattern = rf"^{re.escape(key)}\s*=.*$"
        replacement = f"{key} = {new_val_str}"
        new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if count == 0:
            logger.warning(f"Could not find '{key}' in config.py to update - skipped.")
            continue
        text = new_text

    with open(config_path, "w") as f:
        f.write(text)
    logger.info(f"Saved {len(updates)} value(s) to config.py")


def build_controls_window(screen_h: int) -> None:
    sidebar_height = min(screen_h - 80, SIDEBAR_MARGIN * 2 + SIDEBAR_ROW_HEIGHT * (len(SLIDERS) + 2))
    cv.namedWindow(controls_window, cv.WINDOW_NORMAL)
    cv.resizeWindow(controls_window, SIDEBAR_WIDTH, sidebar_height)
    cv.moveWindow(controls_window, 0, 0)

    for attr, lo, hi, scale in SLIDERS:
        initial = get_config_value_for_slider(attr, scale)
        cv.createTrackbar(attr, controls_window, initial, hi, lambda v: None)
        cv.setTrackbarMin(attr, controls_window, lo)

    cv.createTrackbar("style(0=fill,1=outline)", controls_window,
                       1 if config.sam2_style == "outline" else 0, 1, lambda v: None)
    cv.createTrackbar("mask_on", controls_window, 1, 1, lambda v: None)


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python -m tools.tune_edges <video_filename_in_{config.input_folder}>")
        sys.exit(1)

    filename = sys.argv[1]
    video_path = os.path.join(config.input_folder, filename)
    job = job_store.get_job(filename)
    if not job.has_points:
        logger.warning(f"No prompt points saved for {filename} yet - pick them in the GUI first. "
                        f"Continuing without a mask.")

    frame = video_io.read_first_frame(video_path)

    mask = None
    if job.has_points and config.use_sam2:
        from core.sam2_segment import predict_mask_for_points
        mask = predict_mask_for_points(frame, job.prompt_points, job.prompt_labels)
    mask_enabled = mask is not None

    screen_w, screen_h = 1600, 900  # simple default; adjust if needed
    build_controls_window(screen_h)

    preview_max_w = screen_w - SIDEBAR_WIDTH - 40
    preview_max_h = screen_h - 80
    cv.namedWindow(preview_window, cv.WINDOW_NORMAL)
    cv.moveWindow(preview_window, SIDEBAR_WIDTH + 20, 0)

    while True:
        for attr, lo, hi, scale in SLIDERS:
            trackbar_val = cv.getTrackbarPos(attr, controls_window)
            apply_slider_value(attr, trackbar_val, scale)

        config.sam2_style = "outline" if cv.getTrackbarPos("style(0=fill,1=outline)", controls_window) == 1 else "filled"
        mask_enabled = cv.getTrackbarPos("mask_on", controls_window) == 1

        active_mask = mask if (mask_enabled and mask is not None) else None
        edge_map = edges.edge_detect(frame, mask=active_mask, face_removal_enabled=job.use_face_blur)
        preview = edges.recolor(edge_map, mask=active_mask)

        scaled_preview, _ = fit_to_screen(preview, preview_max_w, preview_max_h)
        cv.imshow(preview_window, scaled_preview)
        key = cv.waitKey(50) & 0xFF

        if key == ord("q") or key == 27:
            logger.info("Quit without saving.")
            break
        elif key == ord("s"):
            updates = {attr: getattr(config, attr) for attr, *_ in SLIDERS}
            updates["sam2_style"] = config.sam2_style
            updates["use_clahe"] = config.use_clahe
            updates["use_sobel"] = config.use_sobel
            update_config_file(updates)
        elif key in TOGGLE_KEYS:
            attr = TOGGLE_KEYS[key]
            setattr(config, attr, not getattr(config, attr))
            logger.info(f"{attr}: {getattr(config, attr)}")

        if (cv.getWindowProperty(preview_window, cv.WND_PROP_VISIBLE) < 1
                or cv.getWindowProperty(controls_window, cv.WND_PROP_VISIBLE) < 1):
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
