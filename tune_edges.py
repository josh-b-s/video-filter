"""
Interactive live tuner for edge detection parameters.

Shows a real frame from your video (scaled to fit your screen) with
the actual edge_detect() / recolor() pipeline from canny.py applied,
in one window, next to a narrow sidebar window holding all the
sliders. When you're happy, press 's' to write the new values back
into config.py.

This reuses canny.py's real functions directly (not a reimplementation)
so what you see here is exactly what canny.py will produce.

Controls:
    Sliders     - adjust values live (in the "Controls" sidebar window)
    'f'         - jump to a different sample frame
    'm'         - toggle SAM2 mask on/off in the preview
    'o'         - toggle sam2_style between "filled" and "outline"
    'c'         - toggle use_clahe
    'b'         - toggle use_sobel (Canny-only vs Canny+Sobel)
    's'         - save current slider/toggle values back into config.py
    'q' / ESC   - quit without saving

Usage:
    python tune_edges.py
"""

import ctypes
import logging
import re

import cv2 as cv
import numpy as np

import config
import canny

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

preview_window = "Preview - f=frame, m=mask, o=style, c=clahe, b=sobel, s=save, q=quit"
controls_window = "Controls"

# Slider definitions: (config_attr, min, max, scale)
# scale converts the integer trackbar value to the real config value,
# e.g. scale=0.1 means trackbar 0-100 maps to config value 0.0-10.0.
SLIDERS = [
    ("canny_low_threshold", 0, 500, 1),
    ("canny_high_threshold", 0, 500, 1),
    ("sobel_threshold", 0, 255, 1),
    ("clahe_clip_limit", 0, 100, 0.1),
    ("gaussian_ksize", 1, 31, 1),   # must end up odd - handled below
    ("open_ksize", 1, 15, 1),
    ("close_ksize", 1, 15, 1),
    ("dilation_ksize", 1, 15, 1),
    ("sam2_mask_dilate_ksize", 0, 31, 1),
    ("edge_min_component_area", 0, 1000, 1),
    ("sam2_outline_thickness", 0, 30, 1),
]

TOGGLE_KEYS = {
    ord("c"): "use_clahe",
    ord("b"): "use_sobel",
}

FRAME_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 0.95]

# Sidebar sizing
SIDEBAR_WIDTH = 340
SIDEBAR_ROW_HEIGHT = 50
SIDEBAR_MARGIN = 20


def get_screen_size() -> tuple[int, int]:
    """Returns (width, height) of the primary screen. Falls back to a
    conservative default if detection fails (e.g. non-Windows)."""
    try:
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1280, 800


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
            n += 1  # GaussianBlur requires odd kernel sizes
        setattr(config, attr, (n, n))
    elif isinstance(current, tuple):
        n = max(1, int(round(real_val)))
        setattr(config, attr, (n, n))
    elif isinstance(current, int) and scale == int(scale):
        setattr(config, attr, int(real_val))
    else:
        setattr(config, attr, real_val)


def load_sample_frames(video_path: str, fractions: list[float]) -> list[np.ndarray]:
    video = cv.VideoCapture(video_path)
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    frames = []
    for frac in fractions:
        idx = min(total - 1, int(frac * total))
        video.set(cv.CAP_PROP_POS_FRAMES, idx)
        success, frame = video.read()
        if success:
            frames.append(frame)
    video.release()
    if not frames:
        raise RuntimeError("Could not read any sample frames from video.")
    return frames


def get_mask_for_frame(frame: np.ndarray) -> np.ndarray | None:
    if not config.use_sam2:
        return None

    points, labels = config.load_prompt_points()
    if not points:
        return None

    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(config.sam2_model_config_path, config.sam2_checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    point_coords = np.array(points, dtype=np.float32)
    point_labels = np.array(labels, dtype=np.int32)
    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )
    return masks[0].astype(bool)


def fit_to_screen(image: np.ndarray, max_width: int, max_height: int) -> tuple[np.ndarray, float]:
    """Scales image down (never up) to fit within max_width x max_height,
    preserving aspect ratio. Returns (scaled_image, scale_factor)."""
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale >= 1.0:
        return image, 1.0
    new_size = (int(w * scale), int(h * scale))
    return cv.resize(image, new_size, interpolation=cv.INTER_AREA), scale


def update_config_file(updates: dict) -> None:
    with open("config.py") as f:
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

    with open("config.py", "w") as f:
        f.write(text)

    logger.info(f"Saved {len(updates)} value(s) to config.py")


def build_controls_window(screen_w: int, screen_h: int) -> None:
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
    cv.createTrackbar("mask_on", controls_window, 1 if config.use_sam2 else 0, 1, lambda v: None)


def main() -> None:
    screen_w, screen_h = get_screen_size()
    logger.info(f"Detected screen size: {screen_w}x{screen_h}")

    frames = load_sample_frames(config.input_video_path, FRAME_FRACTIONS)
    frame_idx = 0
    frame = frames[frame_idx]

    logger.info("Computing preview mask for current frame (if SAM2 enabled)...")
    mask = get_mask_for_frame(frame)
    mask_enabled = mask is not None

    build_controls_window(screen_w, screen_h)

    preview_max_w = screen_w - SIDEBAR_WIDTH - 40
    preview_max_h = screen_h - 80
    cv.namedWindow(preview_window, cv.WINDOW_NORMAL)
    cv.moveWindow(preview_window, SIDEBAR_WIDTH + 20, 0)

    logger.info("Adjust sliders in the Controls window. 's' to save, 'q' to quit, "
                "'f' to change frame, 'm' to toggle mask, 'o' to toggle style, "
                "'c' to toggle CLAHE, 'b' to toggle Sobel.")

    while True:
        for attr, lo, hi, scale in SLIDERS:
            trackbar_val = cv.getTrackbarPos(attr, controls_window)
            apply_slider_value(attr, trackbar_val, scale)

        config.sam2_style = "outline" if cv.getTrackbarPos(
            "style(0=fill,1=outline)", controls_window) == 1 else "filled"
        mask_enabled = cv.getTrackbarPos("mask_on", controls_window) == 1

        active_mask = mask if (mask_enabled and mask is not None) else None
        edges = canny.edge_detect(frame, mask=active_mask)
        preview = canny.recolor(edges, mask=active_mask)

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
        elif key == ord("f"):
            frame_idx = (frame_idx + 1) % len(frames)
            frame = frames[frame_idx]
            if mask_enabled:
                mask = get_mask_for_frame(frame)
            logger.info(f"Switched to sample frame {frame_idx + 1}/{len(frames)}")
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
