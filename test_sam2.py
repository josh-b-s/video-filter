"""
Quick visual test for SAM2 segmentation output.

Runs the segmenter on your video, overlays the mask (in red) on a
handful of sampled frames, and saves them as PNGs so you can eyeball
whether the prompt point is tracking the right subject through the clip.

All settings come from config.py - edit that file, not this one.

Usage:
    python test_sam2.py
"""

import logging
import os

import cv2 as cv

import config
from sam2_segment import Sam2Segmenter, build_config_from_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

overlay_color_bgr = (0, 0, 255)  # red
overlay_alpha = 0.5


def main() -> None:
    os.makedirs(config.sam2_test_preview_dir, exist_ok=True)

    sam2_config = build_config_from_settings()
    segmenter = Sam2Segmenter(sam2_config)
    frame_dir = segmenter.prepare_frames(config.input_video_path)

    frame_files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    total_frames = len(frame_files)
    logger.info(f"Total frames: {total_frames}")

    num_preview = config.sam2_test_num_preview_frames
    if num_preview >= total_frames:
        preview_indices = set(range(total_frames))
    else:
        preview_indices = set(
            int(i * (total_frames - 1) / (num_preview - 1))
            for i in range(num_preview)
        )

    logger.info(f"Saving overlays for frame indices: {sorted(preview_indices)}")

    saved_count = 0
    empty_mask_count = 0

    for idx, mask in enumerate(segmenter.propagate_masks()):
        if idx not in preview_indices:
            continue

        frame_path = os.path.join(frame_dir, frame_files[idx])
        frame = cv.imread(frame_path)

        mask_pixel_count = int(mask.sum())
        if mask_pixel_count == 0:
            empty_mask_count += 1
            logger.warning(f"Frame {idx}: mask is EMPTY (no foreground detected).")
        else:
            logger.info(f"Frame {idx}: mask covers {mask_pixel_count} pixels.")

        overlay = frame.copy()
        overlay[mask] = overlay_color_bgr
        blended = cv.addWeighted(overlay, overlay_alpha, frame, 1 - overlay_alpha, 0)

        out_path = os.path.join(config.sam2_test_preview_dir, f"frame_{idx:07d}.png")
        cv.imwrite(out_path, blended)
        saved_count += 1

    logger.info(f"Saved {saved_count} preview frames to {config.sam2_test_preview_dir}")
    if empty_mask_count > 0:
        logger.warning(
            f"{empty_mask_count} of {saved_count} previewed frames had empty masks - "
            f"check your prompt points or whether the subject leaves frame."
        )


if __name__ == "__main__":
    main()
