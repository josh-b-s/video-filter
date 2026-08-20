"""
Single source of truth for all settings used across the pipeline:
    canny.py             - edge detection + recolor
    sam2_segment.py       - SAM2 video segmentation (imports Sam2Config from here indirectly)
    test_sam2.py           - SAM2 mask preview/test
    pick_prompt_points.py  - interactive prompt point picker

Edit this file only - the other scripts import from it and shouldn't
need their own copies of these settings.
"""

import json
import os

# ============================================================
# INPUT / OUTPUT
# ============================================================

input_video_path = "materials/videos/pullup.mp4"
output_folder = "output"

sam2_frame_extraction_dir = "output/_sam2_frames"
sam2_test_preview_dir = "output/_sam2_test"
prompt_points_path = "prompt_points.json"


# ============================================================
# COLOR SCHEME
# ============================================================

# Preset 3 = balanced
bg_color_rgb = [20, 23, 31]       # #14171F
line_color_rgb = [155, 111, 224]  # mobility purple


# ============================================================
# EDGE DETECTION
# ============================================================

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
# SAM2 SEGMENTATION
# ============================================================

# When enabled, edge detection is restricted to the SAM2-segmented
# foreground subject, and the subject is filled as a flat silhouette
# so limbs stay readable even when crossing in front of the body.
use_sam2 = False

sam2_checkpoint_path = "checkpoints/sam2.1_hiera_tiny.pt"
sam2_model_config_path = "configs/sam2.1/sam2.1_hiera_t.yaml"

sam2_vos_optimized = False           # torch.compile fast path - Linux only, needs Triton
sam2_offload_video_to_cpu = False    # True = safer on low VRAM, slower (CPU<->GPU shuttling)
sam2_offload_state_to_cpu = False    # True = safer on low VRAM, slower

# Grow the mask slightly so edges right at the subject's silhouette
# aren't clipped off. Only affects internal edge clipping, not the
# outer silhouette fill (which stays tight to the true mask).
sam2_mask_dilate_ksize = (15, 15)

# How many sampled frames test_sam2.py saves as overlay previews.
sam2_test_num_preview_frames = 8


def load_prompt_points(default_points=None, default_labels=None):
    """
    Loads (prompt_points, prompt_labels) from prompt_points_path if it
    exists (saved by pick_prompt_points.py), otherwise falls back to
    the given defaults. Centralized here so canny.py, test_sam2.py,
    and anything else that needs a SAM2 prompt all pick up the same
    points automatically.
    """
    if os.path.exists(prompt_points_path):
        with open(prompt_points_path) as f:
            saved = json.load(f)
        points = [tuple(p) for p in saved["prompt_points"]]
        labels = saved["prompt_labels"]
        return points, labels

    if default_points is None:
        default_points = [(640, 360)]
    if default_labels is None:
        default_labels = [1] * len(default_points)
    return default_points, default_labels
