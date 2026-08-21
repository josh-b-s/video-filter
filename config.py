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
use_sam2 = True

sam2_checkpoint_path = "EdgeTAM/checkpoints/edgetam.pt"
sam2_model_config_path = "configs/edgetam.yaml"

sam2_vos_optimized = False           # torch.compile fast path - Linux only, needs Triton

# Frame loading offload: init_state() loads every decoded video frame
# into VRAM at once before segmentation starts, regardless of model
# size (EdgeTAM vs tiny SAM2 makes no difference here - this is a raw
# frame-count x resolution memory cost). On a 4GB card, short clips
# (~100 frames) fit fine with both False; longer clips (1000+ frames)
# will OOM unless offload_video_to_cpu is True. If you still OOM with
# both True, the clip needs to be shorter or downscaled - offload
# alone won't fix an unbounded frame count.
sam2_offload_video_to_cpu = True
sam2_offload_state_to_cpu = True

# Grow the mask slightly so edges right at the subject's silhouette
# aren't clipped off. Only affects internal edge clipping, not the
# outer silhouette fill (which stays tight to the true mask). Larger
# values catch more boundary detail but also let more background
# clutter just outside the person leak through as edge "specks" -
# if you see stray noise just outside the outline, shrink this.
sam2_mask_dilate_ksize = (0, 0)

# Removes small, disconnected edge fragments (speckle noise) after
# Canny/Sobel/morphology - any connected blob of edge pixels smaller
# than this area (in pixels) is dropped entirely. Real limb/silhouette
# contours are large and connected, so this only strips noise. Set to
# 0 to disable.
edge_min_component_area = 40

# Draws a stroke along the outer silhouette boundary (the mask's own
# contour). Set thickness to 0 to disable. Higher = chunkier outline;
# grows both inward and outward from the true mask boundary.
sam2_outline_thickness = 8

# "filled": subject silhouette filled with line_color, internal edges
#           (limb crossover etc.) drawn in bg_color on top, outline
#           drawn in bg_color too (barely visible - line_color already
#           contrasts against bg_color at the boundary).
# "outline": inverted - background stays bg_color everywhere, subject
#            interior is NOT filled, only the outline stroke and
#            internal edges are drawn, both in line_color. Looks like
#            pure line art constrained to the subject's shape.
sam2_style = "outline"

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