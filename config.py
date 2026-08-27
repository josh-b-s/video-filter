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

input_video_path = "materials/videos/test.mp4"
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

# Replaces the actual face-contour region (via MediaPipe's face mesh,
# not a rough ellipse) with a single flat color in the source frame,
# before edge detection runs - guaranteeing zero internal gradient
# there, rather than relying on a blur being "strong enough". This
# never touches the SAM2 mask, so the head's outer silhouette
# (hairline, jaw, chin, ears) stays exactly as filled as the rest of
# the body - only the frame content inside the face is altered. If no
# face is detected in a given frame (person facing away, steep angle,
# motion blur), that frame is left alone - check the
# "Face detection: X/Y frames" log line after a run to see how often
# that happened; if X is 0 or very low, try lowering
# face_detection_confidence below.
#
# Requires the model file below. Download:
#   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
# Save it somewhere in your project and point the path below at it.
use_face_blur = True
face_landmarker_model_path = "models/face_landmarker.task"

# Minimum confidence (0-1) for MediaPipe to count a face as detected.
# Default MediaPipe behavior is 0.5, which can silently skip most/all
# frames on footage with motion blur, extreme camera angles, or a
# partially out-of-frame face - all common in workout clips. Lower
# this if the "Face detection: X/Y frames" summary log shows a low
# hit rate; raise it if it's flattening things that aren't actually
# faces.
face_detection_confidence = 0.2

# How wide the blend between the flat-fill and normal-body is. Larger
# = softer/more gradual transition at the face's edge; smaller = a
# more defined (but still soft, never hard) boundary. Must be odd.
face_blur_feather_ksize = (51, 51)

# Face detection searches the WHOLE person region (via the SAM2 mask)
# rather than assuming the head is in any particular position - this
# is what makes it robust to any pose (hanging, lying flat, sideways,
# arms overhead), not just upright-with-head-on-top. It works by
# splitting the person's bounding box into an overlapping grid of
# tiles and trying detection on each (upscaled) tile in turn.
#
# face_search_tile_grid is (rows, cols) for that grid - more tiles
# search more thoroughly (better for unusual poses) at the cost of
# more detector calls per frame when early tiles miss. face_search_
# tile_overlap is how much adjacent tiles overlap (as a fraction of
# tile size) so a face isn't missed by falling right on a tile seam.
# head_crop_upscale_target is the pixel size each tile's longer side
# is scaled up to before detection - higher gives the detector a
# "more zoomed in" view for very small/distant faces, at the cost of
# a bit more per-frame compute.
face_search_tile_grid = (3, 2)  # (rows, cols)
face_search_tile_overlap = 0.2
head_crop_upscale_target = 512

# Once the tile-grid search finds a face, a CSRT correlation tracker
# (OpenCV's built-in cv2.TrackerCSRT) takes over for subsequent frames
# instead of re-running the full tile search every time - much
# cheaper, since it's just following the already-found face rather
# than searching the whole body again. Requires opencv-contrib-python
# (not plain opencv-python) - CSRT is contrib-only regardless of
# OpenCV version. If no usable tracker API is found on your OpenCV
# install, this silently falls back to running the full tile-grid
# search every frame instead (slower, but still correct) - a warning
# is logged once if that happens.
use_face_tracking = True

# How many consecutive frames the tracker's reported position is
# allowed to fail to yield a detected face before giving up on it and
# re-acquiring via a fresh full tile-grid search. Higher tolerates
# more transient misses (motion blur, brief occlusion) before paying
# for a re-search; lower re-acquires sooner if the tracker has drifted
# off the real face.
face_tracker_max_miss_streak = 5


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