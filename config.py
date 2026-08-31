"""
Single source of truth for pipeline settings.

Per-video state (which prompt points were picked, whether face removal
is on for that specific clip) does NOT live here anymore - it lives in
core/job_store.py, keyed per video, so the GUI can queue/process many
videos without them all sharing one global prompt/toggle.

Edit this file for the settings that should apply to every video.
"""

# ============================================================
# INPUT / OUTPUT
# ============================================================

input_folder = "materials/videos"      # folder the GUI scans for videos to pick/queue
output_folder = "output"

sam2_frame_extraction_dir = "output/_sam2_frames"
sam2_test_preview_dir = "output/_sam2_test"
chunk_output_dir = "output/_chunks"
trimmed_video_dir = "output/_trimmed"
resampled_video_dir = "output/_resampled"
job_store_path = "video_jobs.json"     # per-video points + toggles, saved by the GUI

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


# ============================================================
# PERFORMANCE
# ============================================================

target_fps = 24
use_multiprocessing = True
# How many worker processes split each video's frames between them.
# Each worker holds its own slice of full-resolution frames + masks +
# (if face removal is on) its own MediaPipe face model resident at
# once - on 1080p footage this adds up fast, so keep this modest
# unless you know the machine has RAM to spare. None -> min(cpu_count, 3).
# If you see "Unable to allocate ... MiB" or OutOfMemoryError errors,
# lower this (2, or 1 to disable parallelism entirely) before anything
# else.
num_worker_processes = None


# ============================================================
# COLOR SCHEME
# ============================================================

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

edge_min_component_area = 40


# ============================================================
# FACE REMOVAL (flattening)
# ============================================================

# Global default - ON. The GUI shows this as a per-video checkbox
# pre-ticked to this value; each video's actual on/off choice is
# stored in the job store, not written back here.
use_face_blur_default = True

face_landmarker_model_path = "models/face_landmarker.task"
face_detection_confidence = 0.2
face_blur_feather_ksize = (51, 51)
face_search_tile_grid = (3, 2)       # (rows, cols)
face_search_tile_overlap = 0.2
head_crop_upscale_target = 512
use_face_tracking = True
face_tracker_max_miss_streak = 5


# ============================================================
# SAM2 SEGMENTATION
# ============================================================

use_sam2 = True

sam2_checkpoint_path = "EdgeTAM/checkpoints/edgetam.pt"
sam2_model_config_path = "configs/edgetam.yaml"

sam2_vos_optimized = False
sam2_offload_video_to_cpu = True
sam2_offload_state_to_cpu = True

sam2_mask_dilate_ksize = (0, 0)
sam2_outline_thickness = 8
sam2_style = "outline"   # "filled" | "outline"

sam2_test_num_preview_frames = 8
