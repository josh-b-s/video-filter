"""
EdgeTAM-based video foreground segmentation (Meta's fast, distilled
SAM2 variant - same API, ~19x faster and lower VRAM on tested hardware).

Responsibility: given an input video and a prompt (point or box) on the
first frame, produce a per-frame binary foreground mask, propagated
through the clip using EdgeTAM's video predictor.

This module knows nothing about edge detection / recoloring — it just
hands back masks. `canny.py` (or anything else) consumes them.

Requires:
    pip install torch torchvision
    git clone https://github.com/facebookresearch/EdgeTAM.git && cd EdgeTAM && pip install -e .
    pip install timm  (EdgeTAM's image encoder backbone dependency,
                        not pulled in automatically by the install above)

You also need the EdgeTAM checkpoint + model config, e.g.:
    EdgeTAM/checkpoints/edgetam.pt
    configs/edgetam.yaml

Note: EdgeTAM's install registers its own `sam2` package. If you also
have the original facebookresearch/sam2 installed in the same venv,
only one will win `import sam2` depending on install order - uninstall
the one you're not using to avoid confusion (`pip uninstall SAM-2` for
the original repo's package, from inside that repo).
"""

import contextlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Sam2Config:
    checkpoint_path: str = "checkpoints/sam2.1_hiera_tiny.pt"
    model_config_path: str = "configs/sam2.1/sam2.1_hiera_t.yaml"

    # Where SAM2 needs the video as a directory of JPEG frames.
    # (The official video predictor reads from a frame directory, not
    # directly from an mp4.)
    frame_extraction_dir: str = "output/_sam2_frames"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Prompt type + coordinates, defined on frame 0, in pixel space.
    prompt_type: Literal["point", "box"] = "point"

    # For prompt_type == "point": list of (x, y) pixel coords.
    prompt_points: list[tuple[int, int]] | None = None
    # labels: 1 = foreground click, 0 = background click. Same length as prompt_points.
    prompt_labels: list[int] | None = None

    # For prompt_type == "box": (x1, y1, x2, y2) in pixel coords.
    prompt_box: tuple[int, int, int, int] | None = None

    # Object id to track (SAM2 supports multiple objects; we default to one).
    object_id: int = 1

    # Uses torch.compile on the full SAM2 model (not just the image
    # encoder) via SAM2VideoPredictorVOS, per the official repo's
    # documented fast path for video inference. First call after
    # building the predictor will be slow (compilation warmup); every
    # frame after that should be significantly faster.
    # Requires torch>=2.5.1. If you hit torch.compile errors (more
    # common on Windows than Linux), set this back to False.
    vos_optimized: bool = False

    # Memory placement for video frames / propagation state. True keeps
    # them in system RAM and shuttles to GPU per-frame (safe on small
    # VRAM, but the CPU<->GPU transfer is itself the main overhead).
    # False keeps everything resident on GPU (much less overhead, but
    # will OOM if the clip + model doesn't fit in VRAM).
    # Try False first on short/low-res clips to use available VRAM
    # more fully; fall back to True if you hit CUDA OOM.
    offload_video_to_cpu: bool = True
    offload_state_to_cpu: bool = True


def build_config_from_settings() -> "Sam2Config":
    """
    Builds a Sam2Config using the shared settings in config.py, plus
    whatever prompt points were last saved by pick_prompt_points.py
    (or config.py's fallback default if none saved yet).

    This is the normal way to get a Sam2Config in this project -
    canny.py and test_sam2.py both use it, so editing config.py is
    enough to change behavior everywhere without touching this file.
    """
    import config

    points, labels = config.load_prompt_points()

    return Sam2Config(
        checkpoint_path=config.sam2_checkpoint_path,
        model_config_path=config.sam2_model_config_path,
        frame_extraction_dir=config.sam2_frame_extraction_dir,
        prompt_type="point",
        prompt_points=points,
        prompt_labels=labels,
        vos_optimized=config.sam2_vos_optimized,
        offload_video_to_cpu=config.sam2_offload_video_to_cpu,
        offload_state_to_cpu=config.sam2_offload_state_to_cpu,
    )


# ============================================================
# SEGMENTER
# ============================================================

class Sam2Segmenter:
    """
    Wraps SAM2's video predictor to yield one boolean mask per frame.

    Usage:
        segmenter = Sam2Segmenter(config)
        segmenter.prepare_frames(input_video_path)   # extracts frames to disk once
        for mask in segmenter.propagate_masks():
            ...  # mask is a (H, W) bool array, True = foreground
    """

    def __init__(self, config: Sam2Config):
        self.config = config
        self._predictor = None
        self._inference_state = None
        self._num_frames = 0
        self._frame_shape: tuple[int, int] | None = None  # (H, W)

    # ------------------------------------------------------------
    # setup
    # ------------------------------------------------------------

    def _build_predictor(self):
        if self._predictor is not None:
            return self._predictor

        # Imported lazily so this module can be imported without SAM2
        # installed, as long as segmentation is never actually used.
        from sam2.build_sam import build_sam2_video_predictor

        logger.info(
            "Loading SAM2 video predictor (%s)%s...",
            self.config.model_config_path,
            " [vos_optimized]" if self.config.vos_optimized else "",
        )
        self._predictor = build_sam2_video_predictor(
            self.config.model_config_path,
            self.config.checkpoint_path,
            device=self.config.device,
            vos_optimized=self.config.vos_optimized,
        )
        if self.config.vos_optimized:
            logger.info(
                "vos_optimized is on: the first propagate_masks() call will be slow "
                "while torch.compile warms up, then faster on subsequent frames."
            )
        return self._predictor

    def prepare_frames(self, input_video_path: str) -> str:
        """
        Extracts video frames to `frame_extraction_dir` as JPEGs
        (0000000.jpg, 0000001.jpg, ...), which is the input format
        SAM2's video predictor expects. Returns the directory path.

        Skips re-extraction if the directory already holds frames
        matching this exact video (checked via a marker file storing
        the source path + frame count + resolution) — but re-extracts
        if you point at a different video, instead of silently reusing
        stale frames from a previous run.
        """
        import cv2 as cv

        frame_dir = self.config.frame_extraction_dir
        os.makedirs(frame_dir, exist_ok=True)

        marker_path = os.path.join(frame_dir, "_source.json")
        source_abspath = os.path.abspath(input_video_path)

        existing = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
        if existing and os.path.exists(marker_path):
            with open(marker_path) as f:
                marker = json.load(f)
            if marker.get("source_video") == source_abspath and marker.get("frame_count") == len(existing):
                logger.info("Reusing %d already-extracted frames in %s (matches %s)",
                             len(existing), frame_dir, input_video_path)
                self._num_frames = len(existing)
                first = cv.imread(os.path.join(frame_dir, existing[0]))
                self._frame_shape = first.shape[:2]
                return frame_dir
            else:
                logger.info(
                    "Frame cache in %s is from a different video (%s) - re-extracting for %s.",
                    frame_dir, marker.get("source_video"), input_video_path,
                )
                for f in existing:
                    os.remove(os.path.join(frame_dir, f))
        elif existing:
            # Old cache from before this marker system existed - can't
            # verify it matches, so re-extract to be safe.
            logger.info("Frame cache in %s has no source marker - re-extracting to be safe.", frame_dir)
            for f in existing:
                os.remove(os.path.join(frame_dir, f))

        video = cv.VideoCapture(input_video_path)
        if not video.isOpened():
            raise RuntimeError(f"Could not open video: {input_video_path}")

        idx = 0
        while True:
            success, frame = video.read()
            if not success:
                break
            if self._frame_shape is None:
                self._frame_shape = frame.shape[:2]
            out_path = os.path.join(frame_dir, f"{idx:07d}.jpg")
            cv.imwrite(out_path, frame, [cv.IMWRITE_JPEG_QUALITY, 95])
            idx += 1
        video.release()

        self._num_frames = idx
        with open(marker_path, "w") as f:
            json.dump({"source_video": source_abspath, "frame_count": idx}, f)

        logger.info("Extracted %d frames to %s", idx, frame_dir)
        return frame_dir

    # ------------------------------------------------------------
    # prompting + propagation
    # ------------------------------------------------------------

    def _init_state_and_prompt(self):
        predictor = self._build_predictor()

        self._inference_state = predictor.init_state(
            video_path=self.config.frame_extraction_dir,
            offload_video_to_cpu=self.config.offload_video_to_cpu,
            offload_state_to_cpu=self.config.offload_state_to_cpu,
        )
        predictor.reset_state(self._inference_state)

        cfg = self.config
        if cfg.prompt_type == "point":
            if not cfg.prompt_points:
                raise ValueError("prompt_type='point' requires prompt_points to be set.")
            labels = cfg.prompt_labels or [1] * len(cfg.prompt_points)
            points = np.array(cfg.prompt_points, dtype=np.float32)
            labels_arr = np.array(labels, dtype=np.int32)

            predictor.add_new_points_or_box(
                inference_state=self._inference_state,
                frame_idx=0,
                obj_id=cfg.object_id,
                points=points,
                labels=labels_arr,
            )
        elif cfg.prompt_type == "box":
            if not cfg.prompt_box:
                raise ValueError("prompt_type='box' requires prompt_box to be set.")
            box = np.array(cfg.prompt_box, dtype=np.float32)

            predictor.add_new_points_or_box(
                inference_state=self._inference_state,
                frame_idx=0,
                obj_id=cfg.object_id,
                box=box,
            )
        else:
            raise ValueError(f"Unknown prompt_type: {cfg.prompt_type}")

    def propagate_masks(self) -> Iterator[np.ndarray]:
        """
        Runs EdgeTAM propagation across the whole video and yields one
        (H, W) bool mask per frame, in frame order. Must be called
        after prepare_frames().

        Wrapped in inference_mode + bfloat16 autocast on CUDA, matching
        EdgeTAM's own recommended usage - this materially affects speed
        and memory, not just a style preference.
        """
        if self._num_frames == 0:
            raise RuntimeError("Call prepare_frames() before propagate_masks().")

        use_autocast = self.config.device == "cuda" and torch.cuda.is_available()
        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast
            else contextlib.nullcontext()
        )

        with torch.inference_mode(), autocast_ctx:
            predictor = self._build_predictor()
            self._init_state_and_prompt()

            if use_autocast:
                torch.cuda.reset_peak_memory_stats()
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                logger.info(
                    "GPU memory after init_state: %.2f GiB allocated, %.2f GiB reserved, %.2f GiB total.",
                    allocated, reserved, total,
                )

            # frame_idx -> mask, collected then yielded in order since the
            # predictor's propagate_in_video can process forward from the
            # prompted frame.
            masks_by_frame: dict[int, np.ndarray] = {}

            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(self._inference_state):
                if self.config.object_id not in obj_ids:
                    masks_by_frame[frame_idx] = np.zeros(self._frame_shape, dtype=bool)
                    continue
                obj_pos = obj_ids.index(self.config.object_id)
                mask = (mask_logits[obj_pos] > 0.0).cpu().numpy().squeeze()
                masks_by_frame[frame_idx] = mask.astype(bool)

            logger.info("Propagated masks for %d frames.", len(masks_by_frame))

            if use_autocast:
                peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
                logger.info("Peak GPU memory during propagation: %.2f GiB.", peak)

        for idx in range(self._num_frames):
            yield masks_by_frame.get(idx, np.zeros(self._frame_shape, dtype=bool))
