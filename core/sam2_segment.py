"""
EdgeTAM-based video foreground segmentation (Meta's fast, distilled
SAM2 variant). Given an input video + a prompt (points) on frame 0,
produces a per-frame binary foreground mask, propagated through the
clip. Knows nothing about edge detection/recoloring - callers consume
the masks.

Requires:
    pip install torch torchvision
    git clone https://github.com/facebookresearch/EdgeTAM.git && cd EdgeTAM && pip install -e .
    pip install timm
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


@dataclass
class Sam2Config:
    checkpoint_path: str = "checkpoints/sam2.1_hiera_tiny.pt"
    model_config_path: str = "configs/sam2.1/sam2.1_hiera_t.yaml"
    frame_extraction_dir: str = "output/_sam2_frames"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    prompt_type: Literal["point", "box"] = "point"
    prompt_points: list[tuple[int, int]] | None = None
    prompt_labels: list[int] | None = None
    prompt_box: tuple[int, int, int, int] | None = None

    object_id: int = 1
    vos_optimized: bool = False
    offload_video_to_cpu: bool = True
    offload_state_to_cpu: bool = True


def build_config_for_video(video_path: str, prompt_points: list[tuple[int, int]], prompt_labels: list[int]) -> "Sam2Config":
    """
    Builds a Sam2Config for one specific video + its already-picked
    prompt points (from job_store), using the shared settings in
    config.py for everything else.
    """
    import config

    frame_dir = os.path.join(config.sam2_frame_extraction_dir, os.path.splitext(os.path.basename(video_path))[0])

    return Sam2Config(
        checkpoint_path=config.sam2_checkpoint_path,
        model_config_path=config.sam2_model_config_path,
        frame_extraction_dir=frame_dir,
        prompt_type="point",
        prompt_points=prompt_points,
        prompt_labels=prompt_labels,
        vos_optimized=config.sam2_vos_optimized,
        offload_video_to_cpu=config.sam2_offload_video_to_cpu,
        offload_state_to_cpu=config.sam2_offload_state_to_cpu,
    )


class Sam2Segmenter:
    """
    Usage:
        segmenter = Sam2Segmenter(config)
        segmenter.prepare_frames(input_video_path)
        for mask in segmenter.propagate_masks():
            ...
    """

    def __init__(self, config: Sam2Config):
        self.config = config
        self._predictor = None
        self._inference_state = None
        self._num_frames = 0
        self._frame_shape: tuple[int, int] | None = None

    def _build_predictor(self):
        if self._predictor is not None:
            return self._predictor
        from sam2.build_sam import build_sam2_video_predictor

        if self.config.device == "cpu":
            # torch.cuda.is_available() decided this at Sam2Config
            # creation time (see the device field default) - logging
            # WHY here, not just what, so a slow run doesn't need a
            # separate investigation to find out whether the GPU was
            # ever in play at all.
            try:
                cuda_reason = "no CUDA-capable torch build installed" if not torch.backends.cuda.is_built() \
                    else "no CUDA device detected (no GPU, or driver too old for this torch build)"
            except Exception:
                cuda_reason = "torch.cuda.is_available() returned False"
            logger.warning("Running SAM2 on CPU (%s) - this will be much slower than GPU.", cuda_reason)

        logger.info(
            "Loading SAM2 video predictor (%s) on device=%s%s...",
            self.config.model_config_path,
            self.config.device,
            " [vos_optimized]" if self.config.vos_optimized else "",
        )
        self._predictor = build_sam2_video_predictor(
            self.config.model_config_path,
            self.config.checkpoint_path,
            device=self.config.device,
            vos_optimized=self.config.vos_optimized,
        )
        return self._predictor

    def prepare_frames(self, input_video_path: str) -> str:
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
                logger.info("Reusing %d already-extracted frames in %s", len(existing), frame_dir)
                self._num_frames = len(existing)
                first = cv.imread(os.path.join(frame_dir, existing[0]))
                self._frame_shape = first.shape[:2]
                return frame_dir
            else:
                for f in existing:
                    os.remove(os.path.join(frame_dir, f))
        elif existing:
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
                inference_state=self._inference_state, frame_idx=0, obj_id=cfg.object_id,
                points=points, labels=labels_arr,
            )
        elif cfg.prompt_type == "box":
            if not cfg.prompt_box:
                raise ValueError("prompt_type='box' requires prompt_box to be set.")
            box = np.array(cfg.prompt_box, dtype=np.float32)
            predictor.add_new_points_or_box(
                inference_state=self._inference_state, frame_idx=0, obj_id=cfg.object_id, box=box,
            )
        else:
            raise ValueError(f"Unknown prompt_type: {cfg.prompt_type}")

    def propagate_masks(self) -> Iterator[np.ndarray]:
        if self._num_frames == 0:
            raise RuntimeError("Call prepare_frames() before propagate_masks().")

        use_autocast = self.config.device == "cuda" and torch.cuda.is_available()
        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else contextlib.nullcontext()
        )

        with torch.inference_mode(), autocast_ctx:
            self._build_predictor()
            self._init_state_and_prompt()

            masks_by_frame: dict[int, np.ndarray] = {}
            for frame_idx, obj_ids, mask_logits in self._predictor.propagate_in_video(self._inference_state):
                if self.config.object_id not in obj_ids:
                    masks_by_frame[frame_idx] = np.zeros(self._frame_shape, dtype=bool)
                    continue
                obj_pos = obj_ids.index(self.config.object_id)
                mask = (mask_logits[obj_pos] > 0.0).cpu().numpy().squeeze()
                masks_by_frame[frame_idx] = mask.astype(bool)

            logger.info("Propagated masks for %d frames.", len(masks_by_frame))

        for idx in range(self._num_frames):
            yield masks_by_frame.get(idx, np.zeros(self._frame_shape, dtype=bool))


# ------------------------------------------------------------
# Single-image predictor, used by the GUI's live point-picker preview
# (cheap - no video propagation, just "what would this click select").
# ------------------------------------------------------------

_image_predictor = None


def get_image_predictor():
    global _image_predictor
    if _image_predictor is None:
        import config
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_sam2(config.sam2_model_config_path, config.sam2_checkpoint_path, device=device)
        _image_predictor = SAM2ImagePredictor(model)
    return _image_predictor


def predict_mask_for_points(frame_bgr: np.ndarray, points: list[tuple[int, int]], labels: list[int]) -> np.ndarray | None:
    """Single-frame mask prediction for the GUI preview. Returns None if no points given."""
    import cv2 as cv

    if not points:
        return None
    predictor = get_image_predictor()
    predictor.set_image(cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB))
    point_coords = np.array(points, dtype=np.float32)
    point_labels = np.array(labels, dtype=np.int32)
    masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=False)
    return masks[0].astype(bool)