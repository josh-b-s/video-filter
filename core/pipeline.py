"""
Responsibility: orchestrate one full video through the pipeline -
SAM2 masks -> chunked/parallel edge_detect+recolor -> stitched output.
This is the only module that knows how the other pieces fit together;
it doesn't implement any of them itself.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm import tqdm

import config
from core import video_io, face_flatten

logger = logging.getLogger(__name__)


def make_sure_file_exists(file_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")


def _process_frame_range(
    video_path: str,
    start_frame: int,
    end_frame: int,
    masks: list,
    output_path: str,
    width: int,
    height: int,
    fps: float,
    face_removal_enabled: bool,
) -> str:
    """
    Worker function (must stay module-level to be picklable for
    multiprocessing's 'spawn' start method on Windows). Imports
    core.edges lazily so each spawned worker only pulls in torch/CUDA
    if edges.py's face_flatten import chain needs it - matches the
    original canny.py's reasoning for keeping heavy imports out of the
    worker process's startup path.
    """
    from core import edges  # local import - see docstring

    face_flatten.reset_tracker_state()

    video = cv.VideoCapture(video_path)
    video.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(output_path, fourcc, fps, (width, height), True)

    for i in tqdm(range(end_frame - start_frame), desc=f"Frames {start_frame}-{end_frame}"):
        success, frame = video.read()
        if not success:
            break
        mask = masks[i]
        edge_map = edges.edge_detect(frame, mask=mask, face_removal_enabled=face_removal_enabled)
        output_frame = edges.recolor(edge_map, mask=mask)
        writer.write(output_frame)

    writer.release()
    video.release()
    return output_path


def process_video(
    input_path: str,
    output_path: str,
    prompt_points: list,
    prompt_labels: list,
    use_face_blur: bool,
    use_sam2: bool = True,
    crop_rect: tuple[int, int, int, int] | None = None,
    trim_start_frame: int = 0,
    trim_end_frame: int | None = None,
) -> None:
    """
    Runs the full pipeline for one video using its own already-picked
    prompt points, face-removal toggle, and crop/trim range (from
    job_store) - does not touch or depend on any other video's state.
    """
    make_sure_file_exists(input_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    working_input_path = input_path

    # Trim + crop run first, before FPS resampling - this cuts frame
    # count and resolution before SAM2, edge detection, or the resample
    # step ever touch the video, so it's a speedup as well as a
    # convenience. trim_start/end_frame are indices into the ORIGINAL
    # video, which is why this has to happen before resampling shifts
    # what's at each index.
    if crop_rect is not None or trim_start_frame or trim_end_frame is not None:
        working_input_path = video_io.trim_and_crop_video(
            working_input_path, trim_start_frame, trim_end_frame, crop_rect,
            config.trimmed_video_dir,
        )

    if config.target_fps is not None:
        working_input_path = video_io.resample_video_to_fps(working_input_path, config.target_fps, config.resampled_video_dir)

    info = video_io.get_video_info(working_input_path)
    width, height, fps, total_frames = info["width"], info["height"], info["fps"], info["total_frames"]
    logger.info(f"width: {width}; height: {height}; fps: {fps}; total_frames: {total_frames}.")

    masks: list = [None] * total_frames
    if use_sam2:
        from core.sam2_segment import Sam2Segmenter, build_config_for_video

        logger.info("Running SAM2 segmentation pass...")
        segmenter = Sam2Segmenter(build_config_for_video(working_input_path, prompt_points, prompt_labels))
        segmenter.prepare_frames(working_input_path)
        if segmenter._num_frames != total_frames:
            raise RuntimeError(
                f"SAM2 extracted {segmenter._num_frames} frames but the video has "
                f"{total_frames} frames - these should always match."
            )
        masks = list(tqdm(segmenter.propagate_masks(), total=total_frames, desc="Propagating masks..."))
    else:
        logger.info("use_sam2 is off for this video - skipping segmentation, no mask applied.")

    chunk_dir = os.path.join(config.chunk_output_dir, Path(input_path).stem)

    if not config.use_multiprocessing or config.num_worker_processes == 1:
        chunk_path = _process_frame_range(
            working_input_path, 0, total_frames, masks, output_path, width, height, fps, use_face_blur,
        )
    else:
        num_workers = config.num_worker_processes or min(os.cpu_count() or 1, 3)
        num_workers = min(num_workers, max(1, total_frames))

        chunk_boundaries = np.linspace(0, total_frames, num_workers + 1, dtype=int)
        os.makedirs(chunk_dir, exist_ok=True)

        chunk_args = []
        for i in range(num_workers):
            start, end = int(chunk_boundaries[i]), int(chunk_boundaries[i + 1])
            if end <= start:
                continue
            chunk_output_path = os.path.join(chunk_dir, f"chunk_{i:03d}.mp4")
            chunk_args.append((
                working_input_path, start, end, masks[start:end],
                chunk_output_path, width, height, fps, use_face_blur,
            ))

        logger.info(f"Processing {total_frames} frames across {len(chunk_args)} worker processes...")

        with ProcessPoolExecutor(max_workers=len(chunk_args)) as executor:
            futures = [executor.submit(_process_frame_range, *args) for args in chunk_args]
            chunk_paths = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks..."):
                chunk_paths.append(future.result())

        chunk_paths.sort()
        logger.info("Concatenating chunks into final output...")
        video_io.concatenate_chunks(chunk_paths, output_path, fps, width, height)

        for chunk_path in chunk_paths:
            os.remove(chunk_path)

    if use_face_blur:
        face_flatten.log_face_detection_summary(video_label=Path(input_path).stem)

    logger.info(f"Video written to: {output_path}")

    # Release this video's large in-memory state (frame masks especially -
    # a full-resolution bool mask per frame, times however many frames the
    # clip has, adds up fast) before the next queued video starts. Without
    # this, memory from one video can still be resident when the next one
    # begins, and it compounds video after video until allocations start
    # failing outright.
    del masks
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def process_job(video_path: str, job) -> str:
    """Convenience wrapper the GUI/CLI call with a job_store.VideoJob."""
    filename_noext = Path(video_path).stem
    output_path = os.path.join(config.output_folder, f"{filename_noext}.mp4")
    process_video(
        video_path, output_path, job.prompt_points, job.prompt_labels, job.use_face_blur,
        use_sam2=job.use_sam2,
        crop_rect=job.crop_rect, trim_start_frame=job.trim_start_frame, trim_end_frame=job.trim_end_frame,
    )
    return output_path
