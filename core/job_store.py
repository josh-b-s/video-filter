"""
Responsibility: persist per-video picking/queue state (prompt points,
prompt labels, the face-removal toggle, and queue status) to a single
JSON file, keyed by the video's filename.

This replaces config.py's old single global prompt_points.json, which
only ever supported one video at a time. Every other module (GUI,
pipeline) reads/writes state through this module rather than touching
the JSON file directly.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, asdict
from typing import Literal

import config

_lock = threading.Lock()

JobStatus = Literal["unpicked", "queued", "processing", "done", "error"]


@dataclass
class VideoJob:
    filename: str                              # basename, e.g. "clip1.mp4"
    prompt_points: list[tuple[int, int]] = field(default_factory=list)
    prompt_labels: list[int] = field(default_factory=list)
    use_face_blur: bool = config.use_face_blur_default
    use_sam2: bool = config.use_sam2
    # Crop rectangle in the ORIGINAL video's pixel coordinates
    # (x1, y1, x2, y2). None = no cropping, use the full frame.
    crop_rect: tuple[int, int, int, int] | None = None
    # Trim range in ORIGINAL (pre-resample) frame indices, [start, end).
    # trim_end_frame=None means "to the end of the clip".
    trim_start_frame: int = 0
    trim_end_frame: int | None = None
    status: JobStatus = "unpicked"
    error_message: str | None = None

    @property
    def has_points(self) -> bool:
        return len(self.prompt_points) > 0


def _load_raw() -> dict:
    if not os.path.exists(config.job_store_path):
        return {}
    with open(config.job_store_path) as f:
        return json.load(f)


def _save_raw(data: dict) -> None:
    tmp_path = config.job_store_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, config.job_store_path)


def load_all() -> dict[str, VideoJob]:
    with _lock:
        raw = _load_raw()
    jobs = {}
    for filename, entry in raw.items():
        crop_rect = entry.get("crop_rect")
        jobs[filename] = VideoJob(
            filename=filename,
            prompt_points=[tuple(p) for p in entry.get("prompt_points", [])],
            prompt_labels=entry.get("prompt_labels", []),
            use_face_blur=entry.get("use_face_blur", config.use_face_blur_default),
            use_sam2=entry.get("use_sam2", config.use_sam2),
            crop_rect=tuple(crop_rect) if crop_rect else None,
            trim_start_frame=entry.get("trim_start_frame", 0),
            trim_end_frame=entry.get("trim_end_frame"),
            status=entry.get("status", "unpicked"),
            error_message=entry.get("error_message"),
        )
    return jobs


def save_job(job: VideoJob) -> None:
    with _lock:
        raw = _load_raw()
        raw[job.filename] = asdict(job)
        _save_raw(raw)


def get_job(filename: str) -> VideoJob:
    jobs = load_all()
    return jobs.get(filename, VideoJob(filename=filename))
