"""
Responsibility: own the processing queue and run it on a background
thread, one video at a time (SAM2/GPU work doesn't parallelize well
across videos anyway), while the main thread stays free for picking
points on the next video. Talks to the rest of the GUI only through
Qt signals - never touches widgets directly.
"""

import logging
import queue
import traceback

from PySide6.QtCore import QThread, Signal

import config
from core import job_store, pipeline

logger = logging.getLogger(__name__)


class QueueWorker(QThread):
    job_started = Signal(str)                 # filename
    job_finished = Signal(str, str)            # filename, output_path
    job_failed = Signal(str, str)              # filename, error message
    log_message = Signal(str)

    def __init__(self):
        super().__init__()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._video_paths: dict[str, str] = {}   # filename -> full path
        self._stop_requested = False

    def enqueue(self, video_path: str, filename: str) -> None:
        self._video_paths[filename] = video_path
        self._queue.put(filename)
        self.log_message.emit(f"Queued: {filename}")

    def stop(self) -> None:
        self._stop_requested = True
        self._queue.put(None)  # unblock the get()

    def run(self) -> None:
        while not self._stop_requested:
            filename = self._queue.get()
            if filename is None:
                break

            job = job_store.get_job(filename)
            job.status = "processing"
            job_store.save_job(job)
            self.job_started.emit(filename)
            self.log_message.emit(f"Processing {filename}...")

            try:
                output_path = pipeline.process_job(self._video_paths[filename], job)
                job.status = "done"
                job.error_message = None
                job_store.save_job(job)
                self.job_finished.emit(filename, output_path)
                self.log_message.emit(f"Done: {filename} -> {output_path}")
            except Exception as e:
                logger.exception("Failed processing %s", filename)
                job.status = "error"
                job.error_message = str(e)
                job_store.save_job(job)
                self.job_failed.emit(filename, f"{e}\n{traceback.format_exc(limit=2)}")
            finally:
                # Clear whatever this job left resident (especially
                # relevant when it failed partway through - a crash
                # shouldn't leave memory pressure for the next queued
                # video to inherit).
                import gc
                gc.collect()
