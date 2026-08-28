"""
Headless CLI: processes every video in config.input_folder that
already has picked points saved in the job store (status "queued" or
"done" get reprocessed; "unpicked" videos are skipped with a warning).

Use this on a render machine after someone has already picked points
for a batch via the GUI (video_jobs.json travels with the videos).

Usage:
    python process_queue_cli.py
"""

import logging
import os

import config
from core import job_store, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    jobs = job_store.load_all()
    folder = config.input_folder

    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(config.VIDEO_EXTENSIONS):
            continue

        job = jobs.get(name)
        if job is None or not job.has_points:
            logger.warning(f"Skipping {name} - no prompt points picked yet.")
            continue

        video_path = os.path.join(folder, name)
        logger.info(f"Processing {name}...")
        try:
            output_path = pipeline.process_job(video_path, job)
            job.status = "done"
            job.error_message = None
            logger.info(f"Done: {name} -> {output_path}")
        except Exception as e:
            job.status = "error"
            job.error_message = str(e)
            logger.exception(f"Failed: {name}")
        job_store.save_job(job)


if __name__ == "__main__":
    main()
