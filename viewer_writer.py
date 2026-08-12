"""Writes a self-contained viewer HTML file by injecting a JSON payload
into a template. Shared by export_landmarks.py and
export_landmarks_rigged.py."""

import json
import logging

logger = logging.getLogger(__name__)


def write_viewer_html(payload: dict, template_path: str, out_path: str) -> None:
    with open(template_path, "r") as f:
        template = f.read()

    html = template.replace("__SKELETON_DATA__", json.dumps(payload))

    with open(out_path, "w") as f:
        f.write(html)

    logger.info(f"Viewer written to: {out_path}")
