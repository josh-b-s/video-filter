"""Small shared filesystem helpers."""

import os


def make_sure_file_exists(file_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")
