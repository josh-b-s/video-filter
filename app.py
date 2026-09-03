"""
Entry point for the point-picker / queue GUI.

Usage:
    python app.py

Packaging into a standalone .exe: see build_exe.md.
"""

import os
import sys

# When packaged with PyInstaller's --windowed flag, there's no console
# window, so sys.stdout/sys.stderr are None - any print() or logging
# call that writes to them crashes with "'NoneType' object has no
# attribute 'write'". Redirect both to a log file next to the exe
# instead: fixes the crash, and gives an actual file to check when
# something goes wrong (there's no console to see it in otherwise).
# Running from source (not frozen) is unaffected - sys.frozen is only
# set inside a PyInstaller build.
if getattr(sys, "frozen", False):
    log_path = os.path.join(os.path.dirname(sys.executable), "app.log")
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file

# Must happen before torch is imported anywhere (including
# transitively, via core/sam2_segment.py etc.) - PYTORCH_JIT=0 makes
# torch.jit.script() a no-op that returns the plain eager function
# instead of compiling it. Without this, torch.jit.script tries to
# read the function's source via inspect.getsource() at runtime, which
# fails inside a PyInstaller-packaged exe (bundled as compiled
# bytecode, not .py source) - SAM2Transforms.__init__ hits exactly
# this. Running from source works either way, so this is safe to
# leave on permanently rather than only in the packaged build.
os.environ.setdefault("PYTORCH_JIT", "0")

import logging

from PySide6.QtWidgets import QApplication

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from gui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    import multiprocessing
    # Required on Windows for a frozen (PyInstaller) exe that spawns
    # worker processes - core/pipeline.py does this via
    # config.use_multiprocessing for the chunk-processing step. Without
    # this, spawned workers on a frozen build can misbehave (re-running
    # this whole entry point instead of just the worker function),
    # which is a likely contributor to worker startup being slow/
    # unreliable in the packaged exe specifically. Must be the very
    # first thing that runs, before any other multiprocessing use.
    multiprocessing.freeze_support()
    main()