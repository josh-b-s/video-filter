"""
Entry point for the point-picker / queue GUI.

Usage:
    python app.py

Packaging into a standalone .exe: see build_exe.md.
"""

import logging
import sys

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
    main()
