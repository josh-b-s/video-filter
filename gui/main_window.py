"""
Responsibility: wire the video list, the point-picker canvas, the
per-video face-removal toggle, and the background queue worker
together. This is the only module that owns "what happens when you
click Confirm" - everything it calls (job_store, pipeline, canvas) is
single-purpose on its own.
"""

import logging
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QListWidget,
    QListWidgetItem, QPushButton, QCheckBox, QLabel, QPlainTextEdit,
    QFileDialog, QSplitter, QMessageBox,
)

import config
from core import job_store, video_io
from gui.point_picker import PointPickerCanvas
from gui.queue_worker import QueueWorker

logger = logging.getLogger(__name__)

STATUS_ICON = {
    "unpicked": "\u25cb",    # ○
    "queued": "\u23f3",      # ⏳
    "processing": "\u25b6",  # ▶
    "done": "\u2705",        # ✅
    "error": "\u26a0",       # ⚠
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Edge Art Pipeline")
        self.resize(1280, 800)

        self.folder = config.input_folder
        self.video_paths: dict[str, str] = {}   # filename -> full path
        self.current_filename: str | None = None

        self.worker = QueueWorker()
        self.worker.job_started.connect(self._on_job_started)
        self.worker.job_finished.connect(self._on_job_finished)
        self.worker.job_failed.connect(self._on_job_failed)
        self.worker.log_message.connect(self._log)

        self._build_ui()
        self._load_folder(self.folder)
        self.worker.start()

    # ------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # top bar
        top_bar = QHBoxLayout()
        self.folder_label = QLabel(self.folder)
        choose_btn = QPushButton("Choose video folder...")
        choose_btn.clicked.connect(self._choose_folder)
        top_bar.addWidget(QLabel("Folder:"))
        top_bar.addWidget(self.folder_label, 1)
        top_bar.addWidget(choose_btn)
        root.addLayout(top_bar)

        splitter = QSplitter()
        root.addWidget(splitter, 1)

        # left: video list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Videos"))
        self.video_list = QListWidget()
        self.video_list.currentItemChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self.video_list, 1)
        splitter.addWidget(left)

        # middle: picker canvas + controls
        middle = QWidget()
        middle_layout = QVBoxLayout(middle)
        self.canvas = PointPickerCanvas()
        self.canvas.mask_error.connect(lambda msg: self._log(f"SAM2 mask preview error:\n{msg}"))
        middle_layout.addWidget(self.canvas, 1)

        help_label = QLabel(
            "Left click = keep this region (foreground)   |   "
            "Right click = exclude (background)\n"
            "Click once on each part of the subject (skin, hair, top, pants, shoes) "
            "so the whole person is selected."
        )
        help_label.setWordWrap(True)
        middle_layout.addWidget(help_label)

        controls_row = QHBoxLayout()
        self.face_checkbox = QCheckBox("Remove face")
        self.face_checkbox.setChecked(config.use_face_blur_default)
        controls_row.addWidget(self.face_checkbox)

        undo_btn = QPushButton("Undo point")
        undo_btn.clicked.connect(self.canvas.undo_point)
        controls_row.addWidget(undo_btn)

        clear_btn = QPushButton("Clear points")
        clear_btn.clicked.connect(self.canvas.clear_points)
        controls_row.addWidget(clear_btn)

        controls_row.addStretch(1)

        self.confirm_btn = QPushButton("Confirm && Queue \u2192 Next video")
        self.confirm_btn.setDefault(True)
        self.confirm_btn.clicked.connect(self._confirm_and_next)
        controls_row.addWidget(self.confirm_btn)

        middle_layout.addLayout(controls_row)
        splitter.addWidget(middle)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # bottom: log
        root.addWidget(QLabel("Log"))
        self.log_box = QPlainTextEdit(readOnly=True)
        self.log_box.setMaximumHeight(140)
        root.addWidget(self.log_box)

    # ------------------------------------------------------------
    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose video folder", self.folder)
        if folder:
            self._load_folder(folder)

    def _load_folder(self, folder: str) -> None:
        self.folder = folder
        self.folder_label.setText(folder)
        self.video_list.clear()
        self.video_paths = {}

        if not os.path.isdir(folder):
            self._log(f"Folder not found: {folder}")
            return

        jobs = job_store.load_all()
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith(config.VIDEO_EXTENSIONS):
                continue
            full_path = os.path.join(folder, name)
            self.video_paths[name] = full_path
            job = jobs.get(name, job_store.VideoJob(filename=name))
            item = QListWidgetItem(f"{STATUS_ICON.get(job.status, '')}  {name}")
            item.setData(Qt.UserRole, name)
            self.video_list.addItem(item)

            # Anything left over from a previous run in "queued" or
            # "processing" state needs re-queuing - the worker thread
            # doesn't persist across app restarts.
            if job.status in ("queued", "processing") and job.has_points:
                self.worker.enqueue(full_path, name)

        if self.video_list.count() > 0:
            self.video_list.setCurrentRow(0)

    def _refresh_item(self, filename: str) -> None:
        for i in range(self.video_list.count()):
            item = self.video_list.item(i)
            if item.data(Qt.UserRole) == filename:
                job = job_store.get_job(filename)
                item.setText(f"{STATUS_ICON.get(job.status, '')}  {filename}")
                return

    # ------------------------------------------------------------
    def _on_selection_changed(self, current: QListWidgetItem, _previous) -> None:
        if current is None:
            return
        filename = current.data(Qt.UserRole)
        self._load_video_for_picking(filename)

    def _load_video_for_picking(self, filename: str) -> None:
        self.current_filename = filename
        path = self.video_paths[filename]
        try:
            frame = video_io.read_first_frame(path)
        except Exception as e:
            self._log(f"Could not read {filename}: {e}")
            return

        self.canvas.load_frame(frame)

        job = job_store.get_job(filename)
        for (x, y), label in zip(job.prompt_points, job.prompt_labels):
            self.canvas.points.append((x, y))
            self.canvas.labels.append(label)
        if job.prompt_points:
            self.canvas._request_mask()
            self.canvas._redraw()
        self.face_checkbox.setChecked(job.use_face_blur)

    # ------------------------------------------------------------
    def _confirm_and_next(self) -> None:
        if self.current_filename is None:
            return
        if not self.canvas.points:
            QMessageBox.warning(self, "No points picked", "Click at least one point on the subject first.")
            return

        filename = self.current_filename
        job = job_store.VideoJob(
            filename=filename,
            prompt_points=list(self.canvas.points),
            prompt_labels=list(self.canvas.labels),
            use_face_blur=self.face_checkbox.isChecked(),
            status="queued",
        )
        job_store.save_job(job)
        self._refresh_item(filename)
        self.worker.enqueue(self.video_paths[filename], filename)

        self._select_next_unpicked()

    def _select_next_unpicked(self) -> None:
        jobs = job_store.load_all()
        current_row = self.video_list.currentRow()
        n = self.video_list.count()
        for offset in range(1, n + 1):
            row = (current_row + offset) % n
            item = self.video_list.item(row)
            filename = item.data(Qt.UserRole)
            job = jobs.get(filename)
            if job is None or job.status == "unpicked":
                self.video_list.setCurrentRow(row)
                return
        # everything already picked - just move to the next one in the list
        self.video_list.setCurrentRow((current_row + 1) % n)

    # ------------------------------------------------------------
    def _on_job_started(self, filename: str) -> None:
        self._refresh_item(filename)

    def _on_job_finished(self, filename: str, output_path: str) -> None:
        self._refresh_item(filename)

    def _on_job_failed(self, filename: str, message: str) -> None:
        self._refresh_item(filename)
        self._log(f"ERROR processing {filename}: {message}")

    def _log(self, message: str) -> None:
        self.log_box.appendPlainText(message)

    def closeEvent(self, event) -> None:
        self.worker.stop()
        self.worker.wait(2000)
        super().closeEvent(event)
