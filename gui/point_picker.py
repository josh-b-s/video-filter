"""
Responsibility: the clickable canvas - shows frame 0 of a video,
records fg/bg click points in original-frame coordinates, and shows a
live SAM2 mask preview. Knows nothing about queues, job persistence,
or other videos; the main window reads .points/.labels off it and
decides what to do with them.
"""

import cv2 as cv
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout

from core.sam2_segment import predict_mask_for_points

FG_COLOR = QColor(80, 220, 120)
BG_COLOR = QColor(230, 70, 70)
MASK_COLOR_BGR = (60, 60, 230)  # red, applied as a BGR overlay tint


class MaskPreviewThread(QThread):
    """Runs SAM2's single-image predictor off the UI thread so clicking
    stays snappy even while a mask is being computed."""
    mask_ready = Signal(object, int)  # mask (np.ndarray | None), request_id
    mask_error = Signal(str)

    def __init__(self):
        super().__init__()
        self._frame = None
        self._points: list[tuple[int, int]] = []
        self._labels: list[int] = []
        self._request_id = 0
        self._pending = False

    def request(self, frame: np.ndarray, points: list[tuple[int, int]], labels: list[int]) -> None:
        self._frame = frame
        self._points = list(points)
        self._labels = list(labels)
        self._request_id += 1
        if not self.isRunning():
            self.start()
        else:
            self._pending = True

    def run(self) -> None:
        rid = self._request_id
        mask = None
        if self._points:
            try:
                mask = predict_mask_for_points(self._frame, self._points, self._labels)
            except Exception:
                import traceback
                self.mask_error.emit(traceback.format_exc(limit=6))
        self.mask_ready.emit(mask, rid)
        if self._pending:
            self._pending = False
            self.run()


class PointPickerCanvas(QWidget):
    points_changed = Signal()
    mask_error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame_bgr: np.ndarray | None = None
        self.points: list[tuple[int, int]] = []
        self.labels: list[int] = []
        self.current_mask: np.ndarray | None = None

        self._image_label = QLabel(alignment=Qt.AlignCenter)
        self._image_label.setMinimumSize(480, 270)
        # Without this, clicks land on the QLabel (it's on top) and
        # never reach this widget's mousePressEvent below.
        self._image_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._image_label)

        self._mask_thread = MaskPreviewThread()
        self._mask_thread.mask_ready.connect(self._on_mask_ready)
        self._mask_thread.mask_error.connect(self.mask_error)
        self._latest_request_id = 0

        self._display_scale = 1.0
        self._pixmap_origin = QPoint(0, 0)  # top-left of the scaled pixmap within _image_label

    # ------------------------------------------------------------
    def load_frame(self, frame_bgr: np.ndarray) -> None:
        self.frame_bgr = frame_bgr
        self.points = []
        self.labels = []
        self.current_mask = None
        self._redraw()

    def clear_points(self) -> None:
        self.points = []
        self.labels = []
        self.current_mask = None
        self._redraw()
        self.points_changed.emit()

    def undo_point(self) -> None:
        if self.points:
            self.points.pop()
            self.labels.pop()
            self._request_mask()
            self._redraw()
            self.points_changed.emit()

    # ------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.frame_bgr is None or self._display_scale == 0:
            return
        label_pos = self._image_label.mapFrom(self, event.position().toPoint())
        x = (label_pos.x() - self._pixmap_origin.x()) / self._display_scale
        y = (label_pos.y() - self._pixmap_origin.y()) / self._display_scale
        h, w = self.frame_bgr.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return

        if event.button() == Qt.LeftButton:
            self.points.append((int(x), int(y)))
            self.labels.append(1)
        elif event.button() == Qt.RightButton:
            self.points.append((int(x), int(y)))
            self.labels.append(0)
        else:
            return

        self._request_mask()
        self._redraw()
        self.points_changed.emit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._redraw()

    # ------------------------------------------------------------
    def _request_mask(self) -> None:
        if self.frame_bgr is None:
            return
        self._mask_thread.request(self.frame_bgr, self.points, self.labels)

    def _on_mask_ready(self, mask, request_id: int) -> None:
        if mask is not None and self.frame_bgr is not None and mask.shape[:2] != self.frame_bgr.shape[:2]:
            # Stale result: computed for a frame that's since been
            # replaced (e.g. the user switched to a different video,
            # a different resolution, while this mask was still being
            # computed on a background thread). Applying it would
            # crash _redraw()'s boolean-index assignment below, since
            # a differently-shaped mask can't index this frame -
            # discard it instead. A fresh, correctly-shaped mask for
            # the new frame follows shortly from the next click.
            return
        self.current_mask = mask
        self._redraw()

    def _redraw(self) -> None:
        if self.frame_bgr is None:
            self._image_label.setText("No frame loaded")
            return

        display = self.frame_bgr.copy()
        if self.current_mask is not None:
            overlay = display.copy()
            overlay[self.current_mask] = MASK_COLOR_BGR
            display = cv.addWeighted(overlay, 0.45, display, 0.55, 0)

        h, w = display.shape[:2]
        rgb = cv.cvtColor(display, cv.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        target_w = max(self._image_label.width(), 1)
        target_h = max(self._image_label.height(), 1)
        scaled = pixmap.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._display_scale = scaled.width() / w if w else 1.0
        # AlignCenter means the scaled pixmap sits centered inside the
        # (usually larger) label - record that offset so click
        # coordinates can be mapped back correctly.
        self._pixmap_origin = QPoint(
            (target_w - scaled.width()) // 2,
            (target_h - scaled.height()) // 2,
        )

        painter = QPainter(scaled)
        painter.setRenderHint(QPainter.Antialiasing)
        for (x, y), label in zip(self.points, self.labels):
            color = FG_COLOR if label == 1 else BG_COLOR
            painter.setPen(QPen(Qt.white, 2))
            painter.setBrush(color)
            px, py = x * self._display_scale, y * self._display_scale
            painter.drawEllipse(QPoint(int(px), int(py)), 6, 6)
        painter.end()

        self._image_label.setPixmap(scaled)