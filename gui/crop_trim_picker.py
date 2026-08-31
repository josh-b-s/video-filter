"""
Responsibility: the crop-rectangle + trim-range canvas - shows a frame
from a video, lets you drag out a crop rectangle in original-frame
coordinates, and scrub two sliders to set the trim start/end frame.
Knows nothing about queues or job persistence; the main window reads
.crop_rect/.trim_start_frame/.trim_end_frame off it and decides what
to do with them (same division of responsibility as PointPickerCanvas).
"""

import cv2 as cv
import numpy as np
from PySide6.QtCore import Qt, QPoint, QRect, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSlider

CROP_COLOR = QColor(155, 111, 224)  # matches config.line_color_rgb (mobility purple)
MIN_DRAG_PX = 10  # ignore accidental clicks/tiny drags as "no crop selected"


class CropTrimCanvas(QWidget):
    rect_changed = Signal()
    trim_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_path: str | None = None
        self.total_frames = 0
        self.frame_w = 0
        self.frame_h = 0
        self.current_frame_bgr: np.ndarray | None = None

        self.crop_rect: tuple[int, int, int, int] | None = None  # (x1,y1,x2,y2), original coords
        self.trim_start_frame = 0
        self.trim_end_frame: int | None = None  # None = to the end of the clip

        self._drag_start: tuple[int, int] | None = None
        self._drag_current: tuple[int, int] | None = None
        self._dragging = False

        self._image_label = QLabel(alignment=Qt.AlignCenter)
        self._image_label.setMinimumSize(480, 270)
        # Without this, clicks land on the QLabel (it's on top) and
        # never reach this widget's mouse event handlers below - same
        # trick PointPickerCanvas uses.
        self._image_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self._start_slider = QSlider(Qt.Horizontal)
        self._end_slider = QSlider(Qt.Horizontal)
        self._start_slider.valueChanged.connect(self._on_start_slider)
        self._end_slider.valueChanged.connect(self._on_end_slider)

        self._start_label = QLabel("Start: 0")
        self._end_label = QLabel("End: \u2014")
        self._start_label.setMinimumWidth(90)
        self._end_label.setMinimumWidth(90)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._image_label, 1)

        start_row = QHBoxLayout()
        start_row.addWidget(self._start_label)
        start_row.addWidget(self._start_slider, 1)
        layout.addLayout(start_row)

        end_row = QHBoxLayout()
        end_row.addWidget(self._end_label)
        end_row.addWidget(self._end_slider, 1)
        layout.addLayout(end_row)

        self._display_scale = 1.0
        self._pixmap_origin = QPoint(0, 0)

    # ------------------------------------------------------------
    def load_video(self, video_path: str) -> None:
        """Loads a fresh video with no crop/trim picked yet. Call
        set_job_values() afterwards if this video already has saved
        values you want to restore."""
        self.video_path = video_path
        video = cv.VideoCapture(video_path)
        if not video.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        self.total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        self.frame_w = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
        video.release()

        self.crop_rect = None
        self.trim_start_frame = 0
        self.trim_end_frame = None

        last_frame_idx = max(0, self.total_frames - 1)
        for slider, val in ((self._start_slider, 0), (self._end_slider, last_frame_idx)):
            slider.blockSignals(True)
            slider.setMinimum(0)
            slider.setMaximum(last_frame_idx)
            slider.setValue(val)
            slider.blockSignals(False)

        self._update_labels()
        self._load_frame_at(0)

    def set_job_values(
        self,
        crop_rect: tuple[int, int, int, int] | None,
        trim_start_frame: int,
        trim_end_frame: int | None,
    ) -> None:
        """Restores previously-saved crop/trim values (call after
        load_video() when re-selecting a video that's already been
        picked)."""
        self.crop_rect = crop_rect
        last_frame_idx = max(0, self.total_frames - 1)
        end_val = last_frame_idx if trim_end_frame is None else min(trim_end_frame, last_frame_idx)

        self._start_slider.blockSignals(True)
        self._start_slider.setValue(min(trim_start_frame, last_frame_idx))
        self._start_slider.blockSignals(False)
        self._end_slider.blockSignals(True)
        self._end_slider.setValue(end_val)
        self._end_slider.blockSignals(False)

        self.trim_start_frame = self._start_slider.value()
        self.trim_end_frame = trim_end_frame
        self._update_labels()
        self._load_frame_at(self.trim_start_frame)
        self._redraw()

    def reset_crop(self) -> None:
        self.crop_rect = None
        self._redraw()
        self.rect_changed.emit()

    # ------------------------------------------------------------
    def _load_frame_at(self, frame_idx: int) -> None:
        if self.video_path is None:
            return
        video = cv.VideoCapture(self.video_path)
        video.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = video.read()
        video.release()
        if success:
            self.current_frame_bgr = frame
            self._redraw()

    def _on_start_slider(self, value: int) -> None:
        if value >= self._end_slider.value():
            value = max(0, self._end_slider.value() - 1)
            self._start_slider.blockSignals(True)
            self._start_slider.setValue(value)
            self._start_slider.blockSignals(False)
        self.trim_start_frame = value
        self._update_labels()
        self._load_frame_at(value)
        self.trim_changed.emit()

    def _on_end_slider(self, value: int) -> None:
        if value <= self._start_slider.value():
            value = min(self._start_slider.maximum(), self._start_slider.value() + 1)
            self._end_slider.blockSignals(True)
            self._end_slider.setValue(value)
            self._end_slider.blockSignals(False)
        last_frame_idx = max(0, self.total_frames - 1)
        self.trim_end_frame = None if value >= last_frame_idx else value
        self._update_labels()
        self._load_frame_at(value)
        self.trim_changed.emit()

    def _update_labels(self) -> None:
        self._start_label.setText(f"Start: {self.trim_start_frame}")
        end_display = "end" if self.trim_end_frame is None else str(self.trim_end_frame)
        self._end_label.setText(f"End: {end_display}")

    # ------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.current_frame_bgr is None or event.button() != Qt.LeftButton:
            return
        self._dragging = True
        self._drag_start = self._to_frame_coords(event.position().toPoint())
        self._drag_current = self._drag_start
        self._redraw()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._dragging:
            return
        self._drag_current = self._to_frame_coords(event.position().toPoint())
        self._redraw()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if not self._dragging:
            return
        self._dragging = False
        self._drag_current = self._to_frame_coords(event.position().toPoint())

        x1, y1 = self._drag_start
        x2, y2 = self._drag_current
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        self._drag_start = None
        self._drag_current = None

        # Ignore accidental clicks/tiny drags rather than saving a
        # near-zero-area crop rect.
        if (x2 - x1) < MIN_DRAG_PX or (y2 - y1) < MIN_DRAG_PX:
            self._redraw()
            return

        self.crop_rect = (x1, y1, x2, y2)
        self._redraw()
        self.rect_changed.emit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._redraw()

    def _to_frame_coords(self, widget_pos: QPoint) -> tuple[int, int]:
        label_pos = self._image_label.mapFrom(self, widget_pos)
        x = (label_pos.x() - self._pixmap_origin.x()) / self._display_scale
        y = (label_pos.y() - self._pixmap_origin.y()) / self._display_scale
        x = int(max(0, min(self.frame_w - 1, x))) if self.frame_w else 0
        y = int(max(0, min(self.frame_h - 1, y))) if self.frame_h else 0
        return x, y

    # ------------------------------------------------------------
    def _redraw(self) -> None:
        if self.current_frame_bgr is None:
            self._image_label.setText("No frame loaded")
            return

        display = self.current_frame_bgr
        h, w = display.shape[:2]
        rgb = cv.cvtColor(display, cv.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        target_w = max(self._image_label.width(), 1)
        target_h = max(self._image_label.height(), 1)
        scaled = pixmap.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._display_scale = scaled.width() / w if w else 1.0
        # AlignCenter means the scaled pixmap sits centered inside the
        # (usually larger) label - record that offset so drag
        # coordinates can be mapped back correctly.
        self._pixmap_origin = QPoint(
            (target_w - scaled.width()) // 2,
            (target_h - scaled.height()) // 2,
        )

        rect_to_draw = None
        if self._dragging and self._drag_start and self._drag_current:
            x1, y1 = self._drag_start
            x2, y2 = self._drag_current
            rect_to_draw = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        elif self.crop_rect is not None:
            cx1, cy1, cx2, cy2 = self.crop_rect
            rect_to_draw = (cx1, cy1, cx2 - cx1, cy2 - cy1)

        if rect_to_draw is not None:
            painter = QPainter(scaled)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(CROP_COLOR, 2))
            painter.setBrush(Qt.NoBrush)
            rx, ry, rw, rh = rect_to_draw
            painter.drawRect(QRect(
                int(rx * self._display_scale), int(ry * self._display_scale),
                int(rw * self._display_scale), int(rh * self._display_scale),
            ))
            painter.end()

        self._image_label.setPixmap(scaled)
