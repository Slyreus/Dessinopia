from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget


def rgba_to_qimage(rgba: np.ndarray) -> QImage:
    contiguous = np.ascontiguousarray(rgba, dtype=np.uint8)
    height, width = contiguous.shape[:2]
    return QImage(
        contiguous.data,
        width,
        height,
        int(contiguous.strides[0]),
        QImage.Format.Format_RGBA8888,
    ).copy()


class ImageDropLabel(QLabel):
    file_dropped = Signal(str)

    def __init__(self) -> None:
        super().__init__("Glissez une image ici\nou cliquez sur Importer")
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(150)
        self.setStyleSheet(
            "border: 1px dashed #4c6c65; border-radius: 12px; color: #8fa8a2; background: #111b19;"
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls() and self._accepted(event.mimeData().urls()[0].toLocalFile()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        path = event.mimeData().urls()[0].toLocalFile()
        if self._accepted(path):
            self.file_dropped.emit(path)
            event.acceptProposedAction()

    @staticmethod
    def _accepted(path: str) -> bool:
        return Path(path).suffix.casefold() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
            ".tif",
            ".tiff",
        }


class PixelPreview(QWidget):
    def __init__(self, *, pixelated: bool = True) -> None:
        super().__init__()
        self._image = QImage()
        self._pixelated = pixelated
        self._show_grid = False
        self.setMinimumSize(420, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, image: QImage) -> None:
        self._image = image
        self.update()

    def set_grid_visible(self, visible: bool) -> None:
        self._show_grid = visible
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        tile = 14
        for y in range(0, self.height(), tile):
            for x in range(0, self.width(), tile):
                color = QColor("#17211f") if (x // tile + y // tile) % 2 else QColor("#1d2926")
                painter.fillRect(x, y, tile, tile, color)
        if self._image.isNull():
            painter.setPen(QColor("#6f8580"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Aucun aperçu")
            return

        margin = 18
        area = self.rect().adjusted(margin, margin, -margin, -margin)
        size = self._image.size()
        size.scale(area.size(), Qt.AspectRatioMode.KeepAspectRatio)
        target = area
        target.setSize(size)
        target.moveCenter(area.center())
        mode = (
            Qt.TransformationMode.FastTransformation
            if self._pixelated
            else Qt.TransformationMode.SmoothTransformation
        )
        pixmap = QPixmap.fromImage(self._image).scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            mode,
        )
        painter.drawPixmap(target.topLeft(), pixmap)
        painter.setPen(QPen(QColor("#3d5650"), 1))
        painter.drawRect(target.adjusted(0, 0, -1, -1))

        if self._show_grid and self._pixelated and self._image.width() <= 100:
            cell_w = target.width() / self._image.width()
            cell_h = target.height() / self._image.height()
            for column in range(1, self._image.width()):
                strong = column % 5 == 0
                painter.setPen(
                    QPen(QColor(72, 121, 111, 170 if strong else 80), 2 if strong else 1)
                )
                x = target.left() + column * cell_w
                painter.drawLine(int(x), target.top(), int(x), target.bottom())
            for row in range(1, self._image.height()):
                strong = (self._image.height() - row) % 5 == 0
                painter.setPen(
                    QPen(QColor(72, 121, 111, 170 if strong else 80), 2 if strong else 1)
                )
                y = target.top() + row * cell_h
                painter.drawLine(target.left(), int(y), target.right(), int(y))
