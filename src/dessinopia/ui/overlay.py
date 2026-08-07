from __future__ import annotations

import ctypes
import sys
from contextlib import suppress

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dessinopia.core.models import CalibrationProfile, CanvasSize, ScreenRect
from dessinopia.core.screen_geometry import map_rect_between_spaces, qt_geometry_to_physical
from dessinopia.vision.capture import native_monitors


class GridOverlay(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.profile: CalibrationProfile | None = None
        self.canvas_size: CanvasSize | None = None
        self.target_screen = None
        self._target_physical_bounds: ScreenRect | None = None
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def configure(self, profile: CalibrationProfile, canvas_size: CanvasSize) -> None:
        self.profile = profile
        self.canvas_size = canvas_size
        center = profile.canvas.center
        self.target_screen = None
        self._target_physical_bounds = None
        physical_by_name = {monitor.name.casefold(): monitor.rect for monitor in native_monitors()}
        for candidate in QGuiApplication.screens():
            geometry = candidate.geometry()
            ratio = candidate.devicePixelRatio()
            logical_bounds = ScreenRect(
                geometry.x(),
                geometry.y(),
                geometry.width(),
                geometry.height(),
            )
            physical = physical_by_name.get(
                candidate.name().casefold(),
                qt_geometry_to_physical(logical_bounds, ratio),
            )
            if (
                physical.left <= center.x < physical.right
                and physical.top <= center.y < physical.bottom
            ):
                self.target_screen = candidate
                self._target_physical_bounds = physical
                break
        if self.target_screen is None:
            self.target_screen = self.screen()
            if self.target_screen:
                geometry = self.target_screen.geometry()
                ratio = self.target_screen.devicePixelRatio()
                logical_bounds = ScreenRect(
                    geometry.x(),
                    geometry.y(),
                    geometry.width(),
                    geometry.height(),
                )
                self._target_physical_bounds = physical_by_name.get(
                    self.target_screen.name().casefold(),
                    qt_geometry_to_physical(logical_bounds, ratio),
                )
        # A top-level QWidget spanning screens with different DPRs inherits only one
        # screen's DPR.  The calibrated canvas belongs to a single monitor, so keep the
        # transparent overlay entirely on that monitor.
        if self.target_screen:
            self.setGeometry(self.target_screen.geometry())
        self.update()

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().showEvent(event)
        if sys.platform == "win32":
            # Empêche l'overlay d'apparaître dans les captures utilisées pour la détection.
            with suppress(AttributeError, OSError):
                ctypes.windll.user32.SetWindowDisplayAffinity(int(self.winId()), 0x00000011)

    def paintEvent(self, _event) -> None:  # type: ignore[no-untyped-def]
        if not self.profile or not self.canvas_size:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        origin = self.geometry().topLeft()
        canvas = self.profile.canvas
        if not self.target_screen or not self._target_physical_bounds:
            return
        screen_geometry = self.target_screen.geometry()
        logical_bounds = ScreenRect(
            screen_geometry.x(),
            screen_geometry.y(),
            screen_geometry.width(),
            screen_geometry.height(),
        )
        logical_canvas = map_rect_between_spaces(
            canvas,
            self._target_physical_bounds,
            logical_bounds,
        )
        left = logical_canvas.left - origin.x()
        top = logical_canvas.top - origin.y()
        width = logical_canvas.width
        height = logical_canvas.height
        canvas_rect = QRectF(left, top, width, height)
        painter.fillRect(canvas_rect, QColor(41, 214, 180, 13))
        painter.setPen(QPen(QColor(83, 232, 204, 230), 3))
        painter.drawRect(canvas_rect)

        columns, rows = self.canvas_size.width, self.canvas_size.height
        for column in range(1, columns):
            strong = column % 5 == 0
            alpha = 185 if strong else (95 if columns <= 100 else 55)
            painter.setPen(QPen(QColor(72, 218, 191, alpha), 2 if strong else 1))
            x = left + column * width / columns
            painter.drawLine(QPointF(x, top), QPointF(x, top + height))
        for row in range(1, rows):
            strong = (rows - row) % 5 == 0
            alpha = 185 if strong else (95 if rows <= 100 else 55)
            painter.setPen(QPen(QColor(72, 218, 191, alpha), 2 if strong else 1))
            y = top + row * height / rows
            painter.drawLine(QPointF(left, y), QPointF(left + width, y))


class OverlayToolbar(QFrame):
    start_requested = Signal()
    test_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    stop_requested = Signal()
    calibrate_requested = Signal()
    repair_requested = Signal()
    close_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._drag_offset = None
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setObjectName("overlayToolbar")
        self.setStyleSheet(
            "QFrame#overlayToolbar { background: rgba(13, 24, 22, 235); "
            "border: 1px solid #47766d; border-radius: 14px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(7)
        header = QHBoxLayout()
        title = QLabel("Dessinopia")
        title.setStyleSheet("font-size: 15px; font-weight: 750; color: white;")
        self.state_label = QLabel("Prêt")
        self.state_label.setObjectName("statusPill")
        close = QPushButton("×")
        close.setFixedSize(30, 30)
        close.setStyleSheet("padding: 0; font-size: 18px; font-weight: 700;")
        close.clicked.connect(self.close_requested)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.state_label)
        header.addWidget(close)
        layout.addLayout(header)

        self.status_label = QLabel("Vérifiez les repères puis lancez le dessin")
        self.status_label.setObjectName("muted")
        layout.addWidget(self.status_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Démarrer")
        self.start_button.setObjectName("primary")
        self.pause_button = QPushButton("Pause")
        self.resume_button = QPushButton("Reprendre")
        self.stop_button = QPushButton("Arrêter")
        self.stop_button.setObjectName("danger")
        self.test_button = QPushButton("Test 5 points")
        self.test_button.setToolTip(
            "Peindre temporairement les quatre coins et le centre avant le dessin"
        )
        self.calibration_button = QPushButton("Repères détectés")
        self.calibration_button.setToolTip(
            "Afficher tous les boutons détectés et les ajuster directement sur le jeu"
        )
        for button in (
            self.start_button,
            self.pause_button,
            self.resume_button,
            self.stop_button,
        ):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        setup_buttons = QHBoxLayout()
        setup_buttons.addWidget(self.test_button)
        setup_buttons.addWidget(self.calibration_button)
        layout.addLayout(setup_buttons)
        self.repair_button = QPushButton("Auditer et réparer les trous")
        self.repair_button.setToolTip(
            "Rechercher les cellules encore ivoire et les repeindre en mode prudent"
        )
        layout.addWidget(self.repair_button)
        self.start_button.clicked.connect(self.start_requested)
        self.test_button.clicked.connect(self.test_requested)
        self.pause_button.clicked.connect(self.pause_requested)
        self.resume_button.clicked.connect(self.resume_requested)
        self.stop_button.clicked.connect(self.stop_requested)
        self.calibration_button.clicked.connect(self.calibrate_requested)
        self.repair_button.clicked.connect(self.repair_requested)
        emergency = QLabel("F12 : arrêt d’urgence")
        emergency.setAlignment(Qt.AlignmentFlag.AlignCenter)
        emergency.setStyleSheet("color: #ffb7c9; font-weight: 700;")
        layout.addWidget(emergency)
        self.set_running(False)
        self.adjustSize()

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().showEvent(event)
        if sys.platform == "win32":
            # The toolbar is interactive but must not contaminate vision captures if the
            # user has dragged it over the game UI.
            with suppress(AttributeError, OSError):
                ctypes.windll.user32.SetWindowDisplayAffinity(int(self.winId()), 0x00000011)

    def set_running(self, running: bool, paused: bool = False, stopping: bool = False) -> None:
        active = running or stopping
        self.start_button.setEnabled(not active)
        self.pause_button.setEnabled(running and not paused and not stopping)
        self.resume_button.setEnabled(running and paused and not stopping)
        self.stop_button.setEnabled(running and not stopping)
        self.test_button.setEnabled(not active)
        self.calibration_button.setEnabled(not active)
        self.repair_button.setEnabled(not active)

    def set_progress(self, done: int, total: int) -> None:
        self.progress.setValue(round(done * 100 / total) if total else 0)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() is Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:
        self._drag_offset = None
