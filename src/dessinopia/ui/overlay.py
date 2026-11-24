from __future__ import annotations

import ctypes
import sys
from contextlib import suppress

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from dessinopia.core.models import CalibrationProfile, CanvasSize, ScreenRect
from dessinopia.core.screen_geometry import map_rect_between_spaces, qt_geometry_to_physical
from dessinopia.vision.capture import native_monitors

WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011


def _set_capture_excluded(window_id: int | None, excluded: bool) -> None:
    """Control capture visibility without changing the on-screen overlay.

    Overlays stay shareable by default.  Internal computer-vision screenshots can
    exclude them for the few milliseconds needed to read Heartopia, then restore
    normal screen-sharing visibility immediately afterwards.
    """

    if sys.platform != "win32" or window_id is None:
        return
    affinity = WDA_EXCLUDEFROMCAPTURE if excluded else WDA_NONE
    with suppress(AttributeError, OSError):
        ctypes.windll.user32.SetWindowDisplayAffinity(window_id, affinity)


class GridOverlay(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.profile: CalibrationProfile | None = None
        self._capture_hwnd: int | None = None
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
            if self.screen() != self.target_screen:
                self.setScreen(self.target_screen)
            self.setGeometry(self.target_screen.geometry())
        self.update()

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().showEvent(event)
        self._capture_hwnd = int(self.winId())
        _set_capture_excluded(self._capture_hwnd, False)

    def set_capture_excluded(self, excluded: bool) -> None:
        _set_capture_excluded(self._capture_hwnd, excluded)

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
        painter.setClipRect(canvas_rect)
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
    start_without_test_requested = Signal()
    test_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    reset_requested = Signal()
    stop_requested = Signal()
    calibrate_requested = Signal()
    redetect_requested = Signal()
    repair_requested = Signal()
    close_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._drag_offset = None
        self._capture_hwnd: int | None = None
        self._active = False
        self._resume_available = False
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
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(5)
        header = QHBoxLayout()
        header.setSpacing(8)
        title = QLabel("Dessinopia")
        title.setStyleSheet("font-size: 14px; font-weight: 750; color: white;")
        self.state_label = QLabel("Prêt")
        self.state_label.setObjectName("statusPill")
        self.status_label = QLabel("Vérifiez les repères puis lancez le dessin")
        self.status_label.setObjectName("muted")
        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFixedWidth(150)
        self.progress.setFixedHeight(18)
        emergency = QLabel("F8 pause/reprise · F12 urgence")
        emergency.setStyleSheet("color: #ffb7c9; font-weight: 700;")
        close = QPushButton("×")
        close.setFixedSize(28, 28)
        close.setStyleSheet("padding: 0; font-size: 18px; font-weight: 700;")
        close.clicked.connect(self.close_requested)
        header.addWidget(title)
        header.addWidget(self.state_label)
        header.addWidget(self.status_label, 1)
        header.addWidget(self.progress)
        header.addWidget(emergency)
        header.addWidget(close)
        layout.addLayout(header)

        self.button_layout = QGridLayout()
        self.button_layout.setSpacing(6)
        self.start_button = QPushButton("Démarrer")
        self.start_button.setObjectName("primary")
        self.start_without_test_button = QPushButton("Démarrer sans test")
        self.start_without_test_button.setToolTip(
            "Utiliser les repères affichés sans peindre les cinq points de calibration."
        )
        self.pause_button = QPushButton("Pause")
        self.resume_button = QPushButton("Reprendre")
        self.tempo_combo = QComboBox()
        self.tempo_combo.setAccessibleName("Tempo")
        self.tempo_combo.setToolTip("Modifiable pendant la pause. Appliqué à la reprise.")
        self.tempo_control = QWidget()
        tempo_layout = QHBoxLayout(self.tempo_control)
        tempo_layout.setContentsMargins(0, 0, 0, 0)
        tempo_layout.addWidget(QLabel("Tempo"))
        tempo_layout.addWidget(self.tempo_combo)
        self.stop_button = QPushButton("Arrêter")
        self.stop_button.setObjectName("danger")
        self.test_button = QPushButton("Test 5 points")
        self.test_button.setToolTip(
            "Peindre temporairement les quatre coins et le centre avant le dessin"
        )
        self.calibration_button = QPushButton("Repères")
        self.calibration_button.setToolTip(
            "Afficher tous les boutons détectés et les ajuster directement sur le jeu"
        )
        self.redetect_button = QPushButton("Redétecter la toile")
        self.redetect_button.setToolTip("Recaler la grille et les commandes visibles dans Heartopia")
        self.repair_button = QPushButton("Réparer les trous")
        self.repair_button.setToolTip(
            "Rechercher les cellules encore ivoire et les repeindre en mode prudent"
        )
        self.reset_button = QPushButton("Réinitialiser")
        self.reset_button.setToolTip(
            "Abandonner le checkpoint et recommencer le plan depuis zéro"
        )
        self.action_widgets = (
            self.start_button,
            self.start_without_test_button,
            self.pause_button,
            self.tempo_control,
            self.resume_button,
            self.stop_button,
            self.test_button,
            self.redetect_button,
            self.calibration_button,
            self.repair_button,
            self.reset_button,
        )
        for column, widget in enumerate(self.action_widgets):
            widget.setMinimumHeight(28)
            self.button_layout.addWidget(widget, 0, column)
        layout.addLayout(self.button_layout)
        self.start_button.clicked.connect(self.start_requested)
        self.start_without_test_button.clicked.connect(self.start_without_test_requested)
        self.test_button.clicked.connect(self.test_requested)
        self.pause_button.clicked.connect(self.pause_requested)
        self.resume_button.clicked.connect(self.resume_requested)
        self.reset_button.clicked.connect(self.reset_requested)
        self.stop_button.clicked.connect(self.stop_requested)
        self.calibration_button.clicked.connect(self.calibrate_requested)
        self.redetect_button.clicked.connect(self.redetect_requested)
        self.repair_button.clicked.connect(self.repair_requested)
        self.set_running(False)
        self.adjustSize()

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().showEvent(event)
        self._capture_hwnd = int(self.winId())
        _set_capture_excluded(self._capture_hwnd, False)

    def set_capture_excluded(self, excluded: bool) -> None:
        _set_capture_excluded(self._capture_hwnd, excluded)

    def place_at_top_center(self, screen_geometry) -> None:  # type: ignore[no-untyped-def]
        """Use a wide, shallow strip that does not cover the canvas on small screens."""

        target_width = max(320, min(1220, screen_geometry.width() - 24))
        widths = [widget.sizeHint().width() for widget in self.action_widgets]
        margins = self.layout().contentsMargins()
        available = target_width - margins.left() - margins.right()
        for rows in range(1, len(widths) + 1):
            columns = (len(widths) + rows - 1) // rows
            needed = sum(max(widths[column::columns]) for column in range(columns))
            if needed + (columns - 1) * self.button_layout.spacing() <= available:
                break
        for index, widget in enumerate(self.action_widgets):
            self.button_layout.removeWidget(widget)
            self.button_layout.addWidget(widget, *divmod(index, columns))
        self.setFixedWidth(target_width)
        self.adjustSize()
        x = screen_geometry.left() + max(0, (screen_geometry.width() - self.width()) // 2)
        self.move(x, screen_geometry.top() + 8)

    def set_running(
        self,
        running: bool,
        *,
        paused: bool = False,
        pausing: bool = False,
        stopping: bool = False,
        tempo_editable: bool = False,
    ) -> None:
        active = running or stopping
        self._active = active
        self.start_button.setEnabled(not active)
        self.start_without_test_button.setEnabled(not active)
        self.pause_button.setEnabled(running and not (paused or pausing or stopping))
        self.resume_button.setEnabled(running and paused and not stopping)
        self.tempo_control.setEnabled(tempo_editable)
        self.stop_button.setEnabled(running and not stopping)
        self.test_button.setEnabled(not active)
        self.calibration_button.setEnabled(not active)
        self.redetect_button.setEnabled(not active)
        self.repair_button.setEnabled(not active)
        self.reset_button.setEnabled(not active and self._resume_available)

    def set_resume_available(self, available: bool) -> None:
        self._resume_available = available
        self.reset_button.setEnabled(available and not self._active)

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
