from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable
from ctypes import wintypes

from PySide6.QtCore import QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from dessinopia.core.models import CalibrationProfile, CenterGrid, Point, ScreenRect


class CalibrationDialog(QDialog):
    def __init__(self, profile: CalibrationProfile, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.profile = profile
        self.setWindowTitle("Calibration Heartopia")
        self.setMinimumWidth(640)
        root = QVBoxLayout(self)
        intro = QLabel(
            "La détection automatique a prérempli ces positions. Vous pouvez les corriger "
            "en plaçant le curseur sur une cible après le compte à rebours."
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        root.addWidget(intro)

        canvas_group = QGroupBox("Zone de dessin")
        canvas_form = QFormLayout(canvas_group)
        self.canvas_x = self._spin(profile.canvas.left)
        self.canvas_y = self._spin(profile.canvas.top)
        self.canvas_w = self._spin(profile.canvas.width, minimum=1)
        self.canvas_h = self._spin(profile.canvas.height, minimum=1)
        canvas_form.addRow(
            "Coin haut-gauche",
            self._point_row(self.canvas_x, self.canvas_y, self._pick_canvas_top_left),
        )
        canvas_form.addRow(
            "Dimensions", self._point_row(self.canvas_w, self.canvas_h, None, labels=("L", "H"))
        )
        bottom_right = QPushButton("Capturer le coin bas-droite")
        bottom_right.clicked.connect(self._pick_canvas_bottom_right)
        canvas_form.addRow("", bottom_right)
        root.addWidget(canvas_group)

        anchors = QGroupBox("Boutons et grilles de couleurs")
        form = QFormLayout(anchors)
        self.palette_x = self._spin(profile.palette_icon.x)
        self.palette_y = self._spin(profile.palette_icon.y)
        self.back_x = self._spin(profile.back_button.x)
        self.back_y = self._spin(profile.back_button.y)
        self.brush_x = self._spin(profile.small_brush.x)
        self.brush_y = self._spin(profile.small_brush.y)
        self.pencil_x = self._spin(profile.pencil_tool.x)
        self.pencil_y = self._spin(profile.pencil_tool.y)
        self.fill_x = self._spin(profile.fill_tool.x)
        self.fill_y = self._spin(profile.fill_tool.y)
        self.main_first_x = self._spin(profile.main_colors.first.x)
        self.main_first_y = self._spin(profile.main_colors.first.y)
        self.main_last_x = self._spin(profile.main_colors.last.x)
        self.main_last_y = self._spin(profile.main_colors.last.y)
        self.shade_first_x = self._spin(profile.shades.first.x)
        self.shade_first_y = self._spin(profile.shades.first.y)
        self.shade_last_x = self._spin(profile.shades.last.x)
        self.shade_last_y = self._spin(profile.shades.last.y)
        form.addRow(
            "Icône des nuances",
            self._point_row(
                self.palette_x, self.palette_y, lambda: self._pick(self.palette_x, self.palette_y)
            ),
        )
        form.addRow(
            "Bouton retour",
            self._point_row(self.back_x, self.back_y, lambda: self._pick(self.back_x, self.back_y)),
        )
        form.addRow(
            "Petit pinceau",
            self._point_row(
                self.brush_x,
                self.brush_y,
                lambda: self._pick(self.brush_x, self.brush_y),
            ),
        )
        form.addRow(
            "Outil crayon",
            self._point_row(
                self.pencil_x,
                self.pencil_y,
                lambda: self._pick(self.pencil_x, self.pencil_y),
            ),
        )
        form.addRow(
            "Pot de peinture",
            self._point_row(
                self.fill_x,
                self.fill_y,
                lambda: self._pick(self.fill_x, self.fill_y),
            ),
        )
        form.addRow(
            "Palette principale — 1re case",
            self._point_row(
                self.main_first_x,
                self.main_first_y,
                lambda: self._pick(self.main_first_x, self.main_first_y),
            ),
        )
        form.addRow(
            "Palette principale — dernière case",
            self._point_row(
                self.main_last_x,
                self.main_last_y,
                lambda: self._pick(self.main_last_x, self.main_last_y),
            ),
        )
        form.addRow(
            "Nuances — 1re case",
            self._point_row(
                self.shade_first_x,
                self.shade_first_y,
                lambda: self._pick(self.shade_first_x, self.shade_first_y),
            ),
        )
        form.addRow(
            "Nuances — dernière case",
            self._point_row(
                self.shade_last_x,
                self.shade_last_y,
                lambda: self._pick(self.shade_last_x, self.shade_last_y),
            ),
        )
        root.addWidget(anchors)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Enregistrer la calibration")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    @staticmethod
    def _spin(value: float, minimum: int = -10000) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, 20000)
        spin.setValue(round(value))
        spin.setMinimumWidth(95)
        return spin

    def _point_row(
        self,
        x_spin: QSpinBox,
        y_spin: QSpinBox,
        picker: Callable[[], None] | None,
        labels: tuple[str, str] = ("X", "Y"),
    ) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(labels[0]))
        layout.addWidget(x_spin)
        layout.addWidget(QLabel(labels[1]))
        layout.addWidget(y_spin)
        if picker:
            button = QPushButton("Capturer")
            button.clicked.connect(picker)
            layout.addWidget(button)
        layout.addStretch()
        return widget

    def _pick(self, x_spin: QSpinBox, y_spin: QSpinBox) -> None:
        self._capture_cursor(lambda x, y: (x_spin.setValue(x), y_spin.setValue(y)))

    def _pick_canvas_top_left(self) -> None:
        self._pick(self.canvas_x, self.canvas_y)

    def _pick_canvas_bottom_right(self) -> None:
        def apply(x: int, y: int) -> None:
            self.canvas_w.setValue(max(1, x - self.canvas_x.value()))
            self.canvas_h.setValue(max(1, y - self.canvas_y.value()))

        self._capture_cursor(apply)

    def _capture_cursor(self, callback: Callable[[int, int], None]) -> None:
        QMessageBox.information(
            self,
            "Capturer une position",
            "Après avoir fermé ce message, placez le curseur au centre de la cible. "
            "La position sera enregistrée dans 2 secondes.",
        )
        self.hide()

        def capture() -> None:
            if sys.platform == "win32":
                point = wintypes.POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
                callback(int(point.x), int(point.y))
            else:
                point = QCursor.pos()
                callback(point.x(), point.y())
            self.show()
            self.raise_()
            self.activateWindow()

        QTimer.singleShot(2000, capture)

    def result_profile(self) -> CalibrationProfile:
        pencil_tool = Point(self.pencil_x.value(), self.pencil_y.value())
        fill_tool = Point(self.fill_x.value(), self.fill_y.value())
        pencil_region = self.profile.pencil_region
        fill_region = self.profile.fill_region
        return CalibrationProfile(
            name="Calibration utilisateur",
            reference_width=self.profile.reference_width,
            reference_height=self.profile.reference_height,
            canvas=ScreenRect(
                self.canvas_x.value(),
                self.canvas_y.value(),
                self.canvas_w.value(),
                self.canvas_h.value(),
            ),
            palette_icon=Point(self.palette_x.value(), self.palette_y.value()),
            back_button=Point(self.back_x.value(), self.back_y.value()),
            small_brush=Point(self.brush_x.value(), self.brush_y.value()),
            pencil_tool=pencil_tool,
            fill_tool=fill_tool,
            pencil_tool_roi=ScreenRect(
                pencil_tool.x - pencil_region.width / 2,
                pencil_tool.y - pencil_region.height / 2,
                pencil_region.width,
                pencil_region.height,
            ),
            fill_tool_roi=ScreenRect(
                fill_tool.x - fill_region.width / 2,
                fill_tool.y - fill_region.height / 2,
                fill_region.width,
                fill_region.height,
            ),
            main_colors=CenterGrid(
                Point(self.main_first_x.value(), self.main_first_y.value()),
                Point(self.main_last_x.value(), self.main_last_y.value()),
                2,
                8,
            ),
            shades=CenterGrid(
                Point(self.shade_first_x.value(), self.shade_first_y.value()),
                Point(self.shade_last_x.value(), self.shade_last_y.value()),
                2,
                5,
            ),
            confidence=1.0,
            source="manual",
        )
