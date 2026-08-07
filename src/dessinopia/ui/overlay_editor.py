from __future__ import annotations

import ctypes
import sys
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Literal

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QGuiApplication,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QScreen,
)
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dessinopia.core.models import CalibrationProfile, CanvasSize, Point, ScreenRect
from dessinopia.core.screen_geometry import (
    map_point_between_spaces,
    map_rect_between_spaces,
    qt_geometry_to_physical,
)
from dessinopia.vision.capture import native_monitors

CanvasCorner = Literal["top_left", "top_right", "bottom_left", "bottom_right"]
ToolControlName = Literal["pencil_tool", "fill_tool"]
DragKind = Literal[
    "control",
    "main",
    "shade",
    "canvas_move",
    "canvas_resize",
    "tool_roi_move",
    "tool_roi_resize",
]


def move_canvas_rect(rect: ScreenRect, dx: float, dy: float) -> ScreenRect:
    """Translate a canvas without changing its dimensions."""

    return ScreenRect(rect.left + dx, rect.top + dy, rect.width, rect.height)


def resize_canvas_rect(
    rect: ScreenRect,
    corner: CanvasCorner,
    point: Point,
    *,
    minimum_size: float = 32.0,
) -> ScreenRect:
    """Resize one corner while keeping its opposite corner fixed."""

    if minimum_size <= 0:
        raise ValueError("La taille minimale doit être positive")
    if corner == "top_left":
        left = min(point.x, rect.right - minimum_size)
        top = min(point.y, rect.bottom - minimum_size)
        return ScreenRect(left, top, rect.right - left, rect.bottom - top)
    if corner == "top_right":
        right = max(point.x, rect.left + minimum_size)
        top = min(point.y, rect.bottom - minimum_size)
        return ScreenRect(rect.left, top, right - rect.left, rect.bottom - top)
    if corner == "bottom_left":
        left = min(point.x, rect.right - minimum_size)
        bottom = max(point.y, rect.top + minimum_size)
        return ScreenRect(left, rect.top, rect.right - left, bottom - rect.top)
    if corner == "bottom_right":
        right = max(point.x, rect.left + minimum_size)
        bottom = max(point.y, rect.top + minimum_size)
        return ScreenRect(rect.left, rect.top, right - rect.left, bottom - rect.top)
    raise ValueError(f"Coin de toile inconnu : {corner}")


def manual_profile(profile: CalibrationProfile, **changes: object) -> CalibrationProfile:
    """Apply an edit and mark the resulting calibration as user-controlled."""

    return replace(
        profile,
        name="Calibration utilisateur",
        confidence=1.0,
        source="manual",
        **changes,
    )


@dataclass(frozen=True, slots=True)
class GridSegment:
    start: Point
    end: Point
    strong: bool
    dense: bool


def canvas_grid_segments(rect: ScreenRect, size: CanvasSize) -> tuple[GridSegment, ...]:
    """Build every internal logical-pixel grid line for the active canvas size."""

    segments: list[GridSegment] = []
    for column in range(1, size.width):
        x = rect.left + column * rect.width / size.width
        segments.append(
            GridSegment(
                Point(x, rect.top),
                Point(x, rect.bottom),
                strong=column % 5 == 0,
                dense=size.width > 100,
            )
        )
    for row in range(1, size.height):
        y = rect.top + row * rect.height / size.height
        segments.append(
            GridSegment(
                Point(rect.left, y),
                Point(rect.right, y),
                strong=row % 5 == 0,
                dense=size.height > 100,
            )
        )
    return tuple(segments)


@dataclass(frozen=True, slots=True)
class DragTarget:
    kind: DragKind
    name: str = ""
    column: int = -1
    row: int = -1
    corner: CanvasCorner | None = None


class CalibrationOverlayEditor(QWidget):
    """Interactive full-screen editor for a physical-pixel calibration profile.

    The QWidget lives in Qt logical coordinates, while the profile stays in Win32
    physical desktop pixels. Every conversion therefore uses the selected monitor's
    complete native and Qt bounds, including negative origins and mixed DPI.
    """

    save = Signal(object)
    cancel = Signal()
    profile_changed = Signal(object)

    HANDLE_RADIUS = 11.0
    CANVAS_HANDLE_RADIUS = 13.0
    TOOL_ROI_HANDLE_RADIUS = 10.0
    TOOL_ROI_MINIMUM_SIZE = 16.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.profile: CalibrationProfile | None = None
        self.original_profile: CalibrationProfile | None = None
        self.canvas_size: CanvasSize | None = None
        self.target_screen: QScreen | None = None
        self._physical_bounds: ScreenRect | None = None
        self._logical_bounds: ScreenRect | None = None
        self._calibration_mode = False
        self._drag_target: DragTarget | None = None
        self._drag_origin: Point | None = None
        self._drag_profile: CalibrationProfile | None = None

        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._build_panel()
        self.set_calibration_mode(False)

    @property
    def calibration_mode(self) -> bool:
        return self._calibration_mode

    def _build_panel(self) -> None:
        self.panel = QFrame(self)
        self.panel.setObjectName("calibrationOverlayPanel")
        self.panel.setStyleSheet(
            "QFrame#calibrationOverlayPanel { background: rgba(13, 24, 22, 242); "
            "border: 1px solid #47766d; border-radius: 14px; }"
        )
        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(7)

        title = QLabel("Calibration visuelle")
        title.setStyleSheet("font-size: 16px; font-weight: 750; color: white;")
        layout.addWidget(title)
        instructions = QLabel(
            "Glissez les repères sur leur centre exact. Pour Crayon et Pot, glissez "
            "l'intérieur du cadre pour le déplacer ou ses coins pour le redimensionner. "
            "La toile se règle de la même façon."
        )
        instructions.setWordWrap(True)
        instructions.setMaximumWidth(430)
        instructions.setObjectName("muted")
        layout.addWidget(instructions)

        legend = QLabel(
            '<span style="color:#55e4cb">●</span> boutons fixes&nbsp;&nbsp; '
            '<span style="color:#68a9ff">●</span> P1–P16 palette principale<br>'
            '<span style="color:#ffc76b">●</span> N1–N10 nuances&nbsp;&nbsp; '
            '<span style="color:#ffad66">□</span> zones Crayon/Pot<br>'
            '<span style="color:#ff7ea7">■</span> poignées de toile et de détection'
        )
        legend.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(legend)
        self.layer_combo = QComboBox()
        self.layer_combo.addItem("Cibles éditables : toutes", "all")
        self.layer_combo.addItem("Cibles éditables : palette P", "main")
        self.layer_combo.addItem("Cibles éditables : nuances N", "shade")
        self.layer_combo.setToolTip(
            "Choisissez P ou N lorsque deux boutons des deux menus occupent le même emplacement."
        )
        layout.addWidget(self.layer_combo)
        self.coordinates_label = QLabel("")
        self.coordinates_label.setObjectName("statusPill")
        layout.addWidget(self.coordinates_label)

        buttons = QHBoxLayout()
        self.reset_button = QPushButton("Réinitialiser")
        self.reset_button.setToolTip("Revenir aux repères présents à l’ouverture")
        self.save_button = QPushButton("Enregistrer")
        self.save_button.setObjectName("primary")
        self.cancel_button = QPushButton("Annuler")
        self.cancel_button.setObjectName("danger")
        self.reset_button.clicked.connect(self._reset_profile)
        self.save_button.clicked.connect(self._emit_save)
        self.cancel_button.clicked.connect(self._emit_cancel)
        buttons.addWidget(self.reset_button)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        self.panel.adjustSize()

    def configure(
        self,
        profile: CalibrationProfile,
        canvas_size: CanvasSize | None = None,
        target_screen: QScreen | None = None,
    ) -> None:
        """Load a profile and place the editor on its physical target monitor."""

        self.profile = profile
        self.original_profile = profile
        self.canvas_size = canvas_size
        self.target_screen, self._physical_bounds = self._resolve_target_screen(
            profile, target_screen
        )
        if self.target_screen is None or self._physical_bounds is None:
            raise RuntimeError("Aucun écran ne correspond à la calibration")
        geometry = self.target_screen.geometry()
        self._logical_bounds = ScreenRect(
            float(geometry.x()),
            float(geometry.y()),
            float(geometry.width()),
            float(geometry.height()),
        )
        self.setGeometry(geometry)
        self.panel.move(18, 18)
        self._update_coordinates()
        self.update()

    def begin(
        self,
        profile: CalibrationProfile,
        canvas_size: CanvasSize | None = None,
        target_screen: QScreen | None = None,
    ) -> None:
        """Configure, activate mouse editing and show the overlay."""

        self.configure(profile, canvas_size, target_screen)
        self.set_calibration_mode(True)
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

    def set_calibration_mode(self, enabled: bool) -> None:
        self._calibration_mode = enabled
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not enabled)
        self.save_button.setEnabled(enabled)
        self.cancel_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.layer_combo.setEnabled(enabled)
        if not enabled:
            self._drag_target = None
            self.unsetCursor()

    def _resolve_target_screen(
        self,
        profile: CalibrationProfile,
        explicit_screen: QScreen | None,
    ) -> tuple[QScreen | None, ScreenRect | None]:
        monitors = {monitor.name.casefold(): monitor.rect for monitor in native_monitors()}
        candidates = [explicit_screen] if explicit_screen else list(QGuiApplication.screens())
        center = profile.canvas.center
        fallback: tuple[QScreen | None, ScreenRect | None] = (None, None)
        for screen in candidates:
            if screen is None:
                continue
            geometry = screen.geometry()
            ratio = screen.devicePixelRatio()
            logical = ScreenRect(
                geometry.x(),
                geometry.y(),
                geometry.width(),
                geometry.height(),
            )
            physical = monitors.get(
                screen.name().casefold(),
                qt_geometry_to_physical(logical, ratio),
            )
            if fallback == (None, None):
                fallback = (screen, physical)
            if (
                physical.left <= center.x < physical.right
                and physical.top <= center.y < physical.bottom
            ):
                return screen, physical
        return fallback

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().showEvent(event)
        if sys.platform == "win32":
            with suppress(AttributeError, OSError):
                ctypes.windll.user32.SetWindowDisplayAffinity(int(self.winId()), 0x00000011)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self.panel.adjustSize()
        self.panel.move(18, 18)

    def paintEvent(self, _event) -> None:  # type: ignore[no-untyped-def]
        if self.profile is None or not self._mapping_available():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(3, 9, 8, 70))
        self._paint_canvas(painter)
        self._paint_markers(painter)

    def _paint_canvas(self, painter: QPainter) -> None:
        assert self.profile is not None
        canvas = self._physical_rect_to_local(self.profile.canvas)
        canvas_qrect = QRectF(canvas.left, canvas.top, canvas.width, canvas.height)
        painter.fillRect(canvas_qrect, QColor(41, 214, 180, 30))
        if self.canvas_size is not None:
            for segment in canvas_grid_segments(
                ScreenRect(canvas.left, canvas.top, canvas.width, canvas.height),
                self.canvas_size,
            ):
                alpha = 175 if segment.strong else (55 if segment.dense else 95)
                painter.setPen(QPen(QColor(72, 218, 191, alpha), 2 if segment.strong else 1))
                painter.drawLine(
                    QPointF(segment.start.x, segment.start.y),
                    QPointF(segment.end.x, segment.end.y),
                )
        painter.setPen(QPen(QColor(83, 232, 204, 245), 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(canvas_qrect)
        painter.setPen(QColor(230, 255, 249))
        painter.drawText(
            canvas_qrect.adjusted(10, 8, -10, -8),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
            "TOILE — glisser pour déplacer",
        )
        painter.setPen(QPen(QColor(255, 126, 167), 2))
        painter.setBrush(QColor(255, 126, 167, 230))
        for point in self._canvas_corners_local().values():
            radius = self.CANVAS_HANDLE_RADIUS / 2
            painter.drawRect(QRectF(point.x - radius, point.y - radius, radius * 2, radius * 2))

    def _paint_markers(self, painter: QPainter) -> None:
        assert self.profile is not None
        controls = (
            ("palette_icon", self.profile.palette_icon, "Nuances"),
            ("back_button", self.profile.back_button, "Retour"),
            ("small_brush", self.profile.small_brush, "Pinceau fin"),
        )
        for _name, physical, label in controls:
            self._draw_marker(painter, self._physical_point_to_local(physical), label, "#55e4cb")

        self._draw_tool_region(
            painter,
            self._physical_rect_to_local(self.profile.pencil_region),
            "Crayon — zone de reconnaissance",
            "#55e4cb",
        )
        self._draw_tool_region(
            painter,
            self._physical_rect_to_local(self.profile.fill_region),
            "Pot — zone de reconnaissance",
            "#ffad66",
        )

        for row in range(self.profile.main_colors.rows):
            for column in range(self.profile.main_colors.columns):
                index = row * self.profile.main_colors.columns + column + 1
                point = self.profile.main_color_point(column, row)
                self._draw_marker(
                    painter,
                    self._physical_point_to_local(point),
                    f"P{index}",
                    "#68a9ff",
                )
        for row in range(self.profile.shades.rows):
            for column in range(self.profile.shades.columns):
                index = row * self.profile.shades.columns + column + 1
                point = self.profile.shade_point(column, row)
                self._draw_marker(
                    painter,
                    self._physical_point_to_local(point),
                    f"N{index}",
                    "#ffc76b",
                )

    def _draw_marker(
        self,
        painter: QPainter,
        point: Point,
        label: str,
        color: str,
    ) -> None:
        marker_rect = self._marker_rect(point)
        fill = QColor(color)
        translucent = QColor(fill)
        translucent.setAlpha(52)
        painter.setPen(QPen(fill, 2))
        painter.setBrush(translucent)
        painter.drawRoundedRect(marker_rect, 5, 5)
        painter.setPen(QPen(QColor(4, 14, 12, 240), 2))
        painter.setBrush(fill)
        painter.drawEllipse(QPointF(point.x, point.y), 5, 5)
        painter.setPen(QPen(QColor(245, 255, 252), 1))
        painter.drawText(QPointF(marker_rect.right() + 5, point.y + 5), label)

    def _draw_tool_region(
        self,
        painter: QPainter,
        region: ScreenRect,
        label: str,
        color: str,
    ) -> None:
        qrect = QRectF(region.left, region.top, region.width, region.height)
        fill = QColor(color)
        translucent = QColor(fill)
        translucent.setAlpha(42)
        painter.setPen(QPen(fill, 3))
        painter.setBrush(translucent)
        painter.drawRoundedRect(qrect, 8, 8)

        center = region.center
        painter.setPen(QPen(QColor(4, 14, 12, 240), 2))
        painter.setBrush(fill)
        painter.drawEllipse(QPointF(center.x, center.y), 5, 5)
        painter.setPen(QPen(QColor(245, 255, 252), 1))
        painter.drawText(QPointF(region.right + 6, center.y + 5), label)

        radius = self.TOOL_ROI_HANDLE_RADIUS / 2
        painter.setPen(QPen(QColor(245, 255, 252), 2))
        painter.setBrush(fill)
        for point in self._rect_corners(region).values():
            painter.drawRect(
                QRectF(
                    point.x - radius,
                    point.y - radius,
                    radius * 2,
                    radius * 2,
                )
            )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self._calibration_mode or self.profile is None:
            event.ignore()
            return
        if event.button() is not Qt.MouseButton.LeftButton:
            event.ignore()
            return
        target = self._hit_target(Point(event.position().x(), event.position().y()))
        if target is None:
            event.ignore()
            return
        self._drag_target = target
        self._drag_origin = self._local_point_to_physical(
            Point(event.position().x(), event.position().y())
        )
        self._drag_profile = self.profile
        self.grabMouse()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        local = Point(event.position().x(), event.position().y())
        if not self._calibration_mode or self.profile is None:
            event.ignore()
            return
        if self._drag_target is None:
            self._update_hover_cursor(self._hit_target(local))
            return
        assert self._drag_origin is not None and self._drag_profile is not None
        physical = self._local_point_to_physical(local)
        self.profile = self._apply_drag(
            self._drag_profile,
            self._drag_target,
            self._drag_origin,
            physical,
        )
        self._update_coordinates()
        self.profile_changed.emit(self.profile)
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_target is not None and event.button() is Qt.MouseButton.LeftButton:
            self.releaseMouse()
            self._drag_target = None
            self._drag_origin = None
            self._drag_profile = None
            self._update_hover_cursor(
                self._hit_target(Point(event.position().x(), event.position().y()))
            )
            event.accept()
            return
        event.ignore()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape and self._calibration_mode:
            self._emit_cancel()
            event.accept()
            return
        if (
            event.key() == Qt.Key.Key_S
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
            and self._calibration_mode
        ):
            self._emit_save()
            event.accept()
            return
        super().keyPressEvent(event)

    def _hit_target(self, local: Point) -> DragTarget | None:
        if self.profile is None:
            return None
        tool_regions = self._tool_regions_local()
        for name, region in tool_regions.items():
            for corner, marker in self._rect_corners(region).items():
                if (
                    self._distance_squared(local, marker)
                    <= self.TOOL_ROI_HANDLE_RADIUS**2
                ):
                    return DragTarget("tool_roi_resize", name=name, corner=corner)

        layer = self.layer_combo.currentData()
        marker_hits: list[tuple[float, DragTarget]] = []
        for target, marker in self._marker_targets_local():
            if target.kind == "main" and layer == "shade":
                continue
            if target.kind == "shade" and layer == "main":
                continue
            if target.kind == "control" and target.name in tool_regions:
                hit = (
                    self._distance_squared(local, marker)
                    <= self.HANDLE_RADIUS**2
                )
            else:
                hit = self._marker_rect(marker).adjusted(-4, -4, 4, 4).contains(
                    local.x,
                    local.y,
                )
            if hit:
                marker_hits.append((self._distance_squared(local, marker), target))
        if marker_hits:
            # Several controls legitimately occupy almost the same area in different
            # Heartopia menus. Select the centre closest to the pointer instead of the
            # first rectangle painted, otherwise P1 could move Retour, for example.
            return min(marker_hits, key=lambda item: item[0])[1]
        for name, region in tool_regions.items():
            if (
                region.left <= local.x <= region.right
                and region.top <= local.y <= region.bottom
            ):
                return DragTarget("tool_roi_move", name=name)
        for corner, marker in self._canvas_corners_local().items():
            if self._distance_squared(local, marker) <= self.CANVAS_HANDLE_RADIUS**2:
                return DragTarget("canvas_resize", corner=corner)
        canvas = self._physical_rect_to_local(self.profile.canvas)
        if canvas.left <= local.x <= canvas.right and canvas.top <= local.y <= canvas.bottom:
            return DragTarget("canvas_move")
        return None

    def _marker_targets_local(self) -> list[tuple[DragTarget, Point]]:
        assert self.profile is not None
        result = [
            (
                DragTarget("control", name="palette_icon"),
                self._physical_point_to_local(self.profile.palette_icon),
            ),
            (
                DragTarget("control", name="back_button"),
                self._physical_point_to_local(self.profile.back_button),
            ),
            (
                DragTarget("control", name="small_brush"),
                self._physical_point_to_local(self.profile.small_brush),
            ),
            (
                DragTarget("control", name="pencil_tool"),
                self._physical_point_to_local(self.profile.pencil_tool),
            ),
            (
                DragTarget("control", name="fill_tool"),
                self._physical_point_to_local(self.profile.fill_tool),
            ),
        ]
        for row in range(self.profile.main_colors.rows):
            for column in range(self.profile.main_colors.columns):
                result.append(
                    (
                        DragTarget("main", column=column, row=row),
                        self._physical_point_to_local(self.profile.main_color_point(column, row)),
                    )
                )
        for row in range(self.profile.shades.rows):
            for column in range(self.profile.shades.columns):
                result.append(
                    (
                        DragTarget("shade", column=column, row=row),
                        self._physical_point_to_local(self.profile.shade_point(column, row)),
                    )
                )
        return result

    def _canvas_corners_local(self) -> dict[CanvasCorner, Point]:
        assert self.profile is not None
        rect = self._physical_rect_to_local(self.profile.canvas)
        return self._rect_corners(rect)

    def _tool_regions_local(self) -> dict[ToolControlName, ScreenRect]:
        assert self.profile is not None
        return {
            "pencil_tool": self._physical_rect_to_local(self.profile.pencil_region),
            "fill_tool": self._physical_rect_to_local(self.profile.fill_region),
        }

    @staticmethod
    def _rect_corners(rect: ScreenRect) -> dict[CanvasCorner, Point]:
        return {
            "top_left": Point(rect.left, rect.top),
            "top_right": Point(rect.right, rect.top),
            "bottom_left": Point(rect.left, rect.bottom),
            "bottom_right": Point(rect.right, rect.bottom),
        }

    @staticmethod
    def _apply_drag(
        profile: CalibrationProfile,
        target: DragTarget,
        origin: Point,
        current: Point,
    ) -> CalibrationProfile:
        if target.kind == "control":
            if target.name not in {
                "palette_icon",
                "back_button",
                "small_brush",
                "pencil_tool",
                "fill_tool",
            }:
                raise ValueError(f"Contrôle inconnu : {target.name}")
            if target.name in {"pencil_tool", "fill_tool"}:
                region = (
                    profile.pencil_region
                    if target.name == "pencil_tool"
                    else profile.fill_region
                )
                roi_name = f"{target.name}_roi"
                centred = ScreenRect(
                    current.x - region.width / 2,
                    current.y - region.height / 2,
                    region.width,
                    region.height,
                )
                return manual_profile(
                    profile,
                    **{target.name: current, roi_name: centred},
                )
            return manual_profile(profile, **{target.name: current})
        if target.kind == "main":
            return manual_profile(profile.with_main_color_point(target.column, target.row, current))
        if target.kind == "shade":
            return manual_profile(profile.with_shade_point(target.column, target.row, current))
        if target.kind == "canvas_move":
            return manual_profile(
                profile,
                canvas=move_canvas_rect(
                    profile.canvas,
                    current.x - origin.x,
                    current.y - origin.y,
                ),
            )
        if target.kind == "canvas_resize" and target.corner is not None:
            return manual_profile(
                profile,
                canvas=resize_canvas_rect(profile.canvas, target.corner, current),
            )
        if target.kind in {"tool_roi_move", "tool_roi_resize"}:
            if target.name == "pencil_tool":
                region = profile.pencil_region
                roi_name = "pencil_tool_roi"
            elif target.name == "fill_tool":
                region = profile.fill_region
                roi_name = "fill_tool_roi"
            else:
                raise ValueError(f"Zone d'outil inconnue : {target.name}")
            if target.kind == "tool_roi_move":
                updated_region = region.translated(
                    current.x - origin.x,
                    current.y - origin.y,
                )
            elif target.corner is not None:
                updated_region = resize_canvas_rect(
                    region,
                    target.corner,
                    current,
                    minimum_size=CalibrationOverlayEditor.TOOL_ROI_MINIMUM_SIZE,
                )
            else:
                raise ValueError("Une poignée est requise pour redimensionner la zone")
            return manual_profile(
                profile,
                **{
                    target.name: updated_region.center,
                    roi_name: updated_region,
                },
            )
        raise ValueError(f"Cible de déplacement inconnue : {target}")

    def _update_hover_cursor(self, target: DragTarget | None) -> None:
        if target is None:
            self.unsetCursor()
        elif target.kind in {"canvas_resize", "tool_roi_resize"}:
            if target.corner in ("top_left", "bottom_right"):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            else:
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif target.kind in {"canvas_move", "tool_roi_move"}:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def _emit_save(self) -> None:
        if not self._calibration_mode or self.profile is None:
            return
        self.original_profile = self.profile
        self.save.emit(self.profile)

    def _reset_profile(self) -> None:
        if not self._calibration_mode or self.original_profile is None:
            return
        self.profile = self.original_profile
        self.profile_changed.emit(self.profile)
        self._update_coordinates()
        self.update()

    def _emit_cancel(self) -> None:
        if not self._calibration_mode:
            return
        if self.original_profile is not None:
            self.profile = self.original_profile
            self.profile_changed.emit(self.profile)
            self._update_coordinates()
            self.update()
        self.cancel.emit()

    def _update_coordinates(self) -> None:
        if self.profile is None:
            self.coordinates_label.clear()
            return
        canvas = self.profile.canvas
        grid = f" · grille {self.canvas_size.label}" if self.canvas_size is not None else ""
        pencil = self.profile.pencil_region
        fill = self.profile.fill_region
        self.coordinates_label.setText(
            f"Toile : {canvas.left:.0f}, {canvas.top:.0f} · "
            f"{canvas.width:.0f} × {canvas.height:.0f} px physiques{grid}\n"
            f"Reconnaissance : Crayon {pencil.width:.0f} × {pencil.height:.0f} · "
            f"Pot {fill.width:.0f} × {fill.height:.0f} px"
        )

    def _mapping_available(self) -> bool:
        return self._physical_bounds is not None and self._logical_bounds is not None

    def _physical_point_to_local(self, point: Point) -> Point:
        assert self._physical_bounds is not None and self._logical_bounds is not None
        mapped = map_point_between_spaces(point, self._physical_bounds, self._logical_bounds)
        return Point(mapped.x - self._logical_bounds.left, mapped.y - self._logical_bounds.top)

    def _local_point_to_physical(self, point: Point) -> Point:
        assert self._physical_bounds is not None and self._logical_bounds is not None
        global_logical = Point(
            point.x + self._logical_bounds.left,
            point.y + self._logical_bounds.top,
        )
        return map_point_between_spaces(
            global_logical,
            self._logical_bounds,
            self._physical_bounds,
        )

    def _physical_rect_to_local(self, rect: ScreenRect) -> ScreenRect:
        assert self._physical_bounds is not None and self._logical_bounds is not None
        mapped = map_rect_between_spaces(rect, self._physical_bounds, self._logical_bounds)
        return ScreenRect(
            mapped.left - self._logical_bounds.left,
            mapped.top - self._logical_bounds.top,
            mapped.width,
            mapped.height,
        )

    @staticmethod
    def _distance_squared(first: Point, second: Point) -> float:
        return (first.x - second.x) ** 2 + (first.y - second.y) ** 2

    @staticmethod
    def _marker_rect(point: Point) -> QRectF:
        return QRectF(point.x - 36, point.y - 29, 72, 58)
