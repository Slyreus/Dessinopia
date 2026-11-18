from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any


class AspectRatio(StrEnum):
    LANDSCAPE_16_9 = "16:9"
    LANDSCAPE_4_3 = "4:3"
    SQUARE = "1:1"
    PORTRAIT_3_4 = "3:4"
    PORTRAIT_9_16 = "9:16"


class Precision(StrEnum):
    SMALL = "Petit"
    MEDIUM = "Moyen"
    LARGE = "Grand"
    EXTRA_LARGE = "Très grand"


class FitMode(StrEnum):
    CONTAIN = "Conserver l'image (marges)"
    COVER = "Remplir (recadrage centré)"
    STRETCH = "Étirer jusqu'au format"
    NATIVE = "Taille réelle (1:1)"


class ResizeMode(StrEnum):
    PIXEL = "Pixels nets"
    SMOOTH = "Lissé (Lanczos)"


class DitherMode(StrEnum):
    NONE = "Aucun"
    FLOYD_STEINBERG = "Floyd–Steinberg"
    ORDERED = "Tramage ordonné"


DEFAULT_MAX_COLORS = 16
DEFAULT_DITHER_STRENGTH = 35


@dataclass(frozen=True, slots=True)
class CanvasSize:
    width: int
    height: int

    @property
    def pixel_count(self) -> int:
        return self.width * self.height

    @property
    def label(self) -> str:
        return f"{self.width} × {self.height}"


GRID_PRESETS: dict[AspectRatio, tuple[CanvasSize, ...]] = {
    AspectRatio.LANDSCAPE_16_9: (
        CanvasSize(30, 18),
        CanvasSize(50, 28),
        CanvasSize(100, 56),
        CanvasSize(150, 84),
    ),
    AspectRatio.LANDSCAPE_4_3: (
        CanvasSize(30, 24),
        CanvasSize(50, 38),
        CanvasSize(100, 76),
        CanvasSize(150, 114),
    ),
    AspectRatio.SQUARE: (
        CanvasSize(30, 30),
        CanvasSize(50, 50),
        CanvasSize(100, 100),
        CanvasSize(150, 150),
    ),
    AspectRatio.PORTRAIT_3_4: (
        CanvasSize(24, 30),
        CanvasSize(38, 50),
        CanvasSize(76, 100),
        CanvasSize(114, 150),
    ),
    AspectRatio.PORTRAIT_9_16: (
        CanvasSize(18, 30),
        CanvasSize(28, 50),
        CanvasSize(56, 100),
        CanvasSize(84, 150),
    ),
}

PRECISIONS: tuple[Precision, ...] = (
    Precision.SMALL,
    Precision.MEDIUM,
    Precision.LARGE,
    Precision.EXTRA_LARGE,
)


def canvas_size(ratio: AspectRatio, precision: Precision) -> CanvasSize:
    return GRID_PRESETS[ratio][PRECISIONS.index(precision)]


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def scaled(self, sx: float, sy: float) -> Point:
        return Point(self.x * sx, self.y * sy)


@dataclass(frozen=True, slots=True)
class ScreenRect:
    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    @property
    def center(self) -> Point:
        return Point(self.left + self.width / 2, self.top + self.height / 2)

    def cell_center(self, column: int, row: int, columns: int, rows: int) -> Point:
        if not (0 <= column < columns and 0 <= row < rows):
            raise ValueError("Cellule hors de la grille")
        return Point(
            self.left + (column + 0.5) * self.width / columns,
            self.top + (row + 0.5) * self.height / rows,
        )

    def scaled(self, sx: float, sy: float) -> ScreenRect:
        return ScreenRect(self.left * sx, self.top * sy, self.width * sx, self.height * sy)

    def translated(self, dx: float, dy: float) -> ScreenRect:
        return ScreenRect(self.left + dx, self.top + dy, self.width, self.height)


@dataclass(frozen=True, slots=True)
class CenterGrid:
    """Grille définie par le centre de sa première et de sa dernière cellule."""

    first: Point
    last: Point
    columns: int
    rows: int

    def point(self, column: int, row: int) -> Point:
        if not (0 <= column < self.columns and 0 <= row < self.rows):
            raise ValueError("Cellule hors de la grille")
        dx = 0.0 if self.columns == 1 else (self.last.x - self.first.x) / (self.columns - 1)
        dy = 0.0 if self.rows == 1 else (self.last.y - self.first.y) / (self.rows - 1)
        return Point(self.first.x + column * dx, self.first.y + row * dy)

    def scaled(self, sx: float, sy: float) -> CenterGrid:
        return CenterGrid(
            first=self.first.scaled(sx, sy),
            last=self.last.scaled(sx, sy),
            columns=self.columns,
            rows=self.rows,
        )


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    name: str
    reference_width: int
    reference_height: int
    canvas: ScreenRect
    palette_icon: Point
    back_button: Point
    small_brush: Point
    pencil_tool: Point
    fill_tool: Point
    main_colors: CenterGrid
    shades: CenterGrid
    confidence: float = 0.0
    source: str = "manual"
    main_color_points: tuple[Point, ...] = ()
    shade_color_points: tuple[Point, ...] = ()
    pencil_tool_roi: ScreenRect | None = None
    fill_tool_roi: ScreenRect | None = None
    reference_left: float = 0.0
    reference_top: float = 0.0

    def __post_init__(self) -> None:
        """Materialize tool ROIs and keep them centred on their click targets.

        ``pencil_tool`` and ``fill_tool`` predate configurable recognition zones.
        Keeping those points authoritative preserves ``dataclasses.replace`` and
        every legacy caller that only moves a button centre. A resized ROI passes
        its new centre together with its dimensions and is therefore preserved.
        """

        for field_name, point, configured in (
            ("pencil_tool_roi", self.pencil_tool, self.pencil_tool_roi),
            ("fill_tool_roi", self.fill_tool, self.fill_tool_roi),
        ):
            region = configured or default_tool_roi(
                point,
                self.reference_width,
                self.reference_height,
            )
            if region.width <= 0 or region.height <= 0:
                raise ValueError(
                    "Une zone de reconnaissance d'outil doit avoir une taille positive"
                )
            centred = ScreenRect(
                point.x - region.width / 2,
                point.y - region.height / 2,
                region.width,
                region.height,
            )
            object.__setattr__(self, field_name, centred)

    @property
    def pencil_region(self) -> ScreenRect:
        assert self.pencil_tool_roi is not None
        return self.pencil_tool_roi

    @property
    def fill_region(self) -> ScreenRect:
        assert self.fill_tool_roi is not None
        return self.fill_tool_roi

    @property
    def reference_bounds(self) -> ScreenRect:
        """Physical bounds in which the saved calibration was created."""

        return ScreenRect(
            self.reference_left,
            self.reference_top,
            float(self.reference_width),
            float(self.reference_height),
        )

    def main_color_point(self, column: int, row: int) -> Point:
        """Return one main-palette centre, including an individual override."""

        self.main_colors.point(column, row)  # validates the coordinates
        index = row * self.main_colors.columns + column
        if len(self.main_color_points) == self.main_colors.columns * self.main_colors.rows:
            return self.main_color_points[index]
        return self.main_colors.point(column, row)

    def shade_point(self, column: int, row: int) -> Point:
        """Return one shade centre, including an individual override."""

        self.shades.point(column, row)  # validates the coordinates
        index = row * self.shades.columns + column
        if len(self.shade_color_points) == self.shades.columns * self.shades.rows:
            return self.shade_color_points[index]
        return self.shades.point(column, row)

    def with_main_color_point(self, column: int, row: int, point: Point) -> CalibrationProfile:
        """Return a profile with exactly one main-palette centre overridden."""

        self.main_colors.point(column, row)  # validates the coordinates
        points = [
            self.main_color_point(item_column, item_row)
            for item_row in range(self.main_colors.rows)
            for item_column in range(self.main_colors.columns)
        ]
        points[row * self.main_colors.columns + column] = point
        return replace(self, main_color_points=tuple(points))

    def with_shade_point(self, column: int, row: int, point: Point) -> CalibrationProfile:
        """Return a profile with exactly one shade centre overridden."""

        self.shades.point(column, row)  # validates the coordinates
        points = [
            self.shade_point(item_column, item_row)
            for item_row in range(self.shades.rows)
            for item_column in range(self.shades.columns)
        ]
        points[row * self.shades.columns + column] = point
        return replace(self, shade_color_points=tuple(points))

    def scaled_to(self, width: int, height: int) -> CalibrationProfile:
        return self.remapped_to(
            ScreenRect(
                self.reference_left,
                self.reference_top,
                float(width),
                float(height),
            )
        )

    def remapped_to(self, bounds: ScreenRect) -> CalibrationProfile:
        """Scale and translate every marker into new physical display bounds.

        The mapping is relative to the complete capture on which the user calibrated
        Heartopia. It therefore preserves manual offsets on another resolution and
        also handles monitors whose virtual-desktop origin is not ``(0, 0)``.
        """

        source = self.reference_bounds
        if source.width <= 0 or source.height <= 0:
            raise ValueError("La référence de calibration doit avoir une taille positive")
        if bounds.width <= 0 or bounds.height <= 0:
            raise ValueError("La nouvelle résolution doit avoir une taille positive")
        sx = bounds.width / source.width
        sy = bounds.height / source.height

        def map_point(point: Point) -> Point:
            return Point(
                bounds.left + (point.x - source.left) * sx,
                bounds.top + (point.y - source.top) * sy,
            )

        def map_rect(rect: ScreenRect) -> ScreenRect:
            return ScreenRect(
                bounds.left + (rect.left - source.left) * sx,
                bounds.top + (rect.top - source.top) * sy,
                rect.width * sx,
                rect.height * sy,
            )

        return CalibrationProfile(
            name=self.name,
            reference_width=round(bounds.width),
            reference_height=round(bounds.height),
            canvas=map_rect(self.canvas),
            palette_icon=map_point(self.palette_icon),
            back_button=map_point(self.back_button),
            small_brush=map_point(self.small_brush),
            pencil_tool=map_point(self.pencil_tool),
            fill_tool=map_point(self.fill_tool),
            main_colors=CenterGrid(
                map_point(self.main_colors.first),
                map_point(self.main_colors.last),
                self.main_colors.columns,
                self.main_colors.rows,
            ),
            shades=CenterGrid(
                map_point(self.shades.first),
                map_point(self.shades.last),
                self.shades.columns,
                self.shades.rows,
            ),
            confidence=self.confidence,
            source=self.source,
            main_color_points=tuple(map_point(point) for point in self.main_color_points),
            shade_color_points=tuple(map_point(point) for point in self.shade_color_points),
            pencil_tool_roi=map_rect(self.pencil_region),
            fill_tool_roi=map_rect(self.fill_region),
            reference_left=bounds.left,
            reference_top=bounds.top,
        )

    def translated(self, dx: float, dy: float) -> CalibrationProfile:
        def move_point(point: Point) -> Point:
            return Point(point.x + dx, point.y + dy)

        return CalibrationProfile(
            name=self.name,
            reference_width=self.reference_width,
            reference_height=self.reference_height,
            canvas=self.canvas.translated(dx, dy),
            palette_icon=move_point(self.palette_icon),
            back_button=move_point(self.back_button),
            small_brush=move_point(self.small_brush),
            pencil_tool=move_point(self.pencil_tool),
            fill_tool=move_point(self.fill_tool),
            main_colors=CenterGrid(
                move_point(self.main_colors.first),
                move_point(self.main_colors.last),
                self.main_colors.columns,
                self.main_colors.rows,
            ),
            shades=CenterGrid(
                move_point(self.shades.first),
                move_point(self.shades.last),
                self.shades.columns,
                self.shades.rows,
            ),
            confidence=self.confidence,
            source=self.source,
            main_color_points=tuple(move_point(point) for point in self.main_color_points),
            shade_color_points=tuple(move_point(point) for point in self.shade_color_points),
            pencil_tool_roi=self.pencil_region.translated(dx, dy),
            fill_tool_roi=self.fill_region.translated(dx, dy),
            reference_left=self.reference_left + dx,
            reference_top=self.reference_top + dy,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Ces collections sont persistées en JSON : les exposer directement
        # comme listes garantit que to_dict() est identique au document relu.
        data["main_color_points"] = [asdict(point) for point in self.main_color_points]
        data["shade_color_points"] = [asdict(point) for point in self.shade_color_points]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationProfile:
        reference_width = int(data["reference_width"])
        reference_height = int(data["reference_height"])
        brush_data = data.get("small_brush")
        pencil_data = data.get("pencil_tool")
        fill_data = data.get("fill_tool")
        pencil_roi_data = data.get("pencil_tool_roi")
        fill_roi_data = data.get("fill_tool_roi")
        palette_icon = Point(**data["palette_icon"])
        back_button = Point(**data["back_button"])
        pencil_tool = (
            Point(**pencil_data)
            if pencil_data
            else Point(
                palette_icon.x - 1936 * reference_width / 2560,
                palette_icon.y - 257 * reference_height / 1440,
            )
        )
        fill_tool = (
            Point(**fill_data)
            if fill_data
            else Point(
                palette_icon.x - 1936 * reference_width / 2560,
                palette_icon.y - 156 * reference_height / 1440,
            )
        )
        return cls(
            name=data["name"],
            reference_width=reference_width,
            reference_height=reference_height,
            canvas=ScreenRect(**data["canvas"]),
            palette_icon=palette_icon,
            back_button=back_button,
            small_brush=(
                Point(**brush_data)
                if brush_data
                else Point(
                    pencil_tool.x + 136 * reference_width / 2560,
                    pencil_tool.y + 183 * reference_height / 1440,
                )
            ),
            pencil_tool=pencil_tool,
            fill_tool=fill_tool,
            main_colors=CenterGrid(
                first=Point(**data["main_colors"]["first"]),
                last=Point(**data["main_colors"]["last"]),
                columns=int(data["main_colors"]["columns"]),
                rows=int(data["main_colors"]["rows"]),
            ),
            shades=CenterGrid(
                first=Point(**data["shades"]["first"]),
                last=Point(**data["shades"]["last"]),
                columns=int(data["shades"]["columns"]),
                rows=int(data["shades"]["rows"]),
            ),
            confidence=float(data.get("confidence", 0.0)),
            source=str(data.get("source", "manual")),
            main_color_points=tuple(Point(**point) for point in data.get("main_color_points", ())),
            shade_color_points=tuple(
                Point(**point) for point in data.get("shade_color_points", ())
            ),
            pencil_tool_roi=(
                ScreenRect(**pencil_roi_data)
                if pencil_roi_data
                else default_tool_roi(pencil_tool, reference_width, reference_height)
            ),
            fill_tool_roi=(
                ScreenRect(**fill_roi_data)
                if fill_roi_data
                else default_tool_roi(fill_tool, reference_width, reference_height)
            ),
            reference_left=float(data.get("reference_left", 0.0)),
            reference_top=float(data.get("reference_top", 0.0)),
        )


def default_tool_roi(point: Point, reference_width: int, reference_height: int) -> ScreenRect:
    """Return the legacy observer footprint centred on one tool button."""

    width = max(20.0, float(reference_width) * 0.05)
    height = max(16.0, float(reference_height) * 0.056)
    return ScreenRect(point.x - width / 2, point.y - height / 2, width, height)


REFERENCE_PROFILE_2560 = CalibrationProfile(
    name="Heartopia 2560×1440",
    reference_width=2560,
    reference_height=1440,
    canvas=ScreenRect(813, 333, 879, 879),
    palette_icon=Point(2176, 633),
    back_button=Point(2399, 325),
    small_brush=Point(376, 559),
    pencil_tool=Point(240, 376),
    fill_tool=Point(240, 477),
    main_colors=CenterGrid(Point(2302, 347), Point(2466, 1153), 2, 8),
    shades=CenterGrid(Point(2302, 551), Point(2466, 1032), 2, 5),
    confidence=0.72,
    source="reference-images",
)

REFERENCE_CANVASES_2560: dict[AspectRatio, ScreenRect] = {
    AspectRatio.LANDSCAPE_16_9: ScreenRect(519, 333, 1466, 879),
    AspectRatio.LANDSCAPE_4_3: ScreenRect(703, 333, 1099, 879),
    AspectRatio.SQUARE: ScreenRect(813, 333, 879, 879),
    AspectRatio.PORTRAIT_3_4: ScreenRect(901, 333, 703, 879),
    AspectRatio.PORTRAIT_9_16: ScreenRect(988, 333, 528, 879),
}


def reference_profile(ratio: AspectRatio) -> CalibrationProfile:
    return replace(REFERENCE_PROFILE_2560, canvas=REFERENCE_CANVASES_2560[ratio])
