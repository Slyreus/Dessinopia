from __future__ import annotations

from dessinopia.core.models import Point, ScreenRect


def map_point_between_spaces(
    point: Point,
    source_bounds: ScreenRect,
    target_bounds: ScreenRect,
) -> Point:
    """Map a point between coordinate spaces describing the same display."""

    if source_bounds.width <= 0 or source_bounds.height <= 0:
        raise ValueError("La géométrie source doit avoir des dimensions positives")
    return Point(
        target_bounds.left
        + (point.x - source_bounds.left) * target_bounds.width / source_bounds.width,
        target_bounds.top
        + (point.y - source_bounds.top) * target_bounds.height / source_bounds.height,
    )


def qt_geometry_to_physical(logical_bounds: ScreenRect, device_ratio: float) -> ScreenRect:
    """Convert QScreen geometry to Win32 physical bounds on a Qt virtual desktop.

    Qt scales each screen's size but deliberately leaves its virtual-desktop origin
    unchanged. Scaling ``left`` or ``top`` would shift secondary monitors whenever
    their DPR differs from the primary screen.
    """

    if device_ratio <= 0:
        raise ValueError("Le facteur d'échelle de l'écran doit être positif")
    return ScreenRect(
        logical_bounds.left,
        logical_bounds.top,
        logical_bounds.width * device_ratio,
        logical_bounds.height * device_ratio,
    )


def map_rect_between_spaces(
    rect: ScreenRect,
    source_bounds: ScreenRect,
    target_bounds: ScreenRect,
) -> ScreenRect:
    """Map a rectangle between two coordinate spaces describing the same display.

    ``source_bounds`` can be Win32 physical pixels while ``target_bounds`` contains
    Qt device-independent coordinates.  Origins must be translated as well as sizes;
    simply dividing global coordinates by a DPR is wrong on mixed-DPI desktops.
    """

    if source_bounds.width <= 0 or source_bounds.height <= 0:
        raise ValueError("La géométrie source doit avoir des dimensions positives")
    scale_x = target_bounds.width / source_bounds.width
    scale_y = target_bounds.height / source_bounds.height
    return ScreenRect(
        target_bounds.left + (rect.left - source_bounds.left) * scale_x,
        target_bounds.top + (rect.top - source_bounds.top) * scale_y,
        rect.width * scale_x,
        rect.height * scale_y,
    )
