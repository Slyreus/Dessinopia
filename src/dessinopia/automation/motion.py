from __future__ import annotations

import math
from random import Random

from dessinopia.core.models import CalibrationProfile, Point, ScreenRect


def ui_click_radius(profile: CalibrationProfile) -> float:
    """Return a small resolution-aware radius for safe control clicks."""

    scale = min(profile.reference_width / 640, profile.reference_height / 360)
    return max(1.5, min(4.5, scale))


def randomized_click_point(
    point: Point,
    random: Random,
    radius: float,
    safe_region: ScreenRect | None = None,
) -> Point:
    """Sample one precise point inside a tiny disk and an optional hit box."""

    bounded_radius = max(0.0, float(radius))
    distance = bounded_radius * math.sqrt(random.random())
    angle = random.uniform(0.0, math.tau)
    x = point.x + math.cos(angle) * distance
    y = point.y + math.sin(angle) * distance
    if safe_region is not None:
        margin_x = min(1.0, safe_region.width * 0.1)
        margin_y = min(1.0, safe_region.height * 0.1)
        left = safe_region.left + margin_x
        right = safe_region.right - margin_x
        top = safe_region.top + margin_y
        bottom = safe_region.bottom - margin_y
        x = min(max(x, left), right)
        y = min(max(y, top), bottom)
    return Point(x, y)
