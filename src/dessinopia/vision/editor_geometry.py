"""Mesure de la toile et des commandes visibles de l'atelier Heartopia."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import pairwise

import cv2
import numpy as np

from dessinopia.core.models import (
    REFERENCE_PROFILE_2560,
    CalibrationProfile,
    CenterGrid,
    Point,
    ScreenRect,
)
from dessinopia.core.palette import MAIN_CELLS, colors_for_group


@dataclass(frozen=True, slots=True)
class EditorBoard:
    frame: ScreenRect
    canvas: ScreenRect


def _boxes(mask: np.ndarray, x_offset: int = 0) -> list[ScreenRect]:
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    result = []
    for row in stats[1:]:
        x, y, w, h, area = map(int, row)
        if min(w, h) >= 5 and area >= 0.30 * w * h:
            result.append(ScreenRect(x + x_offset, y, w, h))
    return result


def locate_editor(rgb: np.ndarray, expected: ScreenRect | None = None) -> EditorBoard | None:
    """Mesurer le contenu du cadre, en excluant son filet et ses marges crème."""
    height, width = rgb.shape[:2]
    trim = cv2.inRange(rgb, (157, 141, 117), (204, 189, 172))
    contours, hierarchy = cv2.findContours(trim, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    candidates = []
    for index, contour in enumerate(contours):
        child = hierarchy[0, index, 2]
        if child < 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if not (
            min(w, h) > min(width, height) * 0.16
            and 0.65 < w / h < 3.5
            and cv2.contourArea(contour) > 0.95 * w * h
        ):
            continue
        # Le contour enfant borde le trou du filet. Son premier et son dernier
        # pixel appartiennent encore au filet, pas à l'intérieur du tableau.
        cx, cy, cw, ch = cv2.boundingRect(contours[child])
        if cw < w * 0.97 or ch < h * 0.97 or y < 2:
            continue
        left, top = cx + 1, cy + 1
        content = rgb[top : cy + ch - 1, left : cx + cw - 1]
        margin = np.median(rgb[y - 2, x + w // 4 : x + 3 * w // 4], axis=0)
        if np.max(np.abs(margin - (246, 239, 213))) > 19:
            continue
        lower = tuple(map(float, margin - 8))
        upper = tuple(map(float, margin + 8))
        # Un filet anticrénelé peut encore teinter une rangée isolée dans la
        # marge après redimensionnement. Une toile occupe plusieurs rangées.
        ink = cv2.bitwise_not(cv2.inRange(content, lower, upper))
        # À la limite du cadre, un pixel peut mélanger le filet brun et la marge.
        # Le retirer avant de mesurer évite d'ajouter une colonne entière en Full HD.
        for pixels, edge in (
            (content[0], ink[0]),
            (content[-1], ink[-1]),
            (content[:, 0], ink[:, 0]),
            (content[:, -1], ink[:, -1]),
        ):
            difference = margin - pixels
            fraction = difference[:, 1] / 73
            residual = np.abs(difference - fraction[:, None] * (62, 73, 65)).max(axis=1)
            edge[(fraction > 0) & (fraction < 1.2) & (residual < 5)] = 0
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) != 0
        rows = np.flatnonzero(ink.mean(axis=1) > 0.16)
        cols = np.flatnonzero(ink.mean(axis=0) > 0.16)
        if not rows.size or not cols.size:
            continue
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        if (
            max((x1 - x0) / ink.shape[1], (y1 - y0) / ink.shape[0]) < 0.97
            or abs(x0 + x1 - ink.shape[1]) > 0.035 * w
            or abs(y0 + y1 - ink.shape[0]) > 0.035 * h
            or ink[y0:y1, x0:x1].mean() < 0.65
        ):
            continue
        candidates.append(
            EditorBoard(ScreenRect(x, y, w, h), ScreenRect(left + x0, top + y0, x1 - x0, y1 - y0))
        )
    if len(candidates) == 1:
        return candidates[0]
    if candidates and expected is not None:
        return min(
            candidates,
            key=lambda board: (
                abs(board.canvas.left - expected.left)
                + abs(board.canvas.top - expected.top)
                + abs(board.canvas.right - expected.right)
                + abs(board.canvas.bottom - expected.bottom)
            ),
        )
    return None


def _columns(boxes: list[ScreenRect], count: int) -> list[list[ScreenRect]]:
    """Retenir les colonnes entières de commandes de taille et de pas réguliers."""
    result = []
    for first in boxes:
        column = sorted(
            (
                b
                for b in boxes
                if abs(b.center.x - first.center.x) < first.width * 0.18
                and 0.74 < b.height / first.height < 1.35
            ),
            key=lambda b: b.center.y,
        )
        if len(column) != count or column[0] != first:
            continue
        gaps = np.diff([b.center.y for b in column])
        if gaps.min() > first.height * 0.95 and gaps.max() < gaps.min() * 1.12:
            result.append(column)
    return result


def _measure_tools(
    rgb: np.ndarray, profile: CalibrationProfile, frame: ScreenRect
) -> CalibrationProfile:
    left = rgb[:, : round(frame.left)]
    if left.shape[1] < 5:
        return profile
    buttons = cv2.inRange(left, (141, 106, 92), (171, 138, 123))
    buttons |= cv2.inRange(left, (32, 179, 165), (90, 230, 220))
    columns = _columns(
        [
            b
            for b in _boxes(buttons)
            if 0.75 < b.width / b.height < 3.25
            and frame.height * 0.04 < b.height < frame.height * 0.17
        ],
        5,
    )
    if len(columns) != 1:
        return profile
    pencil, fill = columns[0][:2]
    previous = profile
    profile = replace(
        profile,
        pencil_tool=pencil.center,
        fill_tool=fill.center,
        pencil_tool_roi=pencil,
        fill_tool_roi=fill,
    )
    # La flèche inférieure du sélecteur d'épaisseur est à droite du crayon.
    arrows = [
        b
        for b in _boxes(cv2.inRange(left, (116, 87, 85), (135, 107, 107)))
        if 0.65 < b.width / b.height < 2.9
        and 0.22 < b.height / pencil.height < 0.52
        and pencil.right < b.center.x < pencil.right + pencil.width
        and pencil.bottom < b.center.y < columns[0][3].top
    ]
    pairs = _columns(arrows, 2)
    if len(pairs) == 1 and 2.4 < (pairs[0][1].top - pairs[0][0].top) / pairs[0][0].height < 4:
        profile = replace(profile, small_brush=pairs[0][1].center)
    else:
        # Le pot masque les flèches. Les boutons du même panneau donnent son échelle.
        spacing = fill.center.y - pencil.center.y
        legacy = Point(2173 * rgb.shape[1] / 2560, 797 * rgb.shape[0] / 1440)
        brush = previous.small_brush
        if previous.source == "reference-images" or (
            previous.source.startswith("automatic")
            and abs(brush.x - legacy.x) < 1
            and abs(brush.y - legacy.y) < 1
        ):
            # Les anciennes détections plaçaient le pinceau sur une couleur récente.
            brush = Point(
                pencil.center.x + 136 * pencil.width / 120, pencil.center.y + 183 * spacing / 101
            )
        elif previous.fill_tool.y > previous.pencil_tool.y:
            brush = Point(
                pencil.center.x
                + (brush.x - previous.pencil_tool.x) * pencil.width / previous.pencil_region.width,
                pencil.center.y
                + (brush.y - previous.pencil_tool.y)
                * spacing
                / (previous.fill_tool.y - previous.pencil_tool.y),
            )
        profile = replace(profile, small_brush=brush)
    return profile


def _align_palette(
    profile: CalibrationProfile, old: CenterGrid, new: CenterGrid
) -> CalibrationProfile:
    """Adapter les repères du menu masqué à l'échelle de la grille visible."""
    if old.last.x <= old.first.x or old.last.y <= old.first.y:
        return profile
    sx = (new.last.x - new.first.x) / (old.last.x - old.first.x)
    sy = (new.last.y - new.first.y) / (old.last.y - old.first.y)

    def align(point: Point) -> Point:
        return Point(
            new.first.x + (point.x - old.first.x) * sx, new.first.y + (point.y - old.first.y) * sy
        )

    def grid(value: CenterGrid) -> CenterGrid:
        return replace(value, first=align(value.first), last=align(value.last))

    return replace(
        profile,
        palette_icon=align(profile.palette_icon),
        back_button=align(profile.back_button),
        main_colors=grid(profile.main_colors),
        shades=grid(profile.shades),
        main_color_points=tuple(map(align, profile.main_color_points)),
        shade_color_points=tuple(map(align, profile.shade_color_points)),
    )


def _measure_palette(
    rgb: np.ndarray, profile: CalibrationProfile, frame: ScreenRect
) -> CalibrationProfile:
    offset = round(frame.right)
    right = rgb[:, offset:]
    if right.shape[1] < 5:
        return profile
    panel = cv2.inRange(right, (100, 72, 59), (132, 102, 94))
    spots = [
        b
        for b in _boxes(cv2.bitwise_not(panel), offset)
        if frame.height * 0.04 < b.height < frame.height * 0.21 and 0.42 < b.width / b.height < 1.9
    ]
    for rows in (8, 5, 3):
        columns = sorted(_columns(spots, rows), key=lambda c: c[0].center.x)
        for a, b in pairwise(columns):
            span = b[0].center.x - a[0].center.x
            if not 1.4 < span / a[0].width < 2.6:
                continue
            if any(abs(p.center.y - q.center.y) > 0.16 * p.height for p, q in zip(a, b)):
                continue
            points = tuple(box.center for pair in zip(a, b) for box in pair)
            step = (a[-1].center.y - a[0].center.y) / (rows - 1)
            first = points[0]
            if rows == 8:
                # La géométrie seule confondrait des boutons de menus et un nuancier.
                hits = 0
                for family, (col, row) in enumerate(MAIN_CELLS):
                    p = points[2 * row + col]
                    color = colors_for_group(family)[0 if family == 0 else 1].rgb
                    hits += np.max(np.abs(rgb[round(p.y), round(p.x)].astype(int) - color)) < 35
                if hits < 9:
                    continue
                grid = CenterGrid(first, points[-1], 2, 8)
                profile = replace(
                    _align_palette(profile, profile.main_colors, grid),
                    main_colors=grid,
                    main_color_points=points,
                )
                ref = REFERENCE_PROFILE_2560.main_colors
            else:
                backs = [
                    box
                    for box in _boxes(cv2.inRange(right, (183, 143, 125), (212, 173, 156)), offset)
                    if 1.2 < box.width / box.height < 5.5
                    and abs(box.center.x - first.x - span / 2) < span * 0.30
                    and 1.3 < (first.y - box.center.y) / step < 2.5
                ]
                if len(backs) != 1:
                    continue
                grid = CenterGrid(first, Point(b[-1].center.x, first.y + 4 * step), 2, 5)
                profile = _align_palette(profile, profile.shades, grid)
                hidden = tuple(
                    profile.shade_point(i % 2, i // 2)
                    if profile.shade_color_points or profile.source == "manual"
                    else grid.point(i % 2, i // 2)
                    for i in range(len(points), 10)
                )
                profile = replace(
                    profile,
                    back_button=backs[0].center,
                    shades=grid,
                    shade_color_points=points + hidden,
                )
                ref = REFERENCE_PROFILE_2560.shades
            # La silhouette crème de l'icône reste visible dans les deux menus.
            icon = REFERENCE_PROFILE_2560.palette_icon
            predicted = Point(
                first.x + (icon.x - ref.first.x) * span / (ref.last.x - ref.first.x),
                first.y
                + (icon.y - ref.first.y) * step * (ref.rows - 1) / (ref.last.y - ref.first.y),
            )
            glyphs = [
                g
                for g in _boxes(cv2.inRange(right, (236, 228, 197), (255, 249, 231)), offset)
                if 0.28 < g.width / span < 0.47
                and 0.48 < g.width / g.height < 2.2
                and abs(g.center.x - predicted.x) < 0.36 * span
                and abs(g.center.y - predicted.y) < 0.45 * step
            ]
            if len(glyphs) == 1:
                profile = replace(profile, palette_icon=glyphs[0].center)
            return profile
    return profile


def measure_controls(
    rgb: np.ndarray, profile: CalibrationProfile, frame: ScreenRect
) -> CalibrationProfile:
    return _measure_palette(rgb, _measure_tools(rgb, profile, frame), frame)
