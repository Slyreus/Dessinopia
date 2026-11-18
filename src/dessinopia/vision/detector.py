from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

import cv2
import numpy as np

from dessinopia.core.color_spaces import delta_e_ciede2000, srgb_to_lab
from dessinopia.core.models import (
    REFERENCE_PROFILE_2560,
    CalibrationProfile,
    CanvasSize,
    ScreenRect,
)
from dessinopia.core.palette import MAIN_CELLS, colors_for_group
from dessinopia.vision.editor_geometry import locate_editor, measure_controls


class PaletteState(StrEnum):
    MAIN = "palette principale"
    SHADES = "nuances"
    UNKNOWN = "inconnu"


class CanvasEvidence(StrEnum):
    """Visual evidence used to validate the calibrated drawing area."""

    NONE = "aucune"
    LIGHT_SURFACE = "surface claire"
    CALIBRATED_FRAME = "cadre calibré"
    SURFACE_AND_FRAME = "surface et cadre"
    PAINTABLE_AREA = "surface peignable dans le cadre de l’atelier"


@dataclass(frozen=True, slots=True)
class StateDetection:
    state: PaletteState
    confidence: float
    group: int | None = None


@dataclass(frozen=True, slots=True)
class CanvasDetection:
    rect: ScreenRect
    confidence: float


@dataclass(frozen=True, slots=True)
class CanvasAssessment:
    """Result of a canvas check performed around an existing calibration.

    ``detect_canvas`` deliberately keeps its historical behaviour and searches
    for the light, mostly empty drawing surface.  That signal disappears as a
    drawing progresses.  This assessment also looks for the four persistent
    edges of Heartopia's editor, so it remains useful on a painted canvas while
    rejecting an ordinary gameplay scene.
    """

    detection: CanvasDetection | None
    evidence: CanvasEvidence
    confidence: float

    @property
    def visible(self) -> bool:
        return self.detection is not None


def canvas_matches_calibration(
    detected: ScreenRect,
    expected: ScreenRect,
    *,
    tolerance_ratio: float = 0.03,
    minimum_tolerance: float = 16.0,
) -> bool:
    """Whether a freshly detected canvas still matches the calibrated edges.

    Each edge gets the larger of 3% of the calibrated dimension or 16 physical
    pixels.  This absorbs small capture/detection variations without accepting a
    board whose Heartopia format has genuinely changed.
    """

    if tolerance_ratio < 0 or minimum_tolerance < 0:
        raise ValueError("La tolérance ne peut pas être négative")
    if detected.width <= 0 or detected.height <= 0 or expected.width <= 0 or expected.height <= 0:
        return False
    tolerance_x = max(minimum_tolerance, expected.width * tolerance_ratio)
    tolerance_y = max(minimum_tolerance, expected.height * tolerance_ratio)
    return (
        abs(detected.left - expected.left) <= tolerance_x
        and abs(detected.right - expected.right) <= tolerance_x
        and abs(detected.top - expected.top) <= tolerance_y
        and abs(detected.bottom - expected.bottom) <= tolerance_y
    )


def canvas_matches_grid_aspect(
    detected: ScreenRect,
    grid: CanvasSize,
    *,
    tolerance_ratio: float = 0.04,
) -> bool:
    """Check that the visible board format agrees with the selected logical grid."""

    if detected.width <= 0 or detected.height <= 0 or grid.width <= 0 or grid.height <= 0:
        return False
    if tolerance_ratio < 0:
        raise ValueError("La tolérance ne peut pas être négative")
    detected_ratio = detected.width / detected.height
    expected_ratio = grid.width / grid.height
    return abs(detected_ratio / expected_ratio - 1.0) <= tolerance_ratio


@dataclass(frozen=True, slots=True)
class _FrameEdge:
    position: int
    quality: float


class HeartopiaDetector:
    def detect_canvas(self, rgb: np.ndarray) -> CanvasDetection | None:
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError("Une capture RGB est requise")
        height, width = rgb.shape[:2]
        red, green, blue = (rgb[..., index] for index in range(3))
        neutral = (
            (red > 235) & (green > 232) & (blue > 225) & ((rgb.max(axis=2) - rgb.min(axis=2)) < 38)
        )
        region = np.zeros_like(neutral, dtype=np.uint8)
        left, right = int(width * 0.12), int(width * 0.81)
        top, bottom = int(height * 0.14), int(height * 0.91)
        region[top:bottom, left:right] = neutral[top:bottom, left:right].astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel, iterations=2)
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(region)

        candidates: list[tuple[float, ScreenRect]] = []
        for index in range(1, count):
            x, y, item_width, item_height, area = (int(v) for v in stats[index])
            if item_height < height * 0.42 or item_width < width * 0.15:
                continue
            if item_width > width * 0.72 or item_height > height * 0.82:
                continue
            rectangularity = area / max(1, item_width * item_height)
            if rectangularity < 0.72:
                continue
            center_bias = 1 - abs((x + item_width / 2) / width - 0.49)
            score = 0.7 * rectangularity + 0.3 * center_bias
            candidates.append((score, ScreenRect(x, y, item_width, item_height)))
        if not candidates:
            return None
        score, rect = max(candidates, key=lambda item: item[0])
        return CanvasDetection(rect=rect, confidence=float(min(1.0, score)))

    def detect_canvas_near(
        self,
        rgb: np.ndarray,
        expected: ScreenRect,
        *,
        search_ratio: float = 0.08,
        minimum_search: int = 24,
        maximum_size_delta_ratio: float = 0.12,
    ) -> CanvasDetection | None:
        """Find Heartopia's persistent canvas frame near a calibrated rectangle.

        Contrary to :meth:`detect_canvas`, this method does not rely on a large
        connected area of blank pixels.  It searches for four long, continuous
        edges close to the saved calibration.  A local-prominence requirement is
        important here: textured gameplay has many gradients, but normally no
        four-sided frame whose geometry agrees with the calibrated canvas.

        The returned rectangle uses physical capture coordinates, like
        ``expected``.  The method is intentionally separate from the historical
        detector so existing callers retain the exact blank-canvas behaviour.
        """

        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError("Une capture RGB est requise")
        if search_ratio < 0 or maximum_size_delta_ratio < 0 or minimum_search < 0:
            raise ValueError("Les tolérances de recherche ne peuvent pas être négatives")
        if expected.width <= 0 or expected.height <= 0:
            return None

        height, width = rgb.shape[:2]
        if width < 3 or height < 3:
            return None
        if (
            expected.right < 1
            or expected.bottom < 1
            or expected.left >= width - 1
            or expected.top >= height - 1
        ):
            return None

        pixels = np.asarray(rgb[..., :3], dtype=np.float32)
        gray = pixels[..., 0] * 0.299 + pixels[..., 1] * 0.587 + pixels[..., 2] * 0.114
        vertical_gradient = np.abs(np.diff(gray, axis=1))
        horizontal_gradient = np.abs(np.diff(gray, axis=0))
        search = max(
            minimum_search,
            int(round(min(expected.width, expected.height) * search_ratio)),
        )
        trim_ratio = 0.06

        def select_edge(profile: np.ndarray, offset: int, target: float) -> _FrameEdge | None:
            if profile.size == 0:
                return None
            baseline = float(np.median(profile))
            threshold = max(4.0, baseline + 3.0)
            candidates = np.flatnonzero(profile >= threshold)
            if candidates.size == 0:
                return None

            positions = offset + candidates
            distances = np.abs(positions - target)
            closest_distance = float(np.min(distances))
            closest = candidates[distances <= closest_distance + 0.5]
            index = int(closest[int(np.argmax(profile[closest]))])
            peak = float(profile[index])
            prominence = peak - baseline
            location_quality = 1.0 - abs(offset + index - target) / max(1.0, search + 1.0)
            quality = (
                0.45 * float(np.clip((peak - 3.0) / 8.0, 0.0, 1.0))
                + 0.40 * float(np.clip((prominence - 2.0) / 8.0, 0.0, 1.0))
                + 0.15 * float(np.clip(location_quality, 0.0, 1.0))
            )
            return _FrameEdge(offset + index, quality)

        along_top = max(0, int(round(expected.top + expected.height * trim_ratio)))
        along_bottom = min(
            height,
            int(round(expected.bottom - expected.height * trim_ratio)),
        )
        along_left = max(0, int(round(expected.left + expected.width * trim_ratio)))
        along_right = min(
            width,
            int(round(expected.right - expected.width * trim_ratio)),
        )
        if along_bottom - along_top < 8 or along_right - along_left < 8:
            return None

        def vertical_edge(target: float) -> _FrameEdge | None:
            start = max(0, int(round(target - search)))
            stop = min(vertical_gradient.shape[1], int(round(target + search + 1)))
            if stop <= start:
                return None
            profile = np.quantile(
                vertical_gradient[along_top:along_bottom, start:stop],
                0.35,
                axis=0,
            )
            return select_edge(profile, start, target)

        def horizontal_edge(target: float) -> _FrameEdge | None:
            start = max(0, int(round(target - search)))
            stop = min(horizontal_gradient.shape[0], int(round(target + search + 1)))
            if stop <= start:
                return None
            profile = np.quantile(
                horizontal_gradient[start:stop, along_left:along_right],
                0.35,
                axis=1,
            )
            return select_edge(profile, start, target)

        left_edge = vertical_edge(expected.left - 1.0)
        right_edge = vertical_edge(expected.right - 1.0)
        top_edge = horizontal_edge(expected.top - 1.0)
        bottom_edge = horizontal_edge(expected.bottom - 1.0)
        edges = (left_edge, right_edge, top_edge, bottom_edge)
        if any(edge is None or edge.quality < 0.32 for edge in edges):
            return None

        # The gradient position lies between two pixels; the drawing area starts
        # on the pixel immediately after the left/top edge and ends after the
        # corresponding right/bottom edge.
        assert left_edge is not None
        assert right_edge is not None
        assert top_edge is not None
        assert bottom_edge is not None
        left = left_edge.position + 1
        right = right_edge.position + 1
        top = top_edge.position + 1
        bottom = bottom_edge.position + 1
        detected_width = right - left
        detected_height = bottom - top
        if detected_width <= 0 or detected_height <= 0:
            return None

        width_delta = abs(detected_width / expected.width - 1.0)
        height_delta = abs(detected_height / expected.height - 1.0)
        aspect_delta = abs(
            (detected_width / detected_height) / (expected.width / expected.height) - 1.0
        )
        if (
            width_delta > maximum_size_delta_ratio
            or height_delta > maximum_size_delta_ratio
            or aspect_delta > maximum_size_delta_ratio
        ):
            return None

        mean_edge_quality = float(np.mean([edge.quality for edge in edges]))
        if maximum_size_delta_ratio == 0:
            geometry_quality = float(width_delta == 0 and height_delta == 0)
        else:
            geometry_quality = 1.0 - max(width_delta, height_delta) / maximum_size_delta_ratio
        confidence = min(
            1.0,
            0.45 + 0.45 * mean_edge_quality + 0.10 * max(0.0, geometry_quality),
        )
        return CanvasDetection(
            ScreenRect(left, top, detected_width, detected_height),
            confidence,
        )

    def assess_canvas(
        self,
        rgb: np.ndarray,
        expected: ScreenRect,
    ) -> CanvasAssessment:
        """Assess the calibrated canvas using its surface and persistent frame."""

        board = locate_editor(rgb, expected)
        if board is not None:
            return CanvasAssessment(
                CanvasDetection(board.canvas, 1.0), CanvasEvidence.PAINTABLE_AREA, 1.0
            )
        surface = self.detect_canvas(rgb)
        surface_near = surface is not None and canvas_matches_calibration(
            surface.rect,
            expected,
            tolerance_ratio=0.08,
            minimum_tolerance=24.0,
        )
        frame = self.detect_canvas_near(rgb, expected)

        if frame is not None and surface_near:
            assert surface is not None
            confidence = min(1.0, 0.55 * surface.confidence + 0.45 * frame.confidence + 0.08)
            return CanvasAssessment(
                detection=surface,
                evidence=CanvasEvidence.SURFACE_AND_FRAME,
                confidence=confidence,
            )
        if frame is not None:
            return CanvasAssessment(
                detection=frame,
                evidence=CanvasEvidence.CALIBRATED_FRAME,
                confidence=frame.confidence,
            )
        if surface_near:
            assert surface is not None
            confidence = min(1.0, surface.confidence * 0.88)
            return CanvasAssessment(
                detection=surface,
                evidence=CanvasEvidence.LIGHT_SURFACE,
                confidence=confidence,
            )
        return CanvasAssessment(
            detection=None,
            evidence=CanvasEvidence.NONE,
            confidence=0.0,
        )

    def auto_profile(
        self,
        rgb: np.ndarray,
        name: str = "Détection automatique",
        *,
        expected_canvas: ScreenRect | None = None,
        base_profile: CalibrationProfile | None = None,
    ) -> CalibrationProfile | None:
        height, width = rgb.shape[:2]
        base = (base_profile or REFERENCE_PROFILE_2560).remapped_to(
            ScreenRect(0, 0, width, height)
        )
        board = locate_editor(rgb, expected_canvas or base.canvas)
        if board is None:
            return None
        return replace(
            measure_controls(rgb, base, board.frame),
            name=name,
            canvas=board.canvas,
            confidence=1.0,
            source="automatic",
        )

    def detect_palette_state(self, rgb: np.ndarray, profile: CalibrationProfile) -> StateDetection:
        height, width = rgb.shape[:2]
        p = profile.scaled_to(width, height)
        cx, cy = int(round(p.back_button.x)), int(round(p.back_button.y))
        x0, x1 = max(0, cx - int(width * 0.055)), min(width, cx + int(width * 0.055))
        y0, y1 = max(0, cy - int(height * 0.035)), min(height, cy + int(height * 0.035))
        roi = rgb[y0:y1, x0:x1]
        if roi.size == 0:
            return StateDetection(PaletteState.UNKNOWN, 0.0)

        red, green, blue = (roi[..., channel] for channel in range(3))
        salmon = (
            (red > 165) & (red < 230) & (green > 115) & (green < 185) & (blue > 95) & (blue < 170)
        )
        cream = (red > 225) & (green > 210) & (blue > 175)
        salmon_ratio = float(np.mean(salmon))
        cream_ratio = float(np.mean(cream))
        shades_score = min(1.0, salmon_ratio * 1.8 + cream_ratio * 3.2)
        if shades_score >= 0.56:
            group, group_confidence = self.classify_group(rgb, p)
            return StateDetection(
                PaletteState.SHADES, shades_score, group if group_confidence > 0.35 else None
            )
        main_score = self.main_palette_confidence(rgb, p)
        if main_score >= 0.62:
            return StateDetection(PaletteState.MAIN, main_score)
        return StateDetection(PaletteState.UNKNOWN, max(shades_score, main_score))

    def main_palette_confidence(self, rgb: np.ndarray, profile: CalibrationProfile) -> float:
        """Measure whether the 13 known main-family tiles are really visible.

        Absence of the salmon Back button is not sufficient: a normal gameplay
        scene also lacks that button.  Sampling every known family tile makes a
        false MAIN result outside the painting editor extremely unlikely.
        """

        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return 0.0
        observations: list[np.ndarray] = []
        expected: list[tuple[int, int, int]] = []
        radius = max(3, int(round(rgb.shape[1] / 320)))
        for group, (column, row) in enumerate(MAIN_CELLS):
            point = profile.main_color_point(column, row)
            x, y = int(round(point.x)), int(round(point.y))
            patch = rgb[
                max(0, y - radius) : min(rgb.shape[0], y + radius + 1),
                max(0, x - radius) : min(rgb.shape[1], x + radius + 1),
            ]
            if patch.size == 0:
                return 0.0
            observations.append(np.median(patch.reshape(-1, 3), axis=0))
            colors = colors_for_group(group)
            expected.append(colors[0 if group == 0 else 1].rgb)

        observed_lab = srgb_to_lab(np.asarray(observations, dtype=np.uint8))
        expected_lab = srgb_to_lab(np.asarray(expected, dtype=np.uint8))
        deltas = np.asarray(delta_e_ciede2000(observed_lab, expected_lab), dtype=float)
        matching_ratio = float(np.mean(deltas <= 18.0))
        median_quality = max(0.0, 1.0 - float(np.median(deltas)) / 40.0)
        return min(1.0, 0.72 * matching_ratio + 0.28 * median_quality)

    def classify_group(
        self, rgb: np.ndarray, profile: CalibrationProfile
    ) -> tuple[int | None, float]:
        observations: list[np.ndarray] = []
        radius = max(3, int(round(rgb.shape[1] / 320)))
        for shade_index in range(5):
            point = profile.shade_point(shade_index % 2, shade_index // 2)
            x, y = int(round(point.x)), int(round(point.y))
            patch = rgb[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1]
            if patch.size:
                observations.append(np.median(patch.reshape(-1, 3), axis=0))
        if len(observations) < 3:
            return None, 0.0

        observed_lab = srgb_to_lab(np.asarray(observations, dtype=np.uint8))
        scores: list[float] = []
        for group in range(13):
            group_colors = colors_for_group(group)[: len(observations)]
            expected = np.asarray([color.rgb for color in group_colors], dtype=np.uint8)
            expected_lab = srgb_to_lab(expected)
            scores.append(float(np.mean(delta_e_ciede2000(observed_lab, expected_lab))))
        order = np.argsort(scores)
        best, second = int(order[0]), float(scores[int(order[1])])
        best_score = float(scores[best])
        confidence = max(0.0, min(1.0, (second - best_score) / max(4.0, second)))
        return best, confidence
