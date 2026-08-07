from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from dessinopia.core.color_spaces import delta_e_ciede2000, srgb_to_lab
from dessinopia.core.models import CalibrationProfile, Point

ShadeCell = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ShadeSelectionObservation:
    selected_cell: ShadeCell | None
    expected_cell: ShadeCell
    confidence: float
    best_delta_e: float
    second_delta_e: float
    selection_method: str = "couleur"
    outline_score: float = 0.0
    second_outline_score: float = 0.0

    @property
    def matches_expected(self) -> bool:
        return self.selected_cell == self.expected_cell and self.confidence >= 0.62

    @property
    def summary(self) -> str:
        selected = (
            f"N{self.selected_cell[1] * 2 + self.selected_cell[0] + 1}"
            if self.selected_cell is not None
            else "aucune nuance certaine"
        )
        expected = f"N{self.expected_cell[1] * 2 + self.expected_cell[0] + 1}"
        details = (
            f"sélection={selected}, attendue={expected}, confiance={self.confidence:.0%}, "
            f"ΔE={self.best_delta_e:.2f}/{self.second_delta_e:.2f}"
        )
        if "contour" in self.selection_method:
            details += (
                f", preuve=contour ({self.outline_score:.2f}/"
                f"{self.second_outline_score:.2f})"
            )
        return details


class ShadeSelectionObserver:
    """Read the selected shade from two independent Heartopia UI signals.

    The colour-changing palette icon is compared with all visible swatches. A
    sufficiently isolated cream-white selection outline is also read directly
    around the swatches. The outline can recover a valid selection when UI
    rendering makes the icon colour drift, and it wins a disagreement because
    it is Heartopia's direct selected-state marker.
    """

    def observe(
        self,
        rgb: np.ndarray,
        profile: CalibrationProfile,
        *,
        expected_column: int,
        expected_row: int,
        allow_outline_tiebreaker: bool = False,
    ) -> ShadeSelectionObservation:
        image = self._validate_image(rgb)
        if not (0 <= expected_column < 2 and 0 <= expected_row < 5):
            raise ValueError("La cellule de nuance attendue est hors de la palette")
        height, width = image.shape[:2]
        scaled = profile.scaled_to(width, height)
        icon_point = Point(
            scaled.palette_icon.x,
            scaled.palette_icon.y + height * (55 / 1440),
        )
        icon_rgb = self._median_patch(
            image,
            icon_point,
            radius_x=max(4, round(width * (20 / 2560))),
            radius_y=max(3, round(height * (8 / 1440))),
        )
        icon_lab = srgb_to_lab(icon_rgb)
        candidates: list[tuple[ShadeCell, float]] = []
        for row in range(5):
            for column in range(2):
                swatch_rgb = self._median_patch(
                    image,
                    scaled.shade_point(column, row),
                    radius_x=max(4, round(width * (15 / 2560))),
                    radius_y=max(3, round(height * (10 / 1440))),
                )
                distance = float(delta_e_ciede2000(icon_lab, srgb_to_lab(swatch_rgb)))
                candidates.append(((column, row), distance))
        candidates.sort(key=lambda item: item[1])
        best_cell, best_delta = candidates[0]
        second_delta = candidates[1][1]
        margin = second_delta - best_delta
        selected = best_cell if best_delta <= 6.0 and margin >= 2.0 else None
        absolute_score = max(0.0, min(1.0, 1.0 - best_delta / 8.0))
        separation_score = max(0.0, min(1.0, margin / 6.0))
        confidence = 0.62 * absolute_score + 0.38 * separation_score
        selection_method = "couleur" if selected is not None else "indéterminée"

        outline_candidates = sorted(
            (
                (
                    (column, row),
                    self._selection_outline_score(
                        image,
                        scaled.shade_point(column, row),
                    ),
                )
                for row in range(5)
                for column in range(2)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        outline_cell, outline_score = outline_candidates[0]
        second_outline_score = outline_candidates[1][1]
        outline_margin = outline_score - second_outline_score
        outline_selected = (
            outline_cell
            if outline_score >= 0.24 and outline_margin >= 0.10
            else None
        )
        if allow_outline_tiebreaker and outline_selected is not None:
            outline_absolute = max(0.0, min(1.0, (outline_score - 0.24) / 0.16))
            outline_separation = max(0.0, min(1.0, (outline_margin - 0.10) / 0.15))
            outline_confidence = min(
                1.0,
                0.62 + 0.20 * outline_absolute + 0.18 * outline_separation,
            )
            if selected == outline_selected:
                selection_method = "couleur+contour"
                confidence = max(confidence, outline_confidence)
            else:
                selected = outline_selected
                selection_method = "contour"
                confidence = outline_confidence
        if selected is None:
            confidence = min(confidence, 0.45)
        return ShadeSelectionObservation(
            selected,
            (expected_column, expected_row),
            float(confidence),
            best_delta,
            second_delta,
            selection_method,
            outline_score,
            second_outline_score,
        )

    def observe_many(
        self,
        frames: Iterable[np.ndarray],
        profile: CalibrationProfile,
        *,
        expected_column: int,
        expected_row: int,
        allow_outline_tiebreaker: bool = False,
    ) -> ShadeSelectionObservation:
        observations = tuple(
            self.observe(
                frame,
                profile,
                expected_column=expected_column,
                expected_row=expected_row,
                allow_outline_tiebreaker=allow_outline_tiebreaker,
            )
            for frame in frames
        )
        if not observations:
            raise ValueError("Au moins une image est requise")
        if len(observations) == 1:
            return observations[0]
        counts = Counter(item.selected_cell for item in observations)
        selected, count = counts.most_common(1)[0]
        if selected is None or count * 2 <= len(observations):
            selected = None
        agreeing = tuple(item for item in observations if item.selected_cell == selected)
        if not agreeing:
            agreeing = observations
        confidence = sum(item.confidence for item in agreeing) / len(agreeing)
        confidence *= 0.75 + 0.25 * len(agreeing) / len(observations)
        methods = Counter(item.selection_method for item in agreeing)
        selection_method = methods.most_common(1)[0][0]
        return ShadeSelectionObservation(
            selected,
            (expected_column, expected_row),
            float(confidence),
            sum(item.best_delta_e for item in agreeing) / len(agreeing),
            sum(item.second_delta_e for item in agreeing) / len(agreeing),
            selection_method,
            sum(item.outline_score for item in agreeing) / len(agreeing),
            sum(item.second_outline_score for item in agreeing) / len(agreeing),
        )

    @staticmethod
    def _selection_outline_score(rgb: np.ndarray, point: Point) -> float:
        """Measure Heartopia's white selection ring around one shade swatch."""

        height, width = rgb.shape[:2]
        inner_x = max(8, round(width * (40 / 2560)))
        inner_y = max(8, round(height * (46 / 1440)))
        outer_x = max(inner_x + 3, round(width * (50 / 2560)))
        outer_y = max(inner_y + 3, round(height * (58 / 1440)))
        x, y = round(point.x), round(point.y)
        left = max(0, x - outer_x)
        right = min(width, x + outer_x + 1)
        top = max(0, y - outer_y)
        bottom = min(height, y + outer_y + 1)
        patch = rgb[top:bottom, left:right]
        if patch.size == 0:
            return 0.0
        yy, xx = np.mgrid[top - y : bottom - y, left - x : right - x]
        outside_inner = (xx / inner_x) ** 2 + (yy / inner_y) ** 2 >= 1.0
        inside_outer = (xx / outer_x) ** 2 + (yy / outer_y) ** 2 <= 1.0
        ring = outside_inner & inside_outer
        if not np.any(ring):
            return 0.0
        red, green, blue = (patch[..., channel] for channel in range(3))
        cream = (red > 220) & (green > 210) & (blue > 185)
        return float(np.mean(cream[ring]))

    @staticmethod
    def _validate_image(rgb: np.ndarray) -> np.ndarray:
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Une capture RGB est requise")
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError("La capture RGB ne peut pas être vide")
        return np.asarray(image[..., :3], dtype=np.uint8)

    @staticmethod
    def _median_patch(
        rgb: np.ndarray,
        point: Point,
        *,
        radius_x: int,
        radius_y: int,
    ) -> np.ndarray:
        x, y = round(point.x), round(point.y)
        patch = rgb[
            max(0, y - radius_y) : min(rgb.shape[0], y + radius_y + 1),
            max(0, x - radius_x) : min(rgb.shape[1], x + radius_x + 1),
        ]
        if patch.size == 0:
            raise ValueError("Une zone de nuance calibrée sort de la capture")
        return np.rint(np.median(patch.reshape(-1, 3), axis=0)).astype(np.uint8)
