from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from dessinopia.core.models import CalibrationProfile, ScreenRect


def tool_marker_geometry_error(profile: CalibrationProfile) -> str | None:
    """Return an actionable error when pencil/fill markers can be swapped.

    In every supplied Heartopia layout the two tools form one vertical column,
    with the pencil above the bucket. The selected-background observer alone
    cannot detect an exact marker swap because it intentionally follows the
    calibrated labels, so this independent geometric invariant is mandatory.
    """

    width = max(1.0, float(profile.reference_width))
    height = max(1.0, float(profile.reference_height))
    horizontal_gap = abs(profile.pencil_tool.x - profile.fill_tool.x)
    vertical_gap = profile.fill_tool.y - profile.pencil_tool.y
    if vertical_gap <= 0:
        return (
            "Les repères Crayon et Pot semblent inversés : le Crayon doit être "
            "au-dessus du Pot dans la barre d'outils gauche."
        )
    if horizontal_gap > width * 0.06:
        return "Les repères Crayon et Pot doivent rester dans la même colonne verticale."
    if not height * 0.025 <= vertical_gap <= height * 0.15:
        return (
            "L'écart entre les repères Crayon et Pot est incohérent. "
            "Ajustez-les sur l'overlay avant de dessiner."
        )
    return None


class DrawingTool(StrEnum):
    PENCIL = "crayon"
    FILL = "pot de peinture"
    UNKNOWN = "outil inconnu"


@dataclass(frozen=True, slots=True)
class ToolObservation:
    tool: DrawingTool
    confidence: float
    pencil_score: float
    fill_score: float

    def matches(self, expected: DrawingTool) -> bool:
        return self.tool is expected and self.confidence >= 0.60

    @property
    def summary(self) -> str:
        return (
            f"{self.tool.value}, confiance {self.confidence:.0%} "
            f"(crayon={self.pencil_score:.2f}, pot={self.fill_score:.2f})"
        )


class ToolObserver:
    """Recognize the cyan selected-tool background on Heartopia's left bar."""

    def observe(self, rgb: np.ndarray, profile: CalibrationProfile) -> ToolObservation:
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError("Une capture RGB est requise")
        height, width = rgb.shape[:2]
        scaled = profile.scaled_to(width, height)
        pencil_score = self._selected_score(rgb, scaled.pencil_region)
        fill_score = self._selected_score(rgb, scaled.fill_region)
        scores = (
            (DrawingTool.PENCIL, pencil_score),
            (DrawingTool.FILL, fill_score),
        )
        ordered = sorted(scores, key=lambda item: item[1], reverse=True)
        tool, best = ordered[0]
        margin = best - ordered[1][1]
        if best < 0.28 or margin < 0.16:
            return ToolObservation(
                DrawingTool.UNKNOWN,
                min(1.0, max(best, 0.0)),
                pencil_score,
                fill_score,
            )
        confidence = min(1.0, 0.72 * best / 0.62 + 0.28 * margin / 0.62)
        return ToolObservation(tool, confidence, pencil_score, fill_score)

    def observe_many(
        self,
        frames: Iterable[np.ndarray],
        profile: CalibrationProfile,
    ) -> ToolObservation:
        observations = tuple(self.observe(frame, profile) for frame in frames)
        if not observations:
            raise ValueError("Au moins une image est requise")
        if len(observations) == 1:
            return observations[0]
        weights = {
            tool: sum(
                max(0.05, item.confidence)
                for item in observations
                if item.tool is tool
            )
            for tool in DrawingTool
        }
        tool = max(weights, key=weights.__getitem__)
        agreeing = tuple(item for item in observations if item.tool is tool)
        if len(agreeing) * 2 <= len(observations):
            tool = DrawingTool.UNKNOWN
            agreeing = observations
        confidence = sum(item.confidence for item in agreeing) / len(agreeing)
        confidence = min(1.0, 0.78 * confidence + 0.22 * len(agreeing) / len(observations))
        return ToolObservation(
            tool,
            confidence,
            sum(item.pencil_score for item in agreeing) / len(agreeing),
            sum(item.fill_score for item in agreeing) / len(agreeing),
        )

    @staticmethod
    def _selected_score(rgb: np.ndarray, region: ScreenRect) -> float:
        height, width = rgb.shape[:2]
        left = min(width, max(0, math.floor(region.left)))
        right = min(width, max(0, math.ceil(region.right)))
        top = min(height, max(0, math.floor(region.top)))
        bottom = min(height, max(0, math.ceil(region.bottom)))
        roi = rgb[top:bottom, left:right, :3]
        if roi.size == 0:
            return 0.0
        red, green, blue = (roi[..., channel].astype(np.int16) for channel in range(3))
        cyan = (
            (red < 100)
            & (green > 135)
            & (blue > 125)
            & ((green - red) > 55)
            & ((blue - red) > 45)
        )
        return float(np.mean(cyan))
