from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from math import ceil

import numpy as np

from dessinopia.core.color_spaces import delta_e_ciede2000, srgb_to_lab
from dessinopia.core.models import CalibrationProfile, Point
from dessinopia.core.palette import GROUP_NAMES, MAIN_CELLS, colors_for_group
from dessinopia.vision.detector import PaletteState


class PaletteRecoveryAction(StrEnum):
    """Action sûre suggérée après l'observation de l'interface."""

    READY = "prêt"
    RETRY = "patienter puis relire"
    REOPEN_SHADES = "resélectionner la famille puis rouvrir les nuances"
    BACK_TO_MAIN = "revenir à la palette principale"
    RECALIBRATE = "vérifier les repères de palette"


@dataclass(frozen=True, slots=True)
class PaletteDiagnostics:
    """Mesures conservées pour expliquer une décision de reconnaissance."""

    back_button_score: float
    main_palette_score: float
    shade_palette_score: float
    best_group_delta_e: float | None
    second_group_delta_e: float | None
    matching_shades: int
    sampled_shades: int
    sample_count: int = 1
    messages: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PaletteObservation:
    """Description riche d'un écran de palette Heartopia."""

    state: PaletteState
    confidence: float
    group: int | None
    group_confidence: float
    expected_group: int | None
    recovery: PaletteRecoveryAction
    diagnostics: PaletteDiagnostics

    @property
    def matches_expected(self) -> bool:
        if self.expected_group is None:
            return self.state is PaletteState.MAIN
        return (
            self.state is PaletteState.SHADES
            and self.group == self.expected_group
            and self.recovery is PaletteRecoveryAction.READY
        )

    @property
    def summary(self) -> str:
        if self.state is PaletteState.MAIN:
            visible = "palette principale"
        elif self.state is PaletteState.SHADES and self.group is not None:
            visible = f"nuances {GROUP_NAMES[self.group]}"
        elif self.state is PaletteState.SHADES:
            visible = "menu de nuances (famille incertaine)"
        else:
            visible = "palette non reconnue"
        return f"{visible}, confiance {self.confidence:.0%}; action: {self.recovery.value}"


@dataclass(frozen=True, slots=True)
class _GroupMatch:
    group: int
    score: float
    median_delta_e: float
    matching: int
    sampled: int


class PaletteObserver:
    """Reconnaît la palette par plusieurs indices, sans effectuer de clic.

    Le bouton Retour indique qu'un menu de nuances est ouvert, tandis que les
    couleurs des dix cases identifient la famille. La palette principale est
    validée séparément sur ses treize familles connues. L'appelant peut donc
    distinguer une animation en cours, une mauvaise famille et une calibration
    des repères incorrecte.
    """

    def __init__(
        self,
        *,
        retry_limit: int = 3,
        shade_match_delta_e: float = 22.0,
    ) -> None:
        if retry_limit < 1:
            raise ValueError("retry_limit doit être positif")
        if shade_match_delta_e <= 0:
            raise ValueError("shade_match_delta_e doit être positif")
        self.retry_limit = retry_limit
        self.shade_match_delta_e = shade_match_delta_e

    def observe(
        self,
        rgb: np.ndarray,
        profile: CalibrationProfile,
        *,
        expected_group: int | None = None,
        attempt: int = 1,
    ) -> PaletteObservation:
        """Observe une image et recommande la prochaine action de navigation."""

        self._validate_image(rgb)
        self._validate_expected_group(expected_group)
        if attempt < 1:
            raise ValueError("attempt doit être positif")

        height, width = rgb.shape[:2]
        scaled = profile.scaled_to(width, height)
        back_score = self._back_button_score(rgb, scaled.back_button)
        main_score = self._main_palette_score(rgb, scaled)
        matches = self._group_matches(rgb, scaled)
        best = matches[0]
        second = matches[1]

        absolute_quality = max(0.0, 1.0 - best.score / 34.0)
        matching_ratio = best.matching / max(1, best.sampled)
        separation = max(0.0, min(1.0, (second.score - best.score) / 14.0))
        group_confidence = min(
            1.0,
            0.46 * absolute_quality + 0.34 * matching_ratio + 0.20 * separation,
        )
        shade_content_score = min(1.0, 0.58 * absolute_quality + 0.42 * matching_ratio)
        shade_score = min(1.0, 0.62 * back_score + 0.38 * shade_content_score)

        messages: list[str] = []
        if back_score >= 0.52:
            state = PaletteState.SHADES
            state_confidence = shade_score
            group = best.group if group_confidence >= 0.48 else None
            if group is None:
                messages.append(
                    "Le bouton Retour est visible, mais les nuances sont encore ambiguës."
                )
        elif main_score >= 0.60:
            state = PaletteState.MAIN
            state_confidence = main_score
            group = None
        else:
            state = PaletteState.UNKNOWN
            state_confidence = max(main_score, shade_score * 0.72)
            group = None
            if back_score < 0.32 and main_score < 0.40:
                messages.append("Ni le bouton Retour ni les couleurs principales ne sont fiables.")
            else:
                messages.append("L'interface semble être en transition ou partiellement masquée.")

        recovery = self.recommend_recovery(
            state=state,
            confidence=state_confidence,
            group=group,
            group_confidence=group_confidence,
            expected_group=expected_group,
            attempt=attempt,
        )
        if expected_group is not None and group is not None and group != expected_group:
            messages.append(
                f"Famille attendue: {GROUP_NAMES[expected_group]}; "
                f"famille visible: {GROUP_NAMES[group]}."
            )

        diagnostics = PaletteDiagnostics(
            back_button_score=back_score,
            main_palette_score=main_score,
            shade_palette_score=shade_score,
            best_group_delta_e=best.median_delta_e,
            second_group_delta_e=second.median_delta_e,
            matching_shades=best.matching,
            sampled_shades=best.sampled,
            messages=tuple(messages),
        )
        return PaletteObservation(
            state=state,
            confidence=state_confidence,
            group=group,
            group_confidence=group_confidence,
            expected_group=expected_group,
            recovery=recovery,
            diagnostics=diagnostics,
        )

    def observe_many(
        self,
        frames: Iterable[np.ndarray],
        profile: CalibrationProfile,
        *,
        expected_group: int | None = None,
        attempt: int = 1,
    ) -> PaletteObservation:
        """Combine plusieurs lectures afin d'ignorer une image d'animation.

        Trois images espacées de quelques dizaines de millisecondes conviennent
        bien. Aucun consensus n'est fabriqué si toutes les lectures sont faibles.
        """

        observations = [
            self.observe(frame, profile, expected_group=expected_group, attempt=attempt)
            for frame in frames
        ]
        if not observations:
            raise ValueError("Au moins une image est requise")
        if len(observations) == 1:
            return observations[0]

        state_weights = {
            state: sum(
                max(0.05, observation.confidence)
                for observation in observations
                if observation.state is state
            )
            for state in PaletteState
        }
        state = max(state_weights, key=state_weights.__getitem__)
        candidates = [observation for observation in observations if observation.state is state]
        representative = max(candidates, key=lambda observation: observation.confidence)

        group: int | None = None
        group_confidence = representative.group_confidence
        if state is PaletteState.SHADES:
            group_weights: dict[int, float] = {}
            for observation in candidates:
                if observation.group is not None:
                    group_weights[observation.group] = group_weights.get(
                        observation.group, 0.0
                    ) + max(0.05, observation.group_confidence)
            if group_weights:
                group = max(group_weights, key=group_weights.__getitem__)
                agreeing = [item for item in candidates if item.group == group]
                group_confidence = sum(item.group_confidence for item in agreeing) / len(agreeing)

        confidence = sum(item.confidence for item in candidates) / len(candidates)
        vote_ratio = len(candidates) / len(observations)
        confidence = min(1.0, 0.72 * confidence + 0.28 * vote_ratio)
        recovery = self.recommend_recovery(
            state=state,
            confidence=confidence,
            group=group,
            group_confidence=group_confidence,
            expected_group=expected_group,
            attempt=attempt,
        )
        messages = representative.diagnostics.messages + (
            f"Consensus sur {len(candidates)}/{len(observations)} lectures.",
        )
        diagnostics = replace(
            representative.diagnostics,
            sample_count=len(observations),
            messages=messages,
        )
        return replace(
            representative,
            state=state,
            confidence=confidence,
            group=group,
            group_confidence=group_confidence,
            recovery=recovery,
            diagnostics=diagnostics,
        )

    def recommend_recovery(
        self,
        *,
        state: PaletteState,
        confidence: float,
        group: int | None,
        group_confidence: float,
        expected_group: int | None,
        attempt: int,
    ) -> PaletteRecoveryAction:
        """Transforme une observation en décision de navigation sans ambiguïté."""

        self._validate_expected_group(expected_group)
        if attempt < 1:
            raise ValueError("attempt doit être positif")

        if expected_group is None:
            if state is PaletteState.MAIN and confidence >= 0.60:
                return PaletteRecoveryAction.READY
            if state is PaletteState.SHADES:
                return PaletteRecoveryAction.BACK_TO_MAIN
            return (
                PaletteRecoveryAction.RETRY
                if attempt < self.retry_limit
                else PaletteRecoveryAction.RECALIBRATE
            )

        if state is PaletteState.SHADES:
            if group == expected_group and group_confidence >= 0.48:
                return PaletteRecoveryAction.READY
            if group is not None and group != expected_group and group_confidence >= 0.58:
                return PaletteRecoveryAction.BACK_TO_MAIN
            return (
                PaletteRecoveryAction.RETRY
                if attempt < self.retry_limit
                else PaletteRecoveryAction.BACK_TO_MAIN
            )
        if state is PaletteState.MAIN:
            return PaletteRecoveryAction.REOPEN_SHADES
        return (
            PaletteRecoveryAction.RETRY
            if attempt < self.retry_limit
            else PaletteRecoveryAction.RECALIBRATE
        )

    def _group_matches(
        self,
        rgb: np.ndarray,
        profile: CalibrationProfile,
    ) -> list[_GroupMatch]:
        matches: list[_GroupMatch] = []
        for group in range(len(GROUP_NAMES)):
            expected = colors_for_group(group)
            observed = np.asarray(
                [
                    self._sample_color(
                        rgb,
                        profile.shade_point(color.shade_column, color.shade_row),
                    )
                    for color in expected
                ],
                dtype=np.uint8,
            )
            expected_rgb = np.asarray([color.rgb for color in expected], dtype=np.uint8)
            deltas = np.asarray(
                delta_e_ciede2000(srgb_to_lab(observed), srgb_to_lab(expected_rgb)),
                dtype=float,
            )
            ordered = np.sort(deltas)
            kept = ordered[: max(3, ceil(len(ordered) * 0.8))]
            median = float(np.median(deltas))
            score = 0.58 * median + 0.42 * float(np.mean(kept))
            matches.append(
                _GroupMatch(
                    group=group,
                    score=score,
                    median_delta_e=median,
                    matching=int(np.count_nonzero(deltas <= self.shade_match_delta_e)),
                    sampled=len(deltas),
                )
            )
        matches.sort(key=lambda match: match.score)
        return matches

    def _main_palette_score(self, rgb: np.ndarray, profile: CalibrationProfile) -> float:
        observed: list[np.ndarray] = []
        expected: list[tuple[int, int, int]] = []
        for group, (column, row) in enumerate(MAIN_CELLS):
            observed.append(self._sample_color(rgb, profile.main_color_point(column, row)))
            colors = colors_for_group(group)
            expected.append(colors[0 if group == 0 else 1].rgb)
        observed_lab = srgb_to_lab(np.asarray(observed, dtype=np.uint8))
        expected_lab = srgb_to_lab(np.asarray(expected, dtype=np.uint8))
        deltas = np.asarray(delta_e_ciede2000(observed_lab, expected_lab), dtype=float)
        matching_ratio = float(np.mean(deltas <= 20.0))
        median_quality = max(0.0, 1.0 - float(np.median(deltas)) / 42.0)
        return min(1.0, 0.72 * matching_ratio + 0.28 * median_quality)

    @staticmethod
    def _back_button_score(rgb: np.ndarray, point: Point) -> float:
        height, width = rgb.shape[:2]
        cx, cy = int(round(point.x)), int(round(point.y))
        x_radius = max(8, int(round(width * 0.055)))
        y_radius = max(6, int(round(height * 0.035)))
        roi = rgb[
            max(0, cy - y_radius) : min(height, cy + y_radius),
            max(0, cx - x_radius) : min(width, cx + x_radius),
            :3,
        ]
        if roi.size == 0:
            return 0.0
        red, green, blue = (roi[..., channel] for channel in range(3))
        salmon = (
            (red > 160) & (red < 235) & (green > 105) & (green < 195) & (blue > 85) & (blue < 180)
        )
        cream = (red > 220) & (green > 205) & (blue > 165)
        salmon_ratio = float(np.mean(salmon))
        cream_ratio = float(np.mean(cream))
        return min(1.0, salmon_ratio * 1.65 + cream_ratio * 3.0)

    @staticmethod
    def _sample_color(rgb: np.ndarray, point: Point) -> np.ndarray:
        height, width = rgb.shape[:2]
        radius = max(3, int(round(width / 320)))
        x, y = int(round(point.x)), int(round(point.y))
        patch = rgb[
            max(0, y - radius) : min(height, y + radius + 1),
            max(0, x - radius) : min(width, x + radius + 1),
            :3,
        ]
        if patch.size == 0:
            return np.zeros(3, dtype=np.uint8)
        return np.median(patch.reshape(-1, 3), axis=0).astype(np.uint8)

    @staticmethod
    def _validate_image(rgb: np.ndarray) -> None:
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError("Une capture RGB est requise")

    @staticmethod
    def _validate_expected_group(expected_group: int | None) -> None:
        if expected_group is not None and not 0 <= expected_group < len(GROUP_NAMES):
            raise ValueError("expected_group doit désigner une famille Heartopia")
