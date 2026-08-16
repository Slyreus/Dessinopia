from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from numbers import Real

import numpy as np

from dessinopia.core.color_spaces import delta_e_ciede2000, srgb_to_lab
from dessinopia.core.fill_planner import FillOperation
from dessinopia.core.models import CalibrationProfile
from dessinopia.core.palette import PALETTE_BY_INDEX
from dessinopia.core.path_planner import Cell, Stroke

_LOCAL_THRESHOLD_FLOOR = 2.0
_LOCAL_THRESHOLD_CEILING = 8.0
_SOURCE_SEPARATION_RATIO = 0.45
_EXPECTED_SHADE_MARGIN = 0.75

# Heartopia draws an ivory, diagonally striped background. These references are
# sampled from the supplied empty canvases. They are used only to reject an
# intrinsically ambiguous replay (not to decide whether a normal fill is ready).
_CANVAS_BACKGROUND_RGB = np.asarray(
    (
        (249, 245, 238),
        (255, 251, 245),
        (247, 243, 235),
    ),
    dtype=np.uint8,
)
_CANVAS_BACKGROUND_LAB = srgb_to_lab(_CANVAS_BACKGROUND_RGB)
_PALETTE_LAB_BY_INDEX = {
    index: srgb_to_lab(np.asarray(color.rgb, dtype=np.uint8))
    for index, color in PALETTE_BY_INDEX.items()
}


class FillCheckState(StrEnum):
    READY = "frontière fermée"
    ALREADY_FILLED = "déjà rempli"
    SUCCESS = "remplissage vérifié"
    FALLBACK = "repli crayon"
    INCOMPLETE = "remplissage incomplet"
    OVERFLOW = "débordement détecté"
    UNKNOWN = "contrôle visuel incertain"


@dataclass(frozen=True, slots=True)
class FillCheck:
    state: FillCheckState
    confidence: float
    fence_matches: int
    fence_total: int
    expected_matches: int
    expected_total: int
    guard_matches: int
    guard_total: int

    @property
    def summary(self) -> str:
        return (
            f"{self.state.value}, confiance {self.confidence:.0%}; "
            f"frontière {self.fence_matches}/{self.fence_total}, "
            f"zone {self.expected_matches}/{self.expected_total}, "
            f"sentinelles touchées {self.guard_matches}/{self.guard_total}"
        )


@dataclass(frozen=True, slots=True)
class _Samples:
    rgb: np.ndarray
    lab: np.ndarray


@dataclass(frozen=True, slots=True)
class _FillContext:
    operation_key: tuple[int, int, int, int, int]
    target_lab: np.ndarray
    match_threshold: float


class FillCanvasVerifier:
    """Verify one physical bucket operation against the calibrated canvas.

    The already-painted fence is the best observation of how the selected
    colour is rendered on the canvas. Matching against that empirical colour
    avoids confusing palette RGB values, lighting and the ivory canvas. Every
    fence and guard cell is still checked: uncertainty causes a pencil fallback,
    never an optimistic bucket click.

    ``check_before`` stores a short-lived baseline for the corresponding
    ``check_after`` call. This lets the post-check distinguish a newly painted
    guard (a real overflow) from an unrelated but visually close shade.
    """

    def __init__(
        self,
        *,
        delta_e_threshold: float = 18.0,
        background_ambiguity_threshold: float = 16.0,
    ) -> None:
        if (
            isinstance(delta_e_threshold, bool)
            or not isinstance(delta_e_threshold, Real)
            or not np.isfinite(delta_e_threshold)
            or delta_e_threshold <= 0
        ):
            raise ValueError("Le seuil colorimétrique doit être positif")
        if (
            isinstance(background_ambiguity_threshold, bool)
            or not isinstance(background_ambiguity_threshold, Real)
            or not np.isfinite(background_ambiguity_threshold)
            or background_ambiguity_threshold <= 0
        ):
            raise ValueError("Le seuil d'ambiguïté doit être positif")
        self.delta_e_threshold = float(delta_e_threshold)
        self.background_ambiguity_threshold = float(background_ambiguity_threshold)
        self._context: _FillContext | None = None

    def check_before(
        self,
        rgb: np.ndarray,
        profile: CalibrationProfile,
        width: int,
        height: int,
        operation: FillOperation,
    ) -> FillCheck:
        image, fence, expected, guards, palette_target = self._sample_operation(
            rgb,
            profile,
            width,
            height,
            operation,
        )
        del image  # Validation and sampling are complete.
        target_lab, threshold, target_valid, source_gap = self._target_model(
            fence,
            expected,
            palette_target,
            operation.palette_index,
        )
        pale_replay_is_ambiguous = (
            source_gap <= threshold
            and self._distance_to_canvas_background(palette_target)
            <= self.background_ambiguity_threshold
        )
        fence_mask = self._matches(fence.lab, target_lab, threshold)
        expected_mask = self._matches(expected.lab, target_lab, threshold)
        guard_mask = self._matches(guards.lab, target_lab, threshold)
        if not target_valid or pale_replay_is_ambiguous:
            fence_mask[:] = False
            expected_mask[:] = False
            guard_mask[:] = False

        fence_matches = int(np.count_nonzero(fence_mask))
        expected_matches = int(np.count_nonzero(expected_mask))
        guard_matches = int(np.count_nonzero(guard_mask))
        fence_total = len(operation.required_fence)
        expected_total = len(operation.expected_cells)
        guard_total = len(operation.guard_cells)
        fence_closed = target_valid and fence_matches == fence_total
        guards_clear = guard_matches == 0

        self._context = None
        if target_valid and guards_clear:
            # Preserve the empirical fence colour even when this operation must
            # fall back to the pencil (partial replay or a missing fence cell).
            # The runner can then verify its full repair, including pale shades
            # that cannot be certified safely from a cold post-check.
            self._context = _FillContext(
                self._operation_key(operation),
                target_lab,
                threshold,
            )
        if guard_matches:
            state = FillCheckState.OVERFLOW
        elif (
            fence_closed
            and guards_clear
            and expected_matches == expected_total
            and not pale_replay_is_ambiguous
        ):
            state = FillCheckState.ALREADY_FILLED
        elif fence_closed and guards_clear and expected_matches == 0:
            state = FillCheckState.READY
        else:
            state = FillCheckState.FALLBACK

        expected_good = (
            expected_matches
            if state is FillCheckState.ALREADY_FILLED
            else expected_total - expected_matches
        )
        confidence = self._confidence(
            fence_matches,
            fence_total,
            expected_good,
            expected_total,
            guard_total - guard_matches,
            guard_total,
        )
        return FillCheck(
            state,
            confidence,
            fence_matches,
            fence_total,
            expected_matches,
            expected_total,
            guard_matches,
            guard_total,
        )

    def check_after(
        self,
        rgb: np.ndarray,
        profile: CalibrationProfile,
        width: int,
        height: int,
        operation: FillOperation,
    ) -> FillCheck:
        image, fence, expected, guards, palette_target = self._sample_operation(
            rgb,
            profile,
            width,
            height,
            operation,
        )
        del image
        context = self._context
        context_matches = (
            context is not None and context.operation_key == self._operation_key(operation)
        )
        if context_matches:
            assert context is not None
            target_lab = context.target_lab
            threshold = context.match_threshold
            target_valid = True
        else:
            target_lab, threshold, target_valid, source_gap = self._target_model(
                fence,
                expected,
                palette_target,
                operation.palette_index,
            )
            # A cold post-check cannot distinguish a completed very pale fill
            # from the original canvas. Refuse to certify it as successful.
            if (
                source_gap <= threshold
                and self._distance_to_canvas_background(palette_target)
                <= self.background_ambiguity_threshold
            ):
                target_valid = False

        fence_mask = self._matches(fence.lab, target_lab, threshold)
        expected_mask = self._matches(expected.lab, target_lab, threshold)
        guard_mask = self._matches(guards.lab, target_lab, threshold)
        if not target_valid:
            fence_mask[:] = False
            expected_mask[:] = False

        fence_matches = int(np.count_nonzero(fence_mask))
        expected_matches = int(np.count_nonzero(expected_mask))
        guard_matches = int(np.count_nonzero(guard_mask))
        fence_total = len(operation.required_fence)
        expected_total = len(operation.expected_cells)
        guard_total = len(operation.guard_cells)

        # BEFORE only stores a context when every guard is clear under this
        # exact empirical target model. Any matching guard afterwards is thus a
        # conservative leak signal; no secondary delta may mask the transition.
        overflow = guard_matches > 0

        if overflow:
            state = FillCheckState.OVERFLOW
        elif not target_valid:
            state = FillCheckState.UNKNOWN
        elif expected_matches == expected_total and fence_matches == fence_total:
            state = FillCheckState.SUCCESS
        elif fence_matches == fence_total:
            state = FillCheckState.INCOMPLETE
        else:
            state = FillCheckState.UNKNOWN

        confidence = self._confidence(
            fence_matches,
            fence_total,
            expected_matches,
            expected_total,
            guard_total - guard_matches,
            guard_total,
        )
        return FillCheck(
            state,
            confidence,
            fence_matches,
            fence_total,
            expected_matches,
            expected_total,
            guard_matches,
            guard_total,
        )

    def _sample_operation(
        self,
        rgb: np.ndarray,
        profile: CalibrationProfile,
        width: int,
        height: int,
        operation: FillOperation,
    ) -> tuple[np.ndarray, _Samples, _Samples, _Samples, np.ndarray]:
        image = self._validate_inputs(rgb, profile, width, height, operation)
        scaled = profile.scaled_to(image.shape[1], image.shape[0])
        fence = self._sample_cells(
            image,
            scaled,
            width,
            height,
            operation.required_fence,
        )
        expected = self._sample_cells(
            image,
            scaled,
            width,
            height,
            operation.expected_cells,
        )
        guards = self._sample_cells(
            image,
            scaled,
            width,
            height,
            operation.guard_cells,
        )
        palette_target = np.asarray(
            PALETTE_BY_INDEX[operation.palette_index].rgb,
            dtype=np.uint8,
        )
        return image, fence, expected, guards, palette_target

    def _validate_inputs(
        self,
        rgb: np.ndarray,
        profile: CalibrationProfile,
        width: int,
        height: int,
        operation: FillOperation,
    ) -> np.ndarray:
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Une capture RGB est requise")
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError("La capture RGB ne peut pas être vide")
        if not np.issubdtype(image.dtype, np.number) or np.issubdtype(
            image.dtype,
            np.complexfloating,
        ):
            raise ValueError("La capture RGB doit être numérique")
        channels = np.asarray(image[..., :3], dtype=np.float64)
        if not np.all(np.isfinite(channels)) or np.any(channels < 0) or np.any(channels > 255):
            raise ValueError("Les composantes RGB doivent être comprises entre 0 et 255")
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
        ):
            raise ValueError("La grille doit avoir une taille positive")
        if (
            isinstance(operation.palette_index, bool)
            or operation.palette_index not in PALETTE_BY_INDEX
        ):
            raise ValueError("L'opération utilise une couleur de palette inconnue")
        if not operation.dual_connectivity_verified:
            raise ValueError("Le remplissage doit être sûr en connexité 4 et 8")
        if not operation.required_fence or not operation.expected_cells:
            raise ValueError("Un remplissage exige une frontière et une zone intérieure")
        if operation.seed not in operation.expected_cells:
            raise ValueError("La graine doit appartenir à la zone attendue")

        groups = (
            operation.required_fence,
            operation.expected_cells,
            operation.guard_cells,
        )
        coordinate_groups = [
            {(cell.column, cell.row) for cell in group}
            for group in groups
        ]
        if any(
            isinstance(cell.column, bool)
            or isinstance(cell.row, bool)
            or not isinstance(cell.column, int)
            or not isinstance(cell.row, int)
            for group in groups
            for cell in group
        ):
            raise ValueError("Les coordonnées de cellule doivent être entières")
        if any(
            len(group) != len(coordinates)
            for group, coordinates in zip(groups, coordinate_groups, strict=True)
        ):
            raise ValueError("Les cellules de contrôle doivent être uniques")
        if (
            coordinate_groups[0] & coordinate_groups[1]
            or coordinate_groups[0] & coordinate_groups[2]
            or coordinate_groups[1] & coordinate_groups[2]
        ):
            raise ValueError("La frontière, la zone et les sentinelles doivent être disjointes")
        if any(
            not (0 <= column < width and 0 <= row < height)
            for coordinates in coordinate_groups
            for column, row in coordinates
        ):
            raise ValueError("Une cellule de contrôle est hors de la grille")

        expected_coordinates = coordinate_groups[1]
        fence_coordinates, fence_count = self._stroke_coordinates(
            operation.fence_strokes,
            operation.palette_index,
            "frontière",
        )
        if (
            not operation.fence_strokes
            or fence_count != len(fence_coordinates)
            or fence_coordinates != coordinate_groups[0]
        ):
            raise ValueError("Les traits de frontière doivent couvrir exactement le contour")
        fallback_coordinates, fallback_count = self._stroke_coordinates(
            operation.fallback_strokes,
            operation.palette_index,
            "secours",
        )
        if (
            not operation.fallback_strokes
            or fallback_count != len(fallback_coordinates)
            or fallback_coordinates != expected_coordinates
        ):
            raise ValueError("Les traits de secours doivent couvrir exactement la zone")
        repair_coordinates, repair_count = self._stroke_coordinates(
            operation.repair_strokes,
            operation.palette_index,
            "réparation",
        )
        repair_target = expected_coordinates | coordinate_groups[0]
        if (
            not operation.repair_strokes
            or repair_count != len(repair_coordinates)
            or repair_coordinates != repair_target
        ):
            raise ValueError(
                "Les traits de réparation doivent couvrir la frontière et la zone"
            )

        if profile.reference_width <= 0 or profile.reference_height <= 0:
            raise ValueError("La référence de calibration doit avoir une taille positive")
        scaled = profile.scaled_to(image.shape[1], image.shape[0])
        canvas = scaled.canvas
        epsilon = 1e-6
        if (
            not all(
                np.isfinite(value)
                for value in (
                    canvas.left,
                    canvas.top,
                    canvas.width,
                    canvas.height,
                )
            )
            or canvas.width <= 0
            or canvas.height <= 0
            or canvas.left < -epsilon
            or canvas.top < -epsilon
            or canvas.right > image.shape[1] + epsilon
            or canvas.bottom > image.shape[0] + epsilon
        ):
            raise ValueError("La toile calibrée sort de la capture")
        return np.rint(channels).astype(np.uint8)

    @staticmethod
    def _stroke_coordinates(
        strokes: tuple[Stroke, ...],
        palette_index: int,
        label: str,
    ) -> tuple[set[tuple[int, int]], int]:
        coordinates: set[tuple[int, int]] = set()
        count = 0
        for stroke in strokes:
            if stroke.palette_index != palette_index:
                raise ValueError(f"Les traits de {label} utilisent une autre couleur")
            covered = {
                (cell.column, cell.row)
                for cell in stroke.covered_cells
            }
            coordinates.update(covered)
            count += len(covered)
        return coordinates, count

    @staticmethod
    def _sample_cells(
        rgb: np.ndarray,
        profile: CalibrationProfile,
        width: int,
        height: int,
        cells: tuple[Cell, ...],
    ) -> _Samples:
        if not cells:
            empty_rgb = np.empty((0, 3), dtype=np.uint8)
            return _Samples(empty_rgb, srgb_to_lab(empty_rgb))
        radius_x = max(1, int(profile.canvas.width / width * 0.12))
        radius_y = max(1, int(profile.canvas.height / height * 0.12))
        samples: list[np.ndarray] = []
        for cell in cells:
            point = profile.canvas.cell_center(cell.column, cell.row, width, height)
            x, y = int(round(point.x)), int(round(point.y))
            patch = rgb[
                max(0, y - radius_y) : min(rgb.shape[0], y + radius_y + 1),
                max(0, x - radius_x) : min(rgb.shape[1], x + radius_x + 1),
            ]
            if patch.size == 0:
                raise ValueError("Une cellule calibrée sort de la capture")
            samples.append(np.rint(np.median(patch.reshape(-1, 3), axis=0)).astype(np.uint8))
        sampled_rgb = np.asarray(samples, dtype=np.uint8)
        return _Samples(sampled_rgb, srgb_to_lab(sampled_rgb))

    def _target_model(
        self,
        fence: _Samples,
        expected: _Samples,
        palette_target: np.ndarray,
        palette_index: int,
    ) -> tuple[np.ndarray, float, bool, float]:
        target_rgb = np.rint(np.median(fence.rgb, axis=0)).astype(np.uint8)
        target_lab = srgb_to_lab(target_rgb)
        palette_lab = _PALETTE_LAB_BY_INDEX[palette_index]
        target_delta = float(delta_e_ciede2000(target_lab, palette_lab))
        expected_color = PALETTE_BY_INDEX[palette_index]
        competing_deltas = (
            float(delta_e_ciede2000(target_lab, other_lab))
            for other_index, other_lab in _PALETTE_LAB_BY_INDEX.items()
            if other_index != palette_index
            and PALETTE_BY_INDEX[other_index].group == expected_color.group
        )
        closest_competitor = min(competing_deltas, default=float("inf"))
        target_valid = (
            target_delta <= self.delta_e_threshold
            and target_delta + _EXPECTED_SHADE_MARGIN < closest_competitor
        )

        fence_deltas = np.asarray(
            delta_e_ciede2000(fence.lab, target_lab),
            dtype=float,
        )
        local_threshold = min(
            self.delta_e_threshold,
            _LOCAL_THRESHOLD_CEILING,
            max(
                _LOCAL_THRESHOLD_FLOOR,
                float(np.percentile(fence_deltas, 75)) + 1.0,
            ),
        )
        source_rgb = np.rint(np.median(expected.rgb, axis=0)).astype(np.uint8)
        source_lab = srgb_to_lab(source_rgb)
        source_gap = float(delta_e_ciede2000(target_lab, source_lab))
        if source_gap > local_threshold:
            local_threshold = min(
                local_threshold,
                max(0.5, source_gap * _SOURCE_SEPARATION_RATIO),
            )
        return target_lab, local_threshold, target_valid, source_gap

    @staticmethod
    def _matches(lab: np.ndarray, target_lab: np.ndarray, threshold: float) -> np.ndarray:
        if not len(lab):
            return np.zeros(0, dtype=bool)
        deltas = np.asarray(delta_e_ciede2000(lab, target_lab), dtype=float)
        return deltas <= threshold

    @staticmethod
    def _operation_key(operation: FillOperation) -> tuple[int, int, int, int, int]:
        return (
            operation.component_id,
            operation.palette_index,
            operation.seed.column,
            operation.seed.row,
            len(operation.expected_cells),
        )

    @staticmethod
    def _distance_to_canvas_background(palette_target: np.ndarray) -> float:
        target_lab = srgb_to_lab(palette_target)
        distances = np.asarray(
            delta_e_ciede2000(_CANVAS_BACKGROUND_LAB, target_lab),
            dtype=float,
        )
        return float(np.min(distances))

    @staticmethod
    def _confidence(*values: int) -> float:
        ratios = [
            matched / total if total else 1.0
            for matched, total in zip(values[::2], values[1::2], strict=True)
        ]
        return float(min(ratios))
