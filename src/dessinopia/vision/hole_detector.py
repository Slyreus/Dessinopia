from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dessinopia.core.color_spaces import delta_e_ciede2000, srgb_to_lab
from dessinopia.core.models import CalibrationProfile
from dessinopia.core.palette import PALETTE_RGB
from dessinopia.core.path_planner import Cell

_CANVAS_BACKGROUND_RGB = np.asarray(
    (
        (249, 245, 238),
        (255, 251, 245),
        (247, 243, 235),
    ),
    dtype=np.uint8,
)
_CANVAS_BACKGROUND_LAB = srgb_to_lab(_CANVAS_BACKGROUND_RGB)
_PALETTE_LAB = srgb_to_lab(PALETTE_RGB)
_BACKGROUND_MARGIN = 3.0


@dataclass(frozen=True, slots=True)
class HoleAudit:
    """Missing or visibly different cells on the calibrated canvas."""

    cells: tuple[Cell, ...]
    sampled_cells: int
    closest_background_delta: tuple[float, ...]
    wrong_color_cells: tuple[Cell, ...] = ()

    @property
    def hole_count(self) -> int:
        return len(self.cells)

    @property
    def repair_count(self) -> int:
        return self.hole_count + len(self.wrong_color_cells)


def audit_canvas_holes(
    rgb: np.ndarray,
    profile: CalibrationProfile,
    target_indices: np.ndarray,
    *,
    background_delta_threshold: float = 10.0,
    color_delta_threshold: float = 10.0,
) -> HoleAudit:
    """Compare each opaque target cell with the canvas background and its intended shade."""

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Une capture RGB est requise")
    if target_indices.ndim != 2 or not np.issubdtype(target_indices.dtype, np.integer):
        raise ValueError("La cible doit être une matrice d'indices entiers")
    if background_delta_threshold <= 0 or not np.isfinite(background_delta_threshold):
        raise ValueError("Le seuil de fond doit être positif")
    if color_delta_threshold <= 0 or not np.isfinite(color_delta_threshold):
        raise ValueError("Le seuil de couleur doit être positif")

    height, width = target_indices.shape
    if width <= 0 or height <= 0:
        return HoleAudit((), 0, ())
    scaled = profile.scaled_to(image.shape[1], image.shape[0])
    canvas = scaled.canvas
    if (
        canvas.left < 0
        or canvas.top < 0
        or canvas.right > image.shape[1]
        or canvas.bottom > image.shape[0]
    ):
        raise ValueError("La toile calibrée sort de la capture")

    rows, columns = np.nonzero(target_indices >= 0)
    if not len(rows):
        return HoleAudit((), 0, ())
    palette_indices = target_indices[rows, columns]
    if np.any(palette_indices >= len(_PALETTE_LAB)):
        raise ValueError("La cible contient un indice de palette inconnu")
    cell_width = canvas.width / width
    cell_height = canvas.height / height
    radius_x = max(1, int(cell_width * 0.12)) if cell_width >= 4 else 0
    radius_y = max(1, int(cell_height * 0.12)) if cell_height >= 4 else 0
    x = np.rint(canvas.left + (columns + 0.5) * cell_width).astype(int)
    y = np.rint(canvas.top + (rows + 0.5) * cell_height).astype(int)
    patch_x = np.clip(x[:, None] + np.arange(-radius_x, radius_x + 1), 0, image.shape[1] - 1)
    patch_y = np.clip(y[:, None] + np.arange(-radius_y, radius_y + 1), 0, image.shape[0] - 1)
    patches = image[patch_y[:, :, None], patch_x[:, None, :], :3]
    samples = srgb_to_lab(np.rint(np.median(patches, axis=(1, 2))).astype(np.uint8))
    background_delta = np.min(
        delta_e_ciede2000(samples[:, None, :], _CANVAS_BACKGROUND_LAB),
        axis=1,
    )
    target_delta = delta_e_ciede2000(samples, _PALETTE_LAB[palette_indices])
    near_background = background_delta <= background_delta_threshold
    holes = near_background & (target_delta > background_delta + _BACKGROUND_MARGIN)
    wrong_colors = ~near_background & (target_delta > color_delta_threshold)

    def selected_cells(mask: np.ndarray) -> tuple[Cell, ...]:
        return tuple(
            Cell(int(column), int(row))
            for row, column in zip(rows[mask], columns[mask], strict=True)
        )

    return HoleAudit(
        selected_cells(holes),
        len(rows),
        tuple(float(delta) for delta in background_delta[holes]),
        selected_cells(wrong_colors),
    )


def repair_indices(target_indices: np.ndarray, audit: HoleAudit) -> np.ndarray:
    """Return a sparse image containing only cells selected by an audit."""

    result = np.full(target_indices.shape, -1, dtype=np.int16)
    for cell in (*audit.cells, *audit.wrong_color_cells):
        result[cell.row, cell.column] = int(target_indices[cell.row, cell.column])
    return result
