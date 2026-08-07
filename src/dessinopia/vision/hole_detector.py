from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dessinopia.core.color_spaces import delta_e_ciede2000, srgb_to_lab
from dessinopia.core.models import CalibrationProfile
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


@dataclass(frozen=True, slots=True)
class HoleAudit:
    """Opaque target cells whose observed centre still resembles the canvas."""

    cells: tuple[Cell, ...]
    sampled_cells: int
    closest_background_delta: tuple[float, ...]

    @property
    def hole_count(self) -> int:
        return len(self.cells)


def audit_canvas_holes(
    rgb: np.ndarray,
    profile: CalibrationProfile,
    target_indices: np.ndarray,
    *,
    background_delta_threshold: float = 10.0,
) -> HoleAudit:
    """Find unpainted cells conservatively from Heartopia's striped background.

    The test intentionally accepts false positives for very pale intended
    colours: repainting a cell with its intended shade is idempotent, while
    overlooking an ivory hole would leave a visible defect.
    """

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Une capture RGB est requise")
    if target_indices.ndim != 2 or not np.issubdtype(target_indices.dtype, np.integer):
        raise ValueError("La cible doit être une matrice d'indices entiers")
    if background_delta_threshold <= 0 or not np.isfinite(background_delta_threshold):
        raise ValueError("Le seuil de fond doit être positif")

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

    radius_x = max(1, int(canvas.width / width * 0.12))
    radius_y = max(1, int(canvas.height / height * 0.12))
    cells: list[Cell] = []
    deltas: list[float] = []
    sampled = 0
    for row, column in np.argwhere(target_indices >= 0):
        point = canvas.cell_center(int(column), int(row), width, height)
        x, y = int(round(point.x)), int(round(point.y))
        patch = image[
            max(0, y - radius_y) : min(image.shape[0], y + radius_y + 1),
            max(0, x - radius_x) : min(image.shape[1], x + radius_x + 1),
            :3,
        ]
        if patch.size == 0:
            raise ValueError("Une cellule calibrée sort de la capture")
        sample = np.rint(np.median(patch.reshape(-1, 3), axis=0)).astype(np.uint8)
        sample_lab = srgb_to_lab(sample)
        closest = float(np.min(delta_e_ciede2000(_CANVAS_BACKGROUND_LAB, sample_lab)))
        sampled += 1
        if closest <= background_delta_threshold:
            cells.append(Cell(int(column), int(row)))
            deltas.append(closest)
    return HoleAudit(tuple(cells), sampled, tuple(deltas))


def repair_indices(target_indices: np.ndarray, audit: HoleAudit) -> np.ndarray:
    """Return a sparse image containing only cells selected by an audit."""

    result = np.full(target_indices.shape, -1, dtype=np.int16)
    for cell in audit.cells:
        result[cell.row, cell.column] = int(target_indices[cell.row, cell.column])
    return result
