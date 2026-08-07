from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from dessinopia.core.palette import (
    MAIN_CELLS,
    PALETTE_BY_INDEX,
    TRANSPARENT_SHADE_CELL,
    colors_for_group,
)
from dessinopia.core.path_planner import Cell


class ProbeRole(StrEnum):
    """Emplacement logique d'une sonde du test de toile."""

    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    CENTER = "center"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


@dataclass(frozen=True, slots=True)
class CalibrationProbe:
    """Pixel de test, exprimé dans les coordonnées logiques de la toile."""

    role: ProbeRole
    cell: Cell
    palette_index: int

    @property
    def group(self) -> int:
        return PALETTE_BY_INDEX[self.palette_index].group


@dataclass(frozen=True, slots=True)
class TransparencySelection:
    """Chemin de navigation vers la transparence de la palette Heartopia."""

    group: int
    main_column: int
    main_row: int
    shade_column: int
    shade_row: int


@dataclass(frozen=True, slots=True)
class CalibrationCleanup:
    """Sélection et cellules nécessaires pour effacer les sondes validées."""

    selection: TransparencySelection
    cells: tuple[Cell, ...]


@dataclass(frozen=True, slots=True)
class CalibrationTestPattern:
    """Motif de calibration prêt à être converti en ``PaintingPlan``."""

    width: int
    height: int
    probes: tuple[CalibrationProbe, ...]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("La grille de calibration doit avoir une taille positive")

        cells = [probe.cell for probe in self.probes]
        if len(set(cells)) != len(cells):
            raise ValueError("Deux sondes de calibration ne peuvent pas partager une cellule")
        if len({probe.group for probe in self.probes}) != len(self.probes):
            raise ValueError("Chaque sonde de calibration doit utiliser une famille distincte")
        if any(
            probe.cell.column < 0
            or probe.cell.column >= self.width
            or probe.cell.row < 0
            or probe.cell.row >= self.height
            for probe in self.probes
        ):
            raise ValueError("Une sonde de calibration se trouve hors de la grille")

    @property
    def complete(self) -> bool:
        """Indique si les cinq emplacements existent comme cellules distinctes."""

        return len(self.probes) == len(ProbeRole)

    @property
    def cleanup(self) -> CalibrationCleanup:
        return CalibrationCleanup(TRANSPARENCY_SELECTION, tuple(p.cell for p in self.probes))

    def to_index_matrix(self) -> np.ndarray:
        """Retourne une matrice directement acceptée par ``build_painting_plan``."""

        indices = np.full((self.height, self.width), -1, dtype=np.int16)
        for probe in self.probes:
            indices[probe.cell.row, probe.cell.column] = probe.palette_index
        return indices


_PROBE_GROUPS: dict[ProbeRole, int] = {
    ProbeRole.TOP_LEFT: 1,  # rouge
    ProbeRole.TOP_RIGHT: 4,  # jaune
    ProbeRole.CENTER: 6,  # vert
    ProbeRole.BOTTOM_LEFT: 9,  # bleu
    ProbeRole.BOTTOM_RIGHT: 11,  # violet
}

# Le centre est prioritaire lors d'une collision sur une grille minuscule. Les
# grilles Heartopia normales possèdent toujours cinq cellules distinctes.
_COLLISION_PRIORITY = (
    ProbeRole.CENTER,
    ProbeRole.TOP_LEFT,
    ProbeRole.TOP_RIGHT,
    ProbeRole.BOTTOM_LEFT,
    ProbeRole.BOTTOM_RIGHT,
)
_OUTPUT_ORDER = (
    ProbeRole.TOP_LEFT,
    ProbeRole.TOP_RIGHT,
    ProbeRole.CENTER,
    ProbeRole.BOTTOM_LEFT,
    ProbeRole.BOTTOM_RIGHT,
)

_neutral_main_column, _neutral_main_row = MAIN_CELLS[0]
_transparent_column, _transparent_row = TRANSPARENT_SHADE_CELL
TRANSPARENCY_SELECTION = TransparencySelection(
    group=0,
    main_column=_neutral_main_column,
    main_row=_neutral_main_row,
    shade_column=_transparent_column,
    shade_row=_transparent_row,
)


def build_five_point_pattern(width: int, height: int) -> CalibrationTestPattern:
    """Construit les quatre coins et le centre dans cinq familles très distinctes.

    Pour une grille artificiellement plus petite que cinq emplacements distincts,
    les collisions sont supprimées au lieu de peindre deux fois la même cellule.
    """

    if width <= 0 or height <= 0:
        raise ValueError("La grille de calibration doit avoir une taille positive")

    positions = {
        ProbeRole.TOP_LEFT: Cell(0, 0),
        ProbeRole.TOP_RIGHT: Cell(width - 1, 0),
        ProbeRole.CENTER: Cell(width // 2, height // 2),
        ProbeRole.BOTTOM_LEFT: Cell(0, height - 1),
        ProbeRole.BOTTOM_RIGHT: Cell(width - 1, height - 1),
    }

    selected_roles: set[ProbeRole] = set()
    occupied: set[Cell] = set()
    for role in _COLLISION_PRIORITY:
        cell = positions[role]
        if cell not in occupied:
            selected_roles.add(role)
            occupied.add(cell)

    probes = tuple(
        CalibrationProbe(
            role=role,
            cell=positions[role],
            palette_index=colors_for_group(_PROBE_GROUPS[role])[0].index,
        )
        for role in _OUTPUT_ORDER
        if role in selected_roles
    )
    return CalibrationTestPattern(width=width, height=height, probes=probes)
