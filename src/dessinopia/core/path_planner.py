from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dessinopia.core.palette import PALETTE_BY_INDEX


@dataclass(frozen=True, slots=True)
class Cell:
    column: int
    row: int


@dataclass(frozen=True, slots=True)
class Stroke:
    palette_index: int
    start: Cell
    end: Cell

    @property
    def length(self) -> int:
        return abs(self.end.column - self.start.column) + 1


@dataclass(frozen=True, slots=True)
class ShadePass:
    palette_index: int
    strokes: tuple[Stroke, ...]

    @property
    def pixel_count(self) -> int:
        return sum(stroke.length for stroke in self.strokes)


@dataclass(frozen=True, slots=True)
class FamilyPass:
    group: int
    shades: tuple[ShadePass, ...]


@dataclass(frozen=True, slots=True)
class PaintingPlan:
    width: int
    height: int
    families: tuple[FamilyPass, ...]

    @property
    def pixel_count(self) -> int:
        return sum(shade.pixel_count for family in self.families for shade in family.shades)

    @property
    def stroke_count(self) -> int:
        return sum(len(shade.strokes) for family in self.families for shade in family.shades)

    @property
    def shade_count(self) -> int:
        return sum(len(family.shades) for family in self.families)


def _row_runs(mask: np.ndarray, row: int, palette_index: int, reverse: bool) -> list[Stroke]:
    xs = np.flatnonzero(mask[row])
    if not len(xs):
        return []
    boundaries = np.where(np.diff(xs) > 1)[0]
    starts = np.r_[0, boundaries + 1]
    ends = np.r_[boundaries, len(xs) - 1]
    runs: list[Stroke] = []
    for start_index, end_index in zip(starts, ends, strict=True):
        left = int(xs[start_index])
        right = int(xs[end_index])
        start, end = (right, left) if reverse else (left, right)
        runs.append(Stroke(palette_index, Cell(start, row), Cell(end, row)))
    if reverse:
        runs.reverse()
    return runs


def build_painting_plan(indices: np.ndarray) -> PaintingPlan:
    if indices.ndim != 2:
        raise ValueError("La matrice de pixels doit avoir deux dimensions")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("La matrice de pixels doit contenir des indices entiers")
    if indices.size and (np.any(indices < -1) or np.any(indices >= len(PALETTE_BY_INDEX))):
        raise ValueError("La matrice contient un indice de palette invalide")
    height, width = indices.shape
    opaque_indices = np.unique(indices[indices >= 0]).astype(int)
    by_group: dict[int, list[int]] = {}
    for palette_index in opaque_indices:
        by_group.setdefault(PALETTE_BY_INDEX[int(palette_index)].group, []).append(
            int(palette_index)
        )

    family_passes: list[FamilyPass] = []
    for group in sorted(by_group):
        # Les teintes les plus utilisées sont peintes d'abord au sein d'une famille.
        shade_indices = sorted(
            by_group[group],
            key=lambda value: (
                -int(np.count_nonzero(indices == value)),
                PALETTE_BY_INDEX[value].shade,
            ),
        )
        shade_passes: list[ShadePass] = []
        for palette_index in shade_indices:
            mask = indices == palette_index
            strokes: list[Stroke] = []
            for row in range(height):
                strokes.extend(_row_runs(mask, row, palette_index, reverse=bool(row % 2)))
            shade_passes.append(ShadePass(palette_index, tuple(strokes)))
        family_passes.append(FamilyPass(group, tuple(shade_passes)))
    return PaintingPlan(width, height, tuple(family_passes))
