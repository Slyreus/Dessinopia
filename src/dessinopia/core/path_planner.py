from __future__ import annotations

from collections import deque
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
    trace: tuple[Cell, ...] = ()

    def __post_init__(self) -> None:
        cells = self.trace_cells
        if not cells:
            raise ValueError("Un trait doit contenir au moins une cellule")
        if cells[0] != self.start or cells[-1] != self.end:
            raise ValueError("Les extrémités du trait ne correspondent pas à son parcours")
        if any(
            abs(first.column - second.column) + abs(first.row - second.row) != 1
            for first, second in zip(cells, cells[1:], strict=False)
        ):
            raise ValueError("Un trait continu doit relier des cellules adjacentes")

    @classmethod
    def from_trace(cls, palette_index: int, trace: tuple[Cell, ...]) -> Stroke:
        if not trace:
            raise ValueError("Un trait doit contenir au moins une cellule")
        return cls(palette_index, trace[0], trace[-1], trace)

    @property
    def trace_cells(self) -> tuple[Cell, ...]:
        if self.trace:
            return self.trace
        if self.start.row == self.end.row:
            step = 1 if self.end.column >= self.start.column else -1
            return tuple(
                Cell(column, self.start.row)
                for column in range(self.start.column, self.end.column + step, step)
            )
        if self.start.column == self.end.column:
            step = 1 if self.end.row >= self.start.row else -1
            return tuple(
                Cell(self.start.column, row)
                for row in range(self.start.row, self.end.row + step, step)
            )
        raise ValueError("Un trait simple doit être horizontal ou vertical")

    @property
    def covered_cells(self) -> frozenset[Cell]:
        return frozenset(self.trace_cells)

    @property
    def length(self) -> int:
        return len(self.covered_cells)


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


_CARDINAL_NEIGHBOURS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def _ordered_neighbours(cell: Cell) -> tuple[tuple[int, int], ...]:
    """Favor a serpentine traversal while keeping every move cardinal."""

    if cell.row % 2:
        return ((-1, 0), (0, 1), (1, 0), (0, -1))
    return _CARDINAL_NEIGHBOURS


def _cell_components(cells: frozenset[Cell]) -> tuple[frozenset[Cell], ...]:
    remaining = set(cells)
    components: list[frozenset[Cell]] = []
    while remaining:
        start = min(remaining, key=lambda cell: (cell.row, cell.column))
        remaining.remove(start)
        queue = deque((start,))
        component = {start}
        while queue:
            current = queue.popleft()
            for dx, dy in _CARDINAL_NEIGHBOURS:
                neighbour = Cell(current.column + dx, current.row + dy)
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    component.add(neighbour)
                    queue.append(neighbour)
        components.append(frozenset(component))
    return tuple(components)


def _coverage_trace(component: frozenset[Cell]) -> tuple[Cell, ...]:
    """Return one continuous walk that covers a connected cell component.

    An ordered depth-first tree makes solid areas naturally snake from row to
    row. Branches are retraced only when necessary, and the deepest branch is
    kept for last so the mouse never needs to teleport back to the start.
    """

    start = min(component, key=lambda cell: (cell.row, cell.column))
    parents: dict[Cell, Cell | None] = {start: None}
    depths = {start: 0}
    children: dict[Cell, list[Cell]] = {start: []}
    discovery_order = [start]
    stack: list[tuple[Cell, int]] = [(start, 0)]
    neighbour_cache: dict[Cell, tuple[Cell, ...]] = {}

    while stack:
        current, offset = stack[-1]
        neighbours = neighbour_cache.setdefault(
            current,
            tuple(
                Cell(current.column + dx, current.row + dy)
                for dx, dy in _ordered_neighbours(current)
                if Cell(current.column + dx, current.row + dy) in component
            ),
        )
        if offset >= len(neighbours):
            stack.pop()
            continue
        stack[-1] = (current, offset + 1)
        neighbour = neighbours[offset]
        if neighbour in parents:
            continue
        parents[neighbour] = current
        depths[neighbour] = depths[current] + 1
        children.setdefault(current, []).append(neighbour)
        children[neighbour] = []
        discovery_order.append(neighbour)
        stack.append((neighbour, 0))

    terminal = max(
        discovery_order,
        key=lambda cell: (depths[cell], cell.row, cell.column),
    )
    terminal_child: dict[Cell, Cell] = {}
    cursor = terminal
    while parents[cursor] is not None:
        parent = parents[cursor]
        assert parent is not None
        terminal_child[parent] = cursor
        cursor = parent

    trace = [start]
    events: list[tuple[str, Cell]] = [("visit", start)]
    while events:
        event, cell = events.pop()
        if event == "move":
            trace.append(cell)
            continue
        final_child = terminal_child.get(cell)
        ordered_events: list[tuple[str, Cell]] = []
        for child in children[cell]:
            if child == final_child:
                continue
            ordered_events.extend(
                (("move", child), ("visit", child), ("move", cell))
            )
        if final_child is not None:
            ordered_events.extend((("move", final_child), ("visit", final_child)))
        events.extend(reversed(ordered_events))

    if frozenset(trace) != component:
        raise RuntimeError("Le parcours continu ne couvre pas toute la zone")
    return tuple(trace)


def build_continuous_strokes(
    cells: tuple[Cell, ...],
    palette_index: int,
) -> tuple[Stroke, ...]:
    """Cover each connected component with one continuous mouse gesture."""

    unique_cells = frozenset(cells)
    return tuple(
        Stroke.from_trace(palette_index, _coverage_trace(component))
        for component in _cell_components(unique_cells)
    )


def build_painting_plan(
    indices: np.ndarray, *, point_by_point: bool = False
) -> PaintingPlan:
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
            rows, columns = np.nonzero(mask)
            cells = tuple(
                Cell(int(column), int(row))
                for row, column in zip(rows, columns, strict=True)
            )
            strokes = (
                tuple(Stroke(palette_index, cell, cell) for cell in cells)
                if point_by_point
                else build_continuous_strokes(cells, palette_index)
            )
            shade_passes.append(ShadePass(palette_index, strokes))
        family_passes.append(FamilyPass(group, tuple(shade_passes)))
    return PaintingPlan(width, height, tuple(family_passes))
