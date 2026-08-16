from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from dessinopia.core.palette import PALETTE_BY_INDEX
from dessinopia.core.path_planner import (
    Cell,
    FamilyPass,
    PaintingPlan,
    ShadePass,
    Stroke,
    build_continuous_strokes,
    build_painting_plan,
)

_EMPTY = -2
_NEIGHBOURS: dict[int, tuple[tuple[int, int], ...]] = {
    4: ((-1, 0), (1, 0), (0, -1), (0, 1)),
    8: (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
    ),
}


class FillDecision(StrEnum):
    """Reason why one colour component is filled or drawn directly."""

    FILL = "fill"
    DISABLED = "disabled"
    NO_INTERIOR = "no-interior"
    BELOW_THRESHOLD = "below-threshold"
    UNSAFE_CONNECTIVITY = "unsafe-connectivity"


@dataclass(frozen=True, slots=True)
class PencilOperation:
    """One continuous pencil gesture associated with its source component."""

    component_id: int
    stroke: Stroke

    @property
    def palette_index(self) -> int:
        return self.stroke.palette_index

    @property
    def pixel_count(self) -> int:
        return self.stroke.length


@dataclass(frozen=True, slots=True)
class FillOperation:
    """A flood-fill click whose exact result was proved for both connectivities."""

    component_id: int
    palette_index: int
    seed: Cell
    expected_cells: tuple[Cell, ...]
    required_fence: tuple[Cell, ...]
    guard_cells: tuple[Cell, ...]
    fence_strokes: tuple[Stroke, ...]
    fallback_strokes: tuple[Stroke, ...]
    repair_strokes: tuple[Stroke, ...]
    verified_connectivity_4: bool = True
    verified_connectivity_8: bool = True

    @property
    def filled_pixel_count(self) -> int:
        return len(self.expected_cells)

    @property
    def dual_connectivity_verified(self) -> bool:
        return self.verified_connectivity_4 and self.verified_connectivity_8


HybridOperation = PencilOperation | FillOperation


@dataclass(frozen=True, slots=True)
class ComponentFillPlan:
    """Geometry and decision for one 4-connected, single-colour component."""

    component_id: int
    palette_index: int
    cells: tuple[Cell, ...]
    boundary: tuple[Cell, ...]
    interior_regions: tuple[tuple[Cell, ...], ...]
    decision: FillDecision
    pencil_operations: tuple[PencilOperation, ...]
    fill_operations: tuple[FillOperation, ...]
    safe_connectivity_4: bool | None
    safe_connectivity_8: bool | None

    @property
    def uses_fill(self) -> bool:
        return self.decision is FillDecision.FILL

    @property
    def interior(self) -> tuple[Cell, ...]:
        return tuple(cell for region in self.interior_regions for cell in region)

    @property
    def direct_pixel_count(self) -> int:
        return sum(operation.pixel_count for operation in self.pencil_operations)

    @property
    def potential_savings(self) -> int:
        return len(self.cells) - len(self.boundary) - len(self.interior_regions)


@dataclass(frozen=True, slots=True)
class HybridShadePass:
    palette_index: int
    pencil_operations: tuple[PencilOperation, ...]
    fill_operations: tuple[FillOperation, ...]

    @property
    def operations(self) -> tuple[HybridOperation, ...]:
        # All barriers are completed before the bucket is enabled. This ordering
        # is part of the safety proof performed by ``build_hybrid_fill_plan``.
        return (*self.pencil_operations, *self.fill_operations)

    @property
    def pixel_count(self) -> int:
        return sum(item.pixel_count for item in self.pencil_operations) + sum(
            item.filled_pixel_count for item in self.fill_operations
        )


@dataclass(frozen=True, slots=True)
class HybridFamilyPass:
    group: int
    shades: tuple[HybridShadePass, ...]


@dataclass(frozen=True, slots=True)
class FillMetrics:
    target_pixels: int
    direct_pixels: int
    pencil_strokes: int
    fill_clicks: int
    saved_pixel_actions: int
    component_count: int
    filled_components: int
    direct_components: int
    candidate_components: int
    unsafe_connectivity_components: int
    verified_connectivity_4_fills: int
    verified_connectivity_8_fills: int
    dual_connectivity_verified_fills: int

    @property
    def estimated_pixel_actions(self) -> int:
        return self.direct_pixels + self.fill_clicks

    @property
    def savings_ratio(self) -> float:
        if not self.target_pixels:
            return 0.0
        return self.saved_pixel_actions / self.target_pixels


@dataclass(frozen=True, slots=True)
class HybridPaintingPlan:
    width: int
    height: int
    families: tuple[HybridFamilyPass, ...]
    components: tuple[ComponentFillPlan, ...]
    metrics: FillMetrics
    fill_enabled: bool
    minimum_savings: int

    @property
    def operations(self) -> tuple[HybridOperation, ...]:
        return tuple(
            operation
            for family in self.families
            for shade in family.shades
            for operation in shade.operations
        )

    @property
    def has_fills(self) -> bool:
        return self.metrics.fill_clicks > 0

    @property
    def pixel_count(self) -> int:
        """Logical image pixels committed by the runner.

        A successful bucket click commits every cell of its proved region at
        once, so progress and checkpoints keep the same pixel-based semantics
        as the pencil-only runner.
        """

        return self.metrics.target_pixels

    @property
    def stroke_count(self) -> int:
        return self.metrics.pencil_strokes

    @property
    def shade_count(self) -> int:
        return sum(len(family.shades) for family in self.families)

    @property
    def action_count(self) -> int:
        return self.metrics.pencil_strokes + self.metrics.fill_clicks

    def to_painting_plan(self) -> PaintingPlan:
        """Return the legacy plan when every pixel is assigned to the pencil.

        The old runner cannot express a bucket click. Refusing conversion when
        fills exist avoids silently dropping painted interiors.
        """

        if self.has_fills:
            raise ValueError("Un plan contenant des remplissages n'est pas un plan crayon")
        families = tuple(
            FamilyPass(
                family.group,
                tuple(
                    ShadePass(
                        shade.palette_index,
                        tuple(operation.stroke for operation in shade.pencil_operations),
                    )
                    for shade in family.shades
                ),
            )
            for family in self.families
        )
        return PaintingPlan(self.width, self.height, families)


@dataclass(slots=True)
class _ComponentWork:
    component_id: int
    palette_index: int
    cells: tuple[Cell, ...]
    boundary: tuple[Cell, ...]
    interior_regions: tuple[tuple[Cell, ...], ...]
    decision: FillDecision
    safe_connectivity_4: bool | None = None
    safe_connectivity_8: bool | None = None
    fill_operations: tuple[FillOperation, ...] = ()

    @property
    def is_candidate(self) -> bool:
        return self.decision is FillDecision.FILL

    @property
    def pencil_cells(self) -> tuple[Cell, ...]:
        return self.boundary if self.decision is FillDecision.FILL else self.cells


def build_hybrid_fill_plan(
    indices: np.ndarray,
    *,
    enable_fill: bool = False,
    minimum_savings: int = 4,
) -> HybridPaintingPlan:
    """Build a lossless pencil/bucket plan from palette indices.

    A bucket candidate is kept only when its simulated flood region equals the
    intended interior under *both* 4- and 8-neighbour connectivity. This models
    the two plausible Heartopia behaviours conservatively. ``minimum_savings``
    is the minimum number of pixel actions saved by a component; a candidate
    with no strictly positive saving is never accepted.
    """

    _validate_indices(indices)
    if isinstance(minimum_savings, bool) or not isinstance(minimum_savings, int):
        raise ValueError("Le seuil d'economie doit etre un entier")
    if minimum_savings < 0:
        raise ValueError("Le seuil d'economie ne peut pas etre negatif")

    height, width = indices.shape
    legacy_plan = build_painting_plan(indices)
    working = np.full((height, width), _EMPTY, dtype=np.int16)
    component_sequence = 0
    component_results: list[ComponentFillPlan] = []
    hybrid_families: list[HybridFamilyPass] = []
    candidate_count = 0

    for legacy_family in legacy_plan.families:
        hybrid_shades: list[HybridShadePass] = []
        for legacy_shade in legacy_family.shades:
            palette_index = legacy_shade.palette_index
            component_cells = _connected_components(indices == palette_index, 4)
            work_items: list[_ComponentWork] = []

            for cells in component_cells:
                boundary = _component_boundary(cells, width, height)
                boundary_set = {(cell.column, cell.row) for cell in boundary}
                interior_cells = tuple(
                    cell
                    for cell in cells
                    if (cell.column, cell.row) not in boundary_set
                )
                interior_regions = _regions_from_cells(interior_cells, 4)
                potential_savings = len(cells) - len(boundary) - len(interior_regions)
                if not enable_fill:
                    decision = FillDecision.DISABLED
                elif not interior_regions:
                    decision = FillDecision.NO_INTERIOR
                elif potential_savings <= 0 or potential_savings < minimum_savings:
                    decision = FillDecision.BELOW_THRESHOLD
                else:
                    decision = FillDecision.FILL
                    candidate_count += 1
                work_items.append(
                    _ComponentWork(
                        component_id=component_sequence,
                        palette_index=palette_index,
                        cells=cells,
                        boundary=boundary,
                        interior_regions=interior_regions,
                        decision=decision,
                    )
                )
                component_sequence += 1

            # Pencil every direct component and every prospective barrier before
            # evaluating a bucket. The generated pass executes in this same order.
            for item in work_items:
                for cell in item.pencil_cells:
                    working[cell.row, cell.column] = palette_index

            for item in work_items:
                if not item.is_candidate:
                    continue
                fill_operations: list[FillOperation] = []
                safe_four = True
                safe_eight = True
                guard_cells = _component_guards(
                    item.cells,
                    indices,
                    palette_index,
                )
                for region in item.interior_regions:
                    seed = _safest_seed(region, item.boundary)
                    expected = {(cell.column, cell.row) for cell in region}
                    actual_four = _flood_region(working, seed, 4)
                    actual_eight = _flood_region(working, seed, 8)
                    region_safe_four = actual_four == expected
                    region_safe_eight = actual_eight == expected
                    safe_four = safe_four and region_safe_four
                    safe_eight = safe_eight and region_safe_eight
                    if not (region_safe_four and region_safe_eight):
                        break
                    fill_operations.append(
                        FillOperation(
                            component_id=item.component_id,
                            palette_index=palette_index,
                            seed=seed,
                            expected_cells=region,
                            required_fence=item.boundary,
                            guard_cells=guard_cells,
                            fence_strokes=_fallback_strokes(
                                item.boundary,
                                palette_index,
                                width,
                                height,
                                item.component_id,
                            ),
                            fallback_strokes=_fallback_strokes(
                                region,
                                palette_index,
                                width,
                                height,
                                item.component_id,
                            ),
                            repair_strokes=_fallback_strokes(
                                tuple(
                                    sorted(
                                        {*item.boundary, *region},
                                        key=_cell_key,
                                    )
                                ),
                                palette_index,
                                width,
                                height,
                                item.component_id,
                            ),
                        )
                    )
                    for cell in region:
                        working[cell.row, cell.column] = palette_index

                item.safe_connectivity_4 = safe_four
                item.safe_connectivity_8 = safe_eight
                if safe_four and safe_eight:
                    item.fill_operations = tuple(fill_operations)
                else:
                    # A partially simulated component is deliberately collapsed
                    # back to direct drawing. Painting every target cell also leaves
                    # the proof canvas in exactly the state used by later checks.
                    item.decision = FillDecision.UNSAFE_CONNECTIVITY
                    item.fill_operations = ()
                    for cell in item.cells:
                        working[cell.row, cell.column] = palette_index

            pencil_cells = tuple(cell for item in work_items for cell in item.pencil_cells)
            component_by_coordinate = {
                (cell.column, cell.row): item.component_id
                for item in work_items
                for cell in item.cells
            }
            pencil_operations = _pencil_operations(
                pencil_cells,
                palette_index,
                width,
                height,
                component_by_coordinate,
            )
            fills = tuple(operation for item in work_items for operation in item.fill_operations)
            hybrid_shades.append(HybridShadePass(palette_index, pencil_operations, fills))

            operations_by_component: dict[int, list[PencilOperation]] = {}
            for operation in pencil_operations:
                operations_by_component.setdefault(operation.component_id, []).append(operation)
            for item in work_items:
                component_results.append(
                    ComponentFillPlan(
                        component_id=item.component_id,
                        palette_index=item.palette_index,
                        cells=item.cells,
                        boundary=item.boundary,
                        interior_regions=item.interior_regions,
                        decision=item.decision,
                        pencil_operations=tuple(
                            operations_by_component.get(item.component_id, ())
                        ),
                        fill_operations=item.fill_operations,
                        safe_connectivity_4=item.safe_connectivity_4,
                        safe_connectivity_8=item.safe_connectivity_8,
                    )
                )

        hybrid_families.append(HybridFamilyPass(legacy_family.group, tuple(hybrid_shades)))

    normalized = working.copy()
    normalized[normalized == _EMPTY] = -1
    if not np.array_equal(normalized, indices):
        raise RuntimeError("Le plan hybride interne ne reproduit pas l'image cible")

    direct_pixels = sum(
        operation.pixel_count
        for family in hybrid_families
        for shade in family.shades
        for operation in shade.pencil_operations
    )
    pencil_strokes = sum(
        len(shade.pencil_operations)
        for family in hybrid_families
        for shade in family.shades
    )
    fill_operations = tuple(
        operation
        for family in hybrid_families
        for shade in family.shades
        for operation in shade.fill_operations
    )
    target_pixels = int(np.count_nonzero(indices >= 0))
    filled_components = sum(component.uses_fill for component in component_results)
    metrics = FillMetrics(
        target_pixels=target_pixels,
        direct_pixels=direct_pixels,
        pencil_strokes=pencil_strokes,
        fill_clicks=len(fill_operations),
        saved_pixel_actions=target_pixels - direct_pixels - len(fill_operations),
        component_count=len(component_results),
        filled_components=filled_components,
        direct_components=len(component_results) - filled_components,
        candidate_components=candidate_count,
        unsafe_connectivity_components=sum(
            component.decision is FillDecision.UNSAFE_CONNECTIVITY
            for component in component_results
        ),
        verified_connectivity_4_fills=sum(
            operation.verified_connectivity_4 for operation in fill_operations
        ),
        verified_connectivity_8_fills=sum(
            operation.verified_connectivity_8 for operation in fill_operations
        ),
        dual_connectivity_verified_fills=sum(
            operation.dual_connectivity_verified for operation in fill_operations
        ),
    )
    return HybridPaintingPlan(
        width=width,
        height=height,
        families=tuple(hybrid_families),
        components=tuple(component_results),
        metrics=metrics,
        fill_enabled=enable_fill,
        minimum_savings=minimum_savings,
    )


def simulate_hybrid_plan(plan: HybridPaintingPlan, *, connectivity: int = 4) -> np.ndarray:
    """Execute a plan on an empty logical canvas for tests and diagnostics."""

    if connectivity not in _NEIGHBOURS:
        raise ValueError("La connexite doit valoir 4 ou 8")
    canvas = np.full((plan.height, plan.width), _EMPTY, dtype=np.int16)
    for family in plan.families:
        for shade in family.shades:
            for operation in shade.pencil_operations:
                stroke = operation.stroke
                for cell in stroke.covered_cells:
                    canvas[cell.row, cell.column] = stroke.palette_index
            for operation in shade.fill_operations:
                region = _flood_region(canvas, operation.seed, connectivity)
                for column, row in region:
                    canvas[row, column] = operation.palette_index
    canvas[canvas == _EMPTY] = -1
    return canvas


def _validate_indices(indices: np.ndarray) -> None:
    if not isinstance(indices, np.ndarray) or indices.ndim != 2:
        raise ValueError("La matrice de pixels doit avoir deux dimensions")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("La matrice de pixels doit contenir des indices entiers")
    if indices.size and (
        np.any(indices < -1) or np.any(indices >= len(PALETTE_BY_INDEX))
    ):
        raise ValueError("La matrice contient un indice de palette invalide")


def _connected_components(mask: np.ndarray, connectivity: int) -> tuple[tuple[Cell, ...], ...]:
    height, width = mask.shape
    unseen = mask.copy()
    results: list[tuple[Cell, ...]] = []
    for row in range(height):
        for column in range(width):
            if not unseen[row, column]:
                continue
            unseen[row, column] = False
            queue = deque(((column, row),))
            coordinates: list[Cell] = []
            while queue:
                current_column, current_row = queue.popleft()
                coordinates.append(Cell(current_column, current_row))
                for dx, dy in _NEIGHBOURS[connectivity]:
                    neighbour_column = current_column + dx
                    neighbour_row = current_row + dy
                    if (
                        0 <= neighbour_column < width
                        and 0 <= neighbour_row < height
                        and unseen[neighbour_row, neighbour_column]
                    ):
                        unseen[neighbour_row, neighbour_column] = False
                        queue.append((neighbour_column, neighbour_row))
            results.append(tuple(sorted(coordinates, key=_cell_key)))
    return tuple(results)


def _regions_from_cells(
    cells: tuple[Cell, ...], connectivity: int
) -> tuple[tuple[Cell, ...], ...]:
    if not cells:
        return ()
    remaining = {(cell.column, cell.row) for cell in cells}
    results: list[tuple[Cell, ...]] = []
    while remaining:
        start = min(remaining, key=lambda coordinate: (coordinate[1], coordinate[0]))
        remaining.remove(start)
        queue = deque((start,))
        region: list[Cell] = []
        while queue:
            column, row = queue.popleft()
            region.append(Cell(column, row))
            for dx, dy in _NEIGHBOURS[connectivity]:
                neighbour = (column + dx, row + dy)
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    queue.append(neighbour)
        results.append(tuple(sorted(region, key=_cell_key)))
    return tuple(results)


def _component_boundary(
    cells: tuple[Cell, ...], width: int, height: int
) -> tuple[Cell, ...]:
    """Return a fence thick enough to contain both plausible bucket behaviours.

    Heartopia's exact flood connectivity is not documented. A four-neighbour
    outline can leave diagonal passages around stair-stepped or curved shapes,
    which made most real image components fail the eight-neighbour safety proof.
    Keeping every cell exposed in any of the eight directions closes those
    passages while leaving the largest safe interior available to the bucket.
    """

    coordinates = {(cell.column, cell.row) for cell in cells}
    boundary = []
    for cell in cells:
        if any(
            not (0 <= cell.column + dx < width and 0 <= cell.row + dy < height)
            or (cell.column + dx, cell.row + dy) not in coordinates
            for dx, dy in _NEIGHBOURS[8]
        ):
            boundary.append(cell)
    return tuple(sorted(boundary, key=_cell_key))


def _component_guards(
    cells: tuple[Cell, ...],
    indices: np.ndarray,
    palette_index: int,
) -> tuple[Cell, ...]:
    """Return bounded external sentinels that cannot be valid target colour.

    A diagonally adjacent component of the same palette colour is deliberately
    excluded: observing that cell after a fill could not distinguish legitimate
    target paint from a leak.
    """

    height, width = indices.shape
    component = {(cell.column, cell.row) for cell in cells}
    guards: set[tuple[int, int]] = set()
    for cell in cells:
        for dx, dy in _NEIGHBOURS[8]:
            column = cell.column + dx
            row = cell.row + dy
            coordinate = (column, row)
            if (
                0 <= column < width
                and 0 <= row < height
                and coordinate not in component
                and int(indices[row, column]) != palette_index
            ):
                guards.add(coordinate)
    ordered = sorted(guards, key=lambda item: (item[1], item[0]))
    return tuple(Cell(column, row) for column, row in ordered)


def _safest_seed(region: tuple[Cell, ...], fence: tuple[Cell, ...]) -> Cell:
    """Choose the cell with maximum Euclidean clearance from the fence.

    Squared distances avoid floating-point ambiguity. Ties are resolved by the
    top-most, then left-most cell so identical inputs always yield identical plans.
    """

    if not region or not fence:
        raise ValueError("Une graine de remplissage exige une region et une frontiere")

    def ranking(cell: Cell) -> tuple[int, int, int]:
        clearance_squared = min(
            (cell.column - boundary.column) ** 2 + (cell.row - boundary.row) ** 2
            for boundary in fence
        )
        return -clearance_squared, cell.row, cell.column

    return min(region, key=ranking)


def _fallback_strokes(
    region: tuple[Cell, ...],
    palette_index: int,
    width: int,
    height: int,
    component_id: int,
) -> tuple[Stroke, ...]:
    component_by_coordinate = {
        (cell.column, cell.row): component_id for cell in region
    }
    operations = _pencil_operations(
        region,
        palette_index,
        width,
        height,
        component_by_coordinate,
    )
    strokes = tuple(operation.stroke for operation in operations)
    if _stroke_cells(strokes) != {(cell.column, cell.row) for cell in region}:
        raise RuntimeError("Les traits de secours ne couvrent pas exactement le remplissage")
    return strokes


def _flood_region(canvas: np.ndarray, seed: Cell, connectivity: int) -> set[tuple[int, int]]:
    height, width = canvas.shape
    if not (0 <= seed.column < width and 0 <= seed.row < height):
        return set()
    source_value = int(canvas[seed.row, seed.column])
    queue = deque(((seed.column, seed.row),))
    seen = {(seed.column, seed.row)}
    while queue:
        column, row = queue.popleft()
        for dx, dy in _NEIGHBOURS[connectivity]:
            neighbour_column = column + dx
            neighbour_row = row + dy
            coordinate = (neighbour_column, neighbour_row)
            if (
                0 <= neighbour_column < width
                and 0 <= neighbour_row < height
                and coordinate not in seen
                and int(canvas[neighbour_row, neighbour_column]) == source_value
            ):
                seen.add(coordinate)
                queue.append(coordinate)
    return seen


def _stroke_cells(strokes: tuple[Stroke, ...]) -> set[tuple[int, int]]:
    return {
        (cell.column, cell.row)
        for stroke in strokes
        for cell in stroke.covered_cells
    }


def _pencil_operations(
    cells: tuple[Cell, ...],
    palette_index: int,
    width: int,
    height: int,
    component_by_coordinate: dict[tuple[int, int], int],
) -> tuple[PencilOperation, ...]:
    del width, height
    cells_by_component: dict[int, list[Cell]] = {}
    for cell in cells:
        component_id = component_by_coordinate[(cell.column, cell.row)]
        cells_by_component.setdefault(component_id, []).append(cell)
    operations: list[PencilOperation] = []
    component_order = sorted(
        cells_by_component,
        key=lambda component_id: min(
            (_cell_key(cell) for cell in cells_by_component[component_id]),
        ),
    )
    for component_id in component_order:
        component_cells = tuple(cells_by_component[component_id])
        operations.extend(
            PencilOperation(component_id, stroke)
            for stroke in build_continuous_strokes(component_cells, palette_index)
        )
    return tuple(operations)


def _cell_key(cell: Cell) -> tuple[int, int]:
    return cell.row, cell.column
