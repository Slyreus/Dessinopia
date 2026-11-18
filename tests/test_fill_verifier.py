from dataclasses import replace
from unittest import TestCase

import numpy as np

from dessinopia.core.fill_planner import FillOperation, build_hybrid_fill_plan
from dessinopia.core.models import REFERENCE_PROFILE_2560, ScreenRect
from dessinopia.core.palette import PALETTE_BY_INDEX
from dessinopia.core.path_planner import Cell
from dessinopia.vision.fill_verifier import FillCanvasVerifier, FillCheckState


class FillVerifierTests(TestCase):
    def test_guard_already_painted_before_fill(self) -> None:
        indices = np.full((7, 7), -1, dtype=np.int16)
        indices[1:6, 1:6] = 2
        plan = build_hybrid_fill_plan(indices, enable_fill=True)
        operation = next(op for op in plan.operations if isinstance(op, FillOperation))
        profile = replace(
            REFERENCE_PROFILE_2560,
            reference_width=70,
            reference_height=70,
            canvas=ScreenRect(0, 0, 70, 70),
        )
        frame = np.full((70, 70, 3), (249, 245, 238), dtype=np.uint8)

        def paint(cells: tuple[Cell, ...]) -> None:
            for cell in cells:
                x, y = cell.column * 10, cell.row * 10
                frame[y : y + 10, x : x + 10] = PALETTE_BY_INDEX[2].rgb

        existing_guard, other_guard = operation.guard_cells[:2]
        paint((*operation.required_fence, existing_guard))
        verifier = FillCanvasVerifier()
        before = verifier.check_before(frame, profile, 7, 7, operation)
        self.assertEqual(before.state, FillCheckState.FALLBACK)
        self.assertEqual(before.missing_cells, operation.expected_cells)

        paint(operation.expected_cells)
        after = verifier.check_after(frame, profile, 7, 7, operation)
        self.assertEqual(after.state, FillCheckState.SUCCESS)
        self.assertEqual(after.guard_matches, 0)

        paint((other_guard,))
        overflow = verifier.check_after(frame, profile, 7, 7, operation)
        self.assertEqual(overflow.state, FillCheckState.OVERFLOW)
        self.assertEqual(overflow.guard_matches, 1)
