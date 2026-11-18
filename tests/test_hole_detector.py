from dataclasses import replace
from unittest import TestCase

import numpy as np

from dessinopia.core.models import REFERENCE_PROFILE_2560, ScreenRect
from dessinopia.core.palette import PALETTE_BY_INDEX
from dessinopia.core.path_planner import Cell, build_painting_plan
from dessinopia.vision.hole_detector import audit_canvas_holes, repair_indices


class HoleDetectorTests(TestCase):
    def test_plan_targets_missing_and_wrong_colors_with_single_clicks(self) -> None:
        indices = np.array([[0, 0, 4], [0, -1, -1]], dtype=np.int16)
        colors = np.array(
            [
                [(249, 245, 238), PALETTE_BY_INDEX[5].rgb, PALETTE_BY_INDEX[4].rgb],
                [(0, 0, 0), (249, 245, 238), PALETTE_BY_INDEX[5].rgb],
            ],
            dtype=np.uint8,
        )
        frame = colors.repeat(10, axis=0).repeat(10, axis=1)
        profile = replace(
            REFERENCE_PROFILE_2560,
            reference_width=30,
            reference_height=20,
            canvas=ScreenRect(0, 0, 30, 20),
        )

        audit = audit_canvas_holes(frame, profile, indices)
        self.assertEqual(audit.cells, (Cell(0, 0),))
        self.assertEqual(audit.wrong_color_cells, (Cell(1, 0),))
        self.assertEqual(audit.sampled_cells, 4)
        plan = build_painting_plan(repair_indices(indices, audit), point_by_point=True)
        self.assertEqual(plan.pixel_count, 2)
        self.assertEqual(plan.stroke_count, 2)
