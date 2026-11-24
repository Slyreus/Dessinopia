from dataclasses import replace
from unittest import TestCase
from unittest.mock import Mock, call

from dessinopia.automation.runner import AutomationRunner
from dessinopia.core.models import REFERENCE_PROFILE_2560
from dessinopia.core.path_planner import Cell, PaintingPlan, Stroke
from dessinopia.core.timing import TIMING_PRESETS


class PencilMotionTests(TestCase):
    def test_holds_pencil_at_turn_and_end(self) -> None:
        timing = replace(TIMING_PRESETS["rapide"], jitter_ratio=0)
        profile = REFERENCE_PROFILE_2560
        plan = PaintingPlan(3, 2, ())
        trace = (Cell(0, 0), Cell(1, 0), Cell(2, 0), Cell(2, 1))
        stroke = Stroke.from_trace(0, trace)
        runner = AutomationRunner(plan, profile, timing)
        mouse = Mock()
        mouse.emergency_stop_requested.return_value = False
        mouse.pause_toggle_requested.return_value = False
        runner._mouse = mouse
        runner._motion_sleep = mouse.hold

        runner._drag_stroke(stroke)

        points = [
            profile.canvas.cell_center(cell.column, cell.row, plan.width, plan.height)
            for cell in trace
        ]
        gestures = [
            event for event in mouse.mock_calls
            if event[0] in {"move_to", "left_down", "hold", "left_up"}
        ]
        self.assertEqual(gestures, [
            call.move_to(points[0], timing.cursor_move_seconds),
            call.left_down(),
            call.hold(0.060),
            call.move_to(points[1], timing.stroke_cell_seconds),
            call.move_to(points[2], timing.stroke_cell_seconds),
            call.hold(0.060),
            call.move_to(points[3], timing.stroke_cell_seconds),
            call.hold(0.060),
            call.left_up(),
        ])
        self.assertAlmostEqual(
            timing.estimate_seconds(4, 1, 0, 0, stroke_path=(stroke,)),
            timing.cursor_move_seconds + timing.between_strokes_seconds
            + 3 * (timing.stroke_cell_seconds + 0.060),
        )
