from dataclasses import replace
from unittest import TestCase
from unittest.mock import Mock, call

from dessinopia.automation.runner import AutomationRunner, AutomationState
from dessinopia.core.models import REFERENCE_PROFILE_2560
from dessinopia.core.palette import PALETTE_BY_INDEX
from dessinopia.core.path_planner import Cell, PaintingPlan, Stroke
from dessinopia.core.timing import TIMING_PRESETS
from dessinopia.vision.detector import PaletteState
from dessinopia.vision.palette_observer import (
    PaletteDiagnostics,
    PaletteObservation,
    PaletteRecoveryAction,
)
from dessinopia.vision.tool_observer import DrawingTool


class PencilMotionTests(TestCase):
    def test_changes_tempo_after_the_current_stroke(self) -> None:
        timing = replace(TIMING_PRESETS["lent"], jitter_ratio=0)
        faster = replace(TIMING_PRESETS["rapide"], jitter_ratio=0)
        runner = AutomationRunner(PaintingPlan(2, 1, ()), REFERENCE_PROFILE_2560, timing)
        stroke = Stroke.from_trace(0, (Cell(0, 0), Cell(1, 0)))
        mouse = Mock()
        mouse.emergency_stop_requested.return_value = False
        mouse.pause_toggle_requested.return_value = False
        runner._mouse = mouse
        runner._motion_sleep = mouse.hold
        runner._verify_resume_paint = Mock()

        def press_f8() -> None:
            mouse.pause_toggle_requested.return_value = True

        def on_state(state: AutomationState) -> None:
            if state is AutomationState.PAUSING:
                mouse.left_up.assert_not_called()
                self.assertFalse(runner.set_paused_timing(faster))
                mouse.pause_toggle_requested.return_value = False
            elif state is AutomationState.PAUSED:
                mouse.left_up.assert_called_once()
                self.assertTrue(runner.set_paused_timing(faster))
                runner.resume()

        runner.callbacks.state = on_state
        runner._set_state(AutomationState.RUNNING)
        mouse.left_down.side_effect = press_f8
        runner._drag_stroke(stroke)

        self.assertIs(runner.state, AutomationState.PAUSING)
        self.assertEqual(mouse.move_to.call_args.args[1], timing.stroke_cell_seconds)
        mouse.left_down.side_effect = None
        runner._drag_stroke(stroke)
        self.assertIs(runner.state, AutomationState.RUNNING)
        self.assertEqual(mouse.move_to.call_args.args[1], faster.stroke_cell_seconds)

    def test_reopens_palette_and_blocks_unconfirmed_shade_on_resume(self) -> None:
        color = PALETTE_BY_INDEX[0]
        main_palette = PaletteObservation(
            state=PaletteState.MAIN,
            confidence=1.0,
            group=None,
            group_confidence=0.0,
            expected_group=color.group,
            recovery=PaletteRecoveryAction.REOPEN_SHADES,
            diagnostics=PaletteDiagnostics(0.0, 1.0, 0.0, None, None, 0, 0),
        )
        for matches in (True, False):
            with self.subTest(matches=matches):
                observe_group = Mock(side_effect=(main_palette, True, True))
                verify_tool = Mock(return_value=True)
                verify_shade = Mock(return_value=matches)
                runner = AutomationRunner(
                    PaintingPlan(2, 1, ()),
                    REFERENCE_PROFILE_2560,
                    TIMING_PRESETS["prudent"],
                    verify_group=observe_group,
                    verify_tool=verify_tool,
                    verify_shade=verify_shade,
                    force_verification_failures=True,
                )
                mouse = Mock()
                mouse.emergency_stop_requested.return_value = False
                mouse.pause_toggle_requested.return_value = False
                runner._mouse = mouse
                runner._motion_sleep = Mock()
                runner._sleep = Mock()
                runner._select_family = Mock()
                runner._set_state(AutomationState.RUNNING)
                runner.pause()
                runner.resume()
                stroke = Stroke.from_trace(color.index, (Cell(0, 0), Cell(1, 0)))

                if matches:
                    runner._drag_stroke(stroke)
                    mouse.left_down.assert_called_once()
                else:
                    with self.assertRaisesRegex(RuntimeError, "n'a pas pu être confirmée"):
                        runner._drag_stroke(stroke)
                    mouse.left_down.assert_not_called()
                runner._select_family.assert_called_once_with(color.main_column, color.main_row)
                verify_tool.assert_called_once_with(DrawingTool.PENCIL)
                verify_shade.assert_called_with(color.index)

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
