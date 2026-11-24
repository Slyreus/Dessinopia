import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtWidgets import QApplication

from dessinopia.automation.runner import AutomationRunner, AutomationState
from dessinopia.core.models import REFERENCE_PROFILE_2560
from dessinopia.core.path_planner import Cell, build_painting_plan
from dessinopia.core.timing import TIMING_PRESETS
from dessinopia.settings import AppSettings
from dessinopia.ui.main_window import MainWindow
from dessinopia.vision.detector import CanvasAssessment, CanvasEvidence, PaletteState, StateDetection
from dessinopia.vision.hole_detector import HoleAudit


class DrawingCompletionTests(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with patch.dict(os.environ, {"QT_QPA_PLATFORM": "offscreen"}):
            cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        with patch("dessinopia.ui.main_window.SettingsStore") as store:
            store.return_value.load.return_value = AppSettings()
            self.window = MainWindow()
        self.addCleanup(self.window.close)
        self.window.settings.calibration = REFERENCE_PROFILE_2560
        self.window.settings.timing = TIMING_PRESETS["tres_lent"]
        indices = np.array([[0, 0]], dtype=np.int16)
        self.window.result = SimpleNamespace(indices=indices)
        self.window.plan = build_painting_plan(indices)
        self.journal = Mock()
        self.window._resume_journal = self.journal
        self.window._validate_canvas_before_start = Mock(return_value=(True, "", True))
        self.window._ensure_main_palette = Mock(return_value=True)
        self.window._confirm_hole_repair = Mock(
            side_effect=AssertionError("Confirmation demandée pendant les retouches")
        )
        self.window._launch_worker = Mock(side_effect=self.launch_worker)
        self.window._audit_current_canvas = Mock(
            side_effect=(
                HoleAudit((Cell(0, 0),), 2, (0.0,), (Cell(1, 0),)),
                HoleAudit((Cell(1, 0),), 2, (0.0,)),
                HoleAudit((), 2, ()),
            )
        )
        self.launch_worker(
            AutomationRunner(
                self.window.plan, REFERENCE_PROFILE_2560, self.window.settings.timing
            ),
            "draw",
        )
        self.window.runner._painted = 2

    def launch_worker(self, runner: AutomationRunner, mode: str) -> None:
        self.window.runner = runner
        self.window._run_mode = mode

    def finish_worker(self, state: AutomationState = AutomationState.COMPLETED) -> None:
        self.window.runner._state = state
        self.window._automation_state_changed(state.value)
        self.window._automation_thread_finished()

    def test_draw_automatically_repairs_and_checks_again(self) -> None:
        self.finish_worker()
        self.assertEqual(self.window.runner.plan.stroke_count, 2)
        self.assertIs(self.window.runner.timing, self.window.settings.timing)
        self.assertEqual(self.window._resume_pixels, 2)
        self.journal.clear.assert_not_called()

        self.finish_worker()
        self.assertEqual(self.window.runner.plan.stroke_count, 1)
        self.assertIs(self.window.runner.timing, self.window.settings.timing)
        self.finish_worker()

        self.assertEqual(self.window._launch_worker.call_count, 2)
        self.window._confirm_hole_repair.assert_not_called()
        self.journal.clear.assert_called_once()
        self.assertEqual(self.window.overlay_toolbar.state_label.text(), "Terminé")

    def test_interrupted_touchups_resume_without_replaying_drawing(self) -> None:
        self.finish_worker()
        self.finish_worker(AutomationState.STOPPED)
        self.journal.clear.assert_not_called()
        self.assertEqual(self.window.overlay_toolbar.start_button.text(), "Reprendre les retouches")

        self.window._conversion_is_current = Mock(return_value=True)
        self.window._start_diagnostic_session = Mock()
        self.window.start_automation()
        self.assertEqual(self.window._run_mode, "hole_repair")
        self.assertEqual(self.window.runner.plan.stroke_count, 1)
        self.finish_worker()
        self.journal.clear.assert_called_once()

    def test_automatic_touchups_require_a_visible_canvas(self) -> None:
        self.window._validate_canvas_before_start = (
            MainWindow._validate_canvas_before_start.__get__(self.window)
        )
        self.window._capture_for_detection = Mock(
            return_value=(object(), REFERENCE_PROFILE_2560)
        )
        self.window.detector.assess_canvas = Mock(
            return_value=CanvasAssessment(None, CanvasEvidence.NONE, 0.0)
        )
        self.window.detector.detect_palette_state = Mock(
            return_value=StateDetection(PaletteState.SHADES, 0.95)
        )

        self.finish_worker()

        self.window._launch_worker.assert_not_called()
        self.journal.clear.assert_not_called()
        self.assertEqual(self.window._resume_pixels, 2)
        self.assertEqual(self.window.overlay_toolbar.state_label.text(), "À vérifier")
