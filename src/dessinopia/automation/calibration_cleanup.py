from __future__ import annotations

import threading
import time
from collections.abc import Callable
from random import Random

from dessinopia.automation.calibration_test import CalibrationTestPattern
from dessinopia.automation.motion import randomized_click_point, ui_click_radius
from dessinopia.automation.runner import (
    AutomationCallbacks,
    AutomationState,
    AutomationStopped,
    MouseController,
)
from dessinopia.automation.win32_input import Win32Mouse
from dessinopia.core.models import CalibrationProfile, Point, ScreenRect
from dessinopia.core.timing import TimingProfile
from dessinopia.vision.shade_observer import ShadeSelectionObservation
from dessinopia.vision.tool_observer import (
    DrawingTool,
    ToolObservation,
    tool_marker_geometry_error,
)


class CalibrationCleanupRunner:
    """Efface les sondes validées avec la transparence de Heartopia."""

    def __init__(
        self,
        pattern: CalibrationTestPattern,
        calibration: CalibrationProfile,
        timing: TimingProfile,
        callbacks: AutomationCallbacks | None = None,
        verify_tool: Callable[[DrawingTool], bool | ToolObservation] | None = None,
        verify_shade: Callable[[int, int], bool | ShadeSelectionObservation] | None = None,
        mouse_factory: Callable[[Callable[[], None]], MouseController] | None = None,
        seed: int | None = None,
    ) -> None:
        self.pattern = pattern
        self.calibration = calibration
        self.timing = timing
        self.callbacks = callbacks or AutomationCallbacks()
        self.verify_tool = verify_tool
        self.verify_shade = verify_shade
        self.mouse_factory = mouse_factory or (lambda checkpoint: Win32Mouse(checkpoint=checkpoint))
        self.random = Random(seed)
        self._stop = threading.Event()
        self._run_gate = threading.Event()
        self._run_gate.set()
        self._state_lock = threading.RLock()
        self._state = AutomationState.IDLE
        self._mouse: MouseController | None = None

    @property
    def state(self) -> AutomationState:
        with self._state_lock:
            return self._state

    def pause(self) -> None:
        with self._state_lock:
            if self._state is AutomationState.RUNNING:
                self._run_gate.clear()
                self._set_state(AutomationState.PAUSED)

    def resume(self) -> None:
        with self._state_lock:
            if self._state is AutomationState.PAUSED:
                self._run_gate.set()
                self._set_state(AutomationState.RUNNING)

    def stop(self) -> None:
        with self._state_lock:
            if self._state in (AutomationState.RUNNING, AutomationState.PAUSED):
                self._set_state(AutomationState.STOPPING)
            self._stop.set()
            self._run_gate.set()

    def run(self) -> None:
        with self._state_lock:
            if self._state not in (AutomationState.IDLE, AutomationState.STOPPED):
                raise RuntimeError("Ce nettoyage est déjà actif")
            stopped_before_start = self._state is AutomationState.IDLE and self._stop.is_set()
            if not stopped_before_start:
                self._stop.clear()
                self._run_gate.set()
        if stopped_before_start:
            self.callbacks.status("Nettoyage interrompu avant son lancement")
            self._set_state(AutomationState.STOPPED)
            return

        self._mouse = None
        total = len(self.pattern.cleanup.cells)
        try:
            geometry_error = tool_marker_geometry_error(self.calibration)
            if geometry_error is not None:
                raise RuntimeError(f"Calibration des outils refusée : {geometry_error}")
            self._mouse = self.mouse_factory(self._motion_checkpoint)
            self._set_state(AutomationState.RUNNING)
            self.callbacks.progress(0, total)
            self.callbacks.status("Sélection du crayon et de la transparence")

            # Le nettoyage peut suivre une automatisation interrompue pendant
            # laquelle le pot était actif. Toujours repasser au crayon avant de
            # choisir la transparence et, surtout, avant de toucher la toile.
            self._select_and_verify_pencil()

            selection = self.pattern.cleanup.selection
            self._click_at(
                self.calibration.main_color_point(
                    selection.main_column,
                    selection.main_row,
                )
            )
            self._click_at(self.calibration.palette_icon, palette_delay=True)
            self._select_and_verify_transparency(
                selection.shade_column,
                selection.shade_row,
            )

            self.callbacks.status("Effacement des cinq points de test")
            for done, cell in enumerate(self.pattern.cleanup.cells, start=1):
                self._checkpoint()
                point = self.calibration.canvas.cell_center(
                    cell.column,
                    cell.row,
                    self.pattern.width,
                    self.pattern.height,
                )
                self._click_at(point, jitter=False)
                self.callbacks.progress(done, total)

            self._click_at(self.calibration.back_button, palette_delay=True)
            self.callbacks.status("Points de test effacés")
            self._set_state(AutomationState.COMPLETED)
        except AutomationStopped:
            self.callbacks.status("Nettoyage interrompu en toute sécurité")
            self._set_state(AutomationState.STOPPED)
        except Exception as exc:
            self.callbacks.status(str(exc))
            self._set_state(AutomationState.ERROR)
            raise
        finally:
            if self._mouse is not None:
                self._mouse.release_all()

    def _click_at(
        self,
        point: Point,
        palette_delay: bool = False,
        *,
        jitter: bool = True,
        safe_region: ScreenRect | None = None,
    ) -> None:
        assert self._mouse is not None
        self._checkpoint()
        target = (
            randomized_click_point(
                point,
                self.random,
                ui_click_radius(self.calibration),
                safe_region,
            )
            if jitter
            else point
        )
        self._mouse.move_to(target, self._varied(self.timing.cursor_move_seconds))
        self._mouse.click()
        delay = (
            self.timing.palette_settle_seconds
            if palette_delay
            else self.timing.click_settle_seconds
        )
        self._sleep(delay)

    def _select_and_verify_pencil(self) -> None:
        for attempt in range(1, 4):
            self._click_at(
                self.calibration.pencil_tool,
                safe_region=self.calibration.pencil_region,
            )
            if self.verify_tool is None:
                return
            result = self.verify_tool(DrawingTool.PENCIL)
            if isinstance(result, ToolObservation):
                self.callbacks.status(f"Outil observé — {result.summary}")
                accepted = result.matches(DrawingTool.PENCIL)
            elif isinstance(result, bool):
                accepted = result
            else:
                raise TypeError("La vérification d'outil a renvoyé un résultat invalide")
            if accepted:
                return
            self.callbacks.status(
                f"Crayon non confirmé — nouvelle sélection ({attempt}/3)"
            )
        raise RuntimeError(
            "Le crayon n'a pas pu être confirmé après trois sélections. "
            "Aucun clic n'a été envoyé sur la toile."
        )

    def _select_and_verify_transparency(self, column: int, row: int) -> None:
        point = self.calibration.shade_point(column, row)
        last_summary = "aucune observation disponible"
        for attempt in range(1, 4):
            self._click_at(point)
            if self.verify_shade is None:
                return
            assert self._mouse is not None
            self._mouse.move_to(
                self.calibration.pencil_tool,
                self._varied(self.timing.cursor_move_seconds),
            )
            result = self.verify_shade(column, row)
            if isinstance(result, ShadeSelectionObservation):
                last_summary = result.summary
                self.callbacks.status(f"Transparence observée — {result.summary}")
                accepted = result.matches_expected
            elif isinstance(result, bool):
                last_summary = "vérification booléenne refusée"
                accepted = result
            else:
                raise TypeError("La vérification de transparence est invalide")
            if accepted:
                return
            self.callbacks.status(
                f"Transparence non confirmée — nouvelle sélection ({attempt}/3)"
            )
        raise RuntimeError(
            "Le clic sur la transparence N6 a bien été envoyé trois fois, mais son "
            "contour de sélection n'a pas pu être confirmé. Les cinq points de test "
            "n'ont pas été touchés. Vérifiez visuellement N6 puis relancez le nettoyage. "
            f"Dernière lecture : {last_summary}."
        )

    def _sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + self._varied(seconds)
        while time.monotonic() < deadline:
            self._checkpoint()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.02, remaining))

    def _checkpoint(self) -> None:
        self._motion_checkpoint()
        while not self._run_gate.wait(timeout=0.05):
            self._motion_checkpoint()
        self._motion_checkpoint()

    def _motion_checkpoint(self) -> None:
        if self._stop.is_set():
            raise AutomationStopped
        if self._mouse is not None and self._mouse.emergency_stop_requested():
            self._stop.set()
            raise AutomationStopped

    def _varied(self, value: float) -> float:
        return self.timing.varied(value, self.random)

    def _set_state(self, state: AutomationState) -> None:
        with self._state_lock:
            self._state = state
        self.callbacks.state(state)
