from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from random import Random
from typing import Protocol

from dessinopia.automation.motion import randomized_click_point, ui_click_radius
from dessinopia.automation.win32_input import Win32Mouse
from dessinopia.core.models import CalibrationProfile, Point, ScreenRect
from dessinopia.core.palette import PALETTE_BY_INDEX
from dessinopia.core.path_planner import PaintingPlan, Stroke
from dessinopia.core.timing import TimingProfile
from dessinopia.vision.detector import PaletteState
from dessinopia.vision.palette_observer import (
    PaletteObservation,
    PaletteRecoveryAction,
)
from dessinopia.vision.shade_observer import ShadeSelectionObservation
from dessinopia.vision.tool_observer import (
    DrawingTool,
    ToolObservation,
    tool_marker_geometry_error,
)


class AutomationState(StrEnum):
    IDLE = "Prêt"
    RUNNING = "Dessin en cours"
    PAUSED = "En pause"
    STOPPING = "Arrêt en cours"
    STOPPED = "Arrêté"
    COMPLETED = "Terminé"
    ERROR = "Erreur"


class AutomationStopped(RuntimeError):
    pass


class MouseController(Protocol):
    def emergency_stop_requested(self) -> bool: ...

    def pause_toggle_requested(self) -> bool: ...

    def move_to(self, point: Point, duration: float = 0.0) -> None: ...

    def click(self) -> None: ...

    def left_down(self) -> None: ...

    def left_up(self) -> None: ...

    def release_all(self) -> None: ...


@dataclass(slots=True)
class AutomationCallbacks:
    state: Callable[[AutomationState], None] = lambda _value: None
    status: Callable[[str], None] = lambda _value: None
    progress: Callable[[int, int], None] = lambda _done, _total: None


class AutomationRunner:
    _palette_read_attempts = 5
    _palette_open_attempts = 4
    _tool_selection_attempts = 3

    def __init__(
        self,
        plan: PaintingPlan,
        calibration: CalibrationProfile,
        timing: TimingProfile,
        callbacks: AutomationCallbacks | None = None,
        verify_group: Callable[[int], bool | PaletteObservation] | None = None,
        verify_tool: Callable[[DrawingTool], bool | ToolObservation] | None = None,
        verify_shade: Callable[[int], bool | ShadeSelectionObservation] | None = None,
        mouse_factory: Callable[[Callable[[], None]], MouseController] | None = None,
        seed: int | None = None,
        resume_from_pixels: int = 0,
        force_verification_failures: bool = False,
    ) -> None:
        self.plan = plan
        self.calibration = calibration
        self.timing = timing
        self.callbacks = callbacks or AutomationCallbacks()
        self.verify_group = verify_group
        self.verify_tool = verify_tool
        self.verify_shade = verify_shade
        self.mouse_factory = mouse_factory or (lambda checkpoint: Win32Mouse(checkpoint=checkpoint))
        self.random = Random(seed)
        self.resume_from_pixels = resume_from_pixels
        self.force_verification_failures = force_verification_failures
        self._validate_resume_point()
        self._stop = threading.Event()
        self._run_gate = threading.Event()
        self._run_gate.set()
        self._state_lock = threading.RLock()
        self._state = AutomationState.IDLE
        self._mouse: MouseController | None = None
        self._painted = resume_from_pixels
        self._last_palette_summary = "aucun diagnostic disponible"
        self._last_palette_observation: PaletteObservation | None = None

    @property
    def state(self) -> AutomationState:
        with self._state_lock:
            return self._state

    @property
    def painted_pixels(self) -> int:
        """Number of pixels committed at a complete-stroke boundary.

        A stroke interrupted while the mouse button is down is deliberately not
        counted.  Resuming therefore redraws that whole stroke, which is idempotent
        in Heartopia and guarantees that no logical cell is skipped.
        """

        with self._state_lock:
            return self._painted

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
                raise RuntimeError("Cette exécution est déjà active")
            # Arrêter peut être demandé après QThread.start(), mais avant que ce
            # corps ne s'exécute. Cette demande précoce ne doit jamais être effacée.
            stopped_before_start = self._state is AutomationState.IDLE and self._stop.is_set()
            if not stopped_before_start:
                self._stop.clear()
                self._run_gate.set()
        if stopped_before_start:
            self.callbacks.status("Dessin interrompu avant son lancement")
            self._set_state(AutomationState.STOPPED)
            return
        self._painted = self.resume_from_pixels
        self._mouse = None
        try:
            self._validate_tool_markers()
            # Une pause prend effet entre deux traits. Un arrêt, lui, interrompt aussi
            # un trait en cours et le bouton de souris est toujours relâché en finally.
            self._mouse = self.mouse_factory(self._motion_checkpoint)
            self._set_state(AutomationState.RUNNING)
            self.callbacks.progress(self._painted, self.plan.pixel_count)
            pixels_to_skip = self.resume_from_pixels
            drawing_tool_selected = False
            if self.plan.pixel_count and pixels_to_skip == self.plan.pixel_count:
                self.callbacks.status(
                    "Checkpoint complet — remise de sécurité sur le crayon"
                )
                self._select_and_verify_pencil()
                drawing_tool_selected = True
            for family_number, family in enumerate(self.plan.families, start=1):
                self._checkpoint()
                family_pixels = sum(shade.pixel_count for shade in family.shades)
                if pixels_to_skip >= family_pixels:
                    pixels_to_skip -= family_pixels
                    continue
                exemplar = PALETTE_BY_INDEX[family.shades[0].palette_index]
                family_name = exemplar.name.rsplit(" ", 1)[0]
                self.callbacks.status(
                    f"Famille {family_number}/{len(self.plan.families)} — {family_name}"
                )
                self._open_verified_family(
                    exemplar.main_column,
                    exemplar.main_row,
                    family.group,
                    family_name,
                )
                if not drawing_tool_selected:
                    # Un lancement précédent peut avoir laissé le pot de peinture
                    # actif. Réarmer explicitement le crayon avant le premier
                    # contact avec la toile empêche alors tout remplissage fortuit.
                    self.callbacks.status("Sélection du crayon et du pinceau le plus fin")
                    self._select_and_verify_pencil()
                    self._click_at(self.calibration.small_brush)
                    drawing_tool_selected = True

                for shade_number, shade in enumerate(family.shades, start=1):
                    self._checkpoint()
                    if pixels_to_skip >= shade.pixel_count:
                        pixels_to_skip -= shade.pixel_count
                        continue
                    color = PALETTE_BY_INDEX[shade.palette_index]
                    self.callbacks.status(
                        f"{color.name} — nuance {shade_number}/{len(family.shades)}"
                    )
                    self._select_and_verify_shade(shade.palette_index)
                    for stroke in shade.strokes:
                        if pixels_to_skip >= stroke.length:
                            pixels_to_skip -= stroke.length
                            continue
                        if pixels_to_skip:
                            raise RuntimeError(
                                "Le point de reprise ne correspond pas à une fin de trait"
                            )
                        self._draw_stroke(stroke)
                self._click_at(self.calibration.back_button, palette_delay=True)

            self.callbacks.progress(self.plan.pixel_count, self.plan.pixel_count)
            self.callbacks.status("Dessin terminé")
            self._set_state(AutomationState.COMPLETED)
        except AutomationStopped:
            self.callbacks.status("Dessin interrompu en toute sécurité")
            self._set_state(AutomationState.STOPPED)
        except Exception as exc:
            self.callbacks.status(str(exc))
            self._set_state(AutomationState.ERROR)
            raise
        finally:
            if self._mouse:
                self._mouse.release_all()

    def _select_family(self, column: int, row: int) -> None:
        self._click_at(self.calibration.main_color_point(column, row))
        self._click_at(self.calibration.palette_icon, palette_delay=True)

    def _open_verified_family(
        self,
        column: int,
        row: int,
        expected_group: int,
        family_name: str,
    ) -> None:
        """Open a family with bounded, state-aware recovery.

        A known MAIN or SHADES screen can be navigated safely.  An UNKNOWN
        screen is only observed again; no speculative click is ever sent.
        """

        for opening in range(1, self._palette_open_attempts + 1):
            self._select_family(column, row)
            if self._verify_selected_group(expected_group, family_name):
                return

            observation = self._last_palette_observation
            if observation is None or observation.state is PaletteState.UNKNOWN:
                if self.force_verification_failures:
                    self.callbacks.status(
                        f"Mode Sans échec — famille « {family_name} » supposée ouverte"
                    )
                    return
                raise RuntimeError(
                    f"La famille « {family_name} » reste illisible après "
                    f"{self._palette_read_attempts} contrôles de trois captures. "
                    "Aucun clic de récupération n'a été tenté sur cet écran "
                    "inconnu et aucun pixel de cette famille n'a été dessiné. "
                    "Remettez Heartopia au premier plan sur sa palette, puis reprenez. "
                    f"Dernière lecture : {self._last_palette_summary}."
                )

            if opening == self._palette_open_attempts:
                break
            self.callbacks.status(
                f"{family_name} non confirmée — récupération automatique "
                f"{opening + 1}/{self._palette_open_attempts}"
            )
            # MAIN means that the palette-icon click was lost. SHADES means
            # Back is visible and can safely be used before selecting again.
            if observation.state is PaletteState.SHADES:
                self._click_at(self.calibration.back_button, palette_delay=True)

        if self.force_verification_failures:
            self.callbacks.status(
                f"Mode Sans échec — dernière sélection « {family_name} » conservée"
            )
            return
        restored = self._restore_main_palette_after_failure(family_name)
        recovery_message = (
            "La palette principale a été restaurée"
            if restored
            else "Aucun clic spéculatif n'a été envoyé"
        )
        raise RuntimeError(
            f"Impossible de confirmer la famille « {family_name} » après "
            f"{self._palette_open_attempts} ouvertures automatiques. {recovery_message} ; "
            "aucun pixel de cette famille n'a été dessiné. Vérifiez les repères "
            "P, N, Nuances et Retour, affichez la palette principale, puis reprenez. "
            f"Dernière lecture : {self._last_palette_summary}."
        )

    def _verify_selected_group(self, expected_group: int, family_name: str) -> bool:
        if self.verify_group is None:
            return True
        self._last_palette_observation = None
        retry_delay = min(0.35, self.timing.palette_settle_seconds * 0.35)
        for attempt in range(1, self._palette_read_attempts + 1):
            self._checkpoint()
            result = self.verify_group(expected_group)
            if isinstance(result, PaletteObservation):
                self._last_palette_observation = result
                self._last_palette_summary = result.summary
                self.callbacks.status(
                    f"Vérification de {family_name} "
                    f"{attempt}/{self._palette_read_attempts} — {result.summary}"
                )
                if result.matches_expected:
                    return True
                if result.recovery in (
                    PaletteRecoveryAction.BACK_TO_MAIN,
                    PaletteRecoveryAction.REOPEN_SHADES,
                ):
                    return False
            elif result:
                self._last_palette_observation = None
                return True
            if attempt < self._palette_read_attempts:
                self.callbacks.status(
                    f"Vérification de {family_name} — lecture "
                    f"{attempt + 1}/{self._palette_read_attempts}"
                )
                self._sleep(retry_delay)
        return False

    def _restore_main_palette_after_failure(self, family_name: str) -> bool:
        observation = self._last_palette_observation
        if observation is not None and observation.state is PaletteState.MAIN:
            return True
        if observation is not None and observation.state is PaletteState.UNKNOWN:
            return False
        self.callbacks.status(f"{family_name} non reconnue — retour à la palette principale")
        self._click_at(self.calibration.back_button, palette_delay=True)
        return True

    def _draw_stroke(self, stroke: Stroke) -> None:
        self._drag_stroke(stroke)
        with self._state_lock:
            self._painted += stroke.length
            painted = self._painted
        self.callbacks.progress(painted, self.plan.pixel_count)
        self._sleep(self.timing.between_strokes_seconds)

    def _drag_stroke(self, stroke: Stroke) -> None:
        """Paint one logical component with a single held-button gesture."""

        self._checkpoint()
        assert self._mouse is not None
        trace = stroke.trace_cells
        first = self.calibration.canvas.cell_center(
            trace[0].column,
            trace[0].row,
            self.plan.width,
            self.plan.height,
        )
        self._mouse.move_to(first, self._varied(self.timing.cursor_move_seconds))
        self._motion_checkpoint()
        self._mouse.left_down()
        try:
            self._motion_sleep(self.timing.stroke_cell_seconds)
            for cell in trace[1:]:
                self._motion_checkpoint()
                point = self.calibration.canvas.cell_center(
                    cell.column,
                    cell.row,
                    self.plan.width,
                    self.plan.height,
                )
                self._mouse.move_to(
                    point,
                    self._varied(self.timing.stroke_cell_seconds),
                )
            if len(trace) > 1:
                self._motion_sleep(self.timing.stroke_cell_seconds)
        finally:
            self._mouse.left_up()

    def _motion_sleep(self, seconds: float) -> None:
        """Wait inside a gesture without pausing while the button is held."""

        deadline = time.monotonic() + self._varied(seconds)
        while time.monotonic() < deadline:
            self._motion_checkpoint()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.01, remaining))

    def _click_at(
        self,
        point: Point,
        palette_delay: bool = False,
        *,
        jitter: bool = True,
        safe_region: ScreenRect | None = None,
    ) -> None:
        assert self._mouse is not None
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

    def _validate_tool_markers(self) -> None:
        error = tool_marker_geometry_error(self.calibration)
        if error is not None:
            if self.force_verification_failures:
                self.callbacks.status(
                    f"Mode Sans échec — géométrie des outils acceptée : {error}"
                )
                return
            raise RuntimeError(f"Calibration des outils refusée : {error}")

    def _move_cursor_off_canvas(self) -> None:
        # The cursor and hover effects can cover cells or tint the shade swatches.
        self._checkpoint()
        assert self._mouse is not None
        self._mouse.move_to(
            self.calibration.pencil_tool,
            self._varied(self.timing.cursor_move_seconds),
        )

    def _select_and_verify_pencil(self) -> None:
        for attempt in range(1, self._tool_selection_attempts + 1):
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
                "Crayon non confirmé — nouvelle sélection "
                f"({attempt}/{self._tool_selection_attempts})"
            )
        if self.force_verification_failures:
            self.callbacks.status(
                "Mode Sans échec — dernier clic Crayon accepté sans confirmation"
            )
            return
        raise RuntimeError(
            "Le crayon n'a pas pu être confirmé après trois sélections. "
            "Aucun clic n'a été envoyé sur la toile."
        )

    def _select_and_verify_shade(self, palette_index: int) -> None:
        color = PALETTE_BY_INDEX[palette_index]
        shade_point = self.calibration.shade_point(
            color.shade_column,
            color.shade_row,
        )
        last_summary = "aucune observation disponible"
        for attempt in range(1, 4):
            self._click_at(shade_point)
            if self.verify_shade is None:
                return
            self._move_cursor_off_canvas()
            result = self.verify_shade(palette_index)
            if isinstance(result, ShadeSelectionObservation):
                last_summary = result.summary
                self.callbacks.status(f"Nuance observée — {result.summary}")
                accepted = result.matches_expected
            elif isinstance(result, bool):
                last_summary = "validation booléenne négative"
                accepted = result
            else:
                raise TypeError("La vérification de nuance a renvoyé un résultat invalide")
            if accepted:
                return
            self.callbacks.status(
                f"{color.name} non confirmée — nouvelle sélection ({attempt}/3)"
            )
        if self.force_verification_failures:
            self.callbacks.status(
                f"Mode Sans échec — nuance « {color.name} » acceptée sans confirmation"
            )
            return
        raise RuntimeError(
            f"La nuance « {color.name} » n'a pas pu être confirmée après trois essais. "
            "Aucun pixel de cette nuance n'a été envoyé sur la toile. Vérifiez son contour "
            "blanc ; ajustez le repère uniquement si le clic tombe ailleurs. "
            f"Dernière observation : {last_summary}."
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
        self._handle_pause_hotkey()
        # Event.wait() sans timeout rendrait F12 inopérant tant que le runner est
        # en pause. Le bouton Arrêter réveille explicitement la porte, mais l'arrêt
        # d'urgence doit rester global et ne pas dépendre de l'interface Qt.
        while not self._run_gate.wait(timeout=0.05):
            self._motion_checkpoint()
            self._handle_pause_hotkey()
        self._motion_checkpoint()

    def _handle_pause_hotkey(self) -> None:
        if self._mouse is None or not self._mouse.pause_toggle_requested():
            return
        with self._state_lock:
            if self._state is AutomationState.RUNNING:
                self._run_gate.clear()
                self._set_state(AutomationState.PAUSED)
                self.callbacks.status("Pause demandée avec F8")
            elif self._state is AutomationState.PAUSED:
                self._run_gate.set()
                self._set_state(AutomationState.RUNNING)
                self.callbacks.status("Reprise demandée avec F8")

    def _motion_checkpoint(self) -> None:
        if self._stop.is_set():
            raise AutomationStopped
        if self._mouse is not None and self._mouse.emergency_stop_requested():
            self._stop.set()
            raise AutomationStopped

    def _varied(self, value: float) -> float:
        return self.timing.varied(value, self.random)

    def _validate_resume_point(self) -> None:
        if not 0 <= self.resume_from_pixels <= self.plan.pixel_count:
            raise ValueError("Le point de reprise est hors du plan de peinture")
        completed = 0
        if self.resume_from_pixels == 0:
            return
        for family in self.plan.families:
            for shade in family.shades:
                for stroke in shade.strokes:
                    completed += stroke.length
                    if completed == self.resume_from_pixels:
                        return
                    if completed > self.resume_from_pixels:
                        break
        if self.resume_from_pixels != self.plan.pixel_count:
            raise ValueError("Le point de reprise doit se situer à la fin d'un trait")

    def _set_state(self, state: AutomationState) -> None:
        with self._state_lock:
            self._state = state
        self.callbacks.state(state)
