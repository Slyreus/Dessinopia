from __future__ import annotations

import threading
import time
from collections.abc import Callable
from random import Random

from dessinopia.automation.motion import randomized_click_point, ui_click_radius
from dessinopia.automation.runner import (
    AutomationCallbacks,
    AutomationState,
    AutomationStopped,
    MouseController,
)
from dessinopia.automation.win32_input import Win32Mouse
from dessinopia.core.models import CalibrationProfile, Point
from dessinopia.core.timing import TimingProfile
from dessinopia.vision.detector import PaletteState
from dessinopia.vision.palette_observer import PaletteObservation, PaletteRecoveryAction


class PaletteRecoveryRunner:
    """Ramène Heartopia à sa palette principale, sans peindre.

    Seul le bouton Retour peut être cliqué, et uniquement quand un menu de
    nuances est reconnu avec suffisamment de confiance. Un clic perdu peut être
    rejoué jusqu'à trois fois après une nouvelle observation stable. Une capture
    inconnue est relue au plus cinq fois et ne provoque jamais de clic spéculatif.
    """

    _unknown_read_attempts = 5
    _back_click_attempts = 3

    def __init__(
        self,
        calibration: CalibrationProfile,
        timing: TimingProfile,
        observe_main: Callable[[], PaletteObservation | bool],
        callbacks: AutomationCallbacks | None = None,
        mouse_factory: Callable[[Callable[[], None]], MouseController] | None = None,
        seed: int | None = None,
    ) -> None:
        self.calibration = calibration
        self.timing = timing
        self.observe_main = observe_main
        self.callbacks = callbacks or AutomationCallbacks()
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
                raise RuntimeError("Cette récupération de palette est déjà active")
            stopped_before_start = self._state is AutomationState.IDLE and self._stop.is_set()
            if not stopped_before_start:
                self._stop.clear()
                self._run_gate.set()
        if stopped_before_start:
            self.callbacks.status("Récupération interrompue avant son lancement")
            self._set_state(AutomationState.STOPPED)
            return

        self._mouse = None
        try:
            self._mouse = self.mouse_factory(self._motion_checkpoint)
            self._set_state(AutomationState.RUNNING)
            self.callbacks.progress(0, 1)
            self.callbacks.status("Recherche de la palette principale")
            self._recover_main_palette()
            self.callbacks.progress(1, 1)
            self.callbacks.status("Palette principale prête")
            self._set_state(AutomationState.COMPLETED)
        except AutomationStopped:
            self.callbacks.status("Récupération interrompue en toute sécurité")
            self._set_state(AutomationState.STOPPED)
        except Exception as exc:
            self.callbacks.status(str(exc))
            self._set_state(AutomationState.ERROR)
            raise
        finally:
            if self._mouse is not None:
                self._mouse.release_all()

    def _recover_main_palette(self) -> None:
        unknown_reads = 0
        back_clicks = 0
        while True:
            self._checkpoint()
            observation = self.observe_main()
            if self._is_main(observation):
                return

            if isinstance(observation, PaletteObservation) and (
                observation.state is PaletteState.SHADES
            ):
                unknown_reads = 0
                if observation.recovery is not PaletteRecoveryAction.BACK_TO_MAIN:
                    raise RuntimeError(
                        "Un menu de nuances est visible, mais son état n'est pas assez fiable "
                        "pour cliquer. Ajustez les repères Nuances et Retour."
                    )
                if back_clicks >= self._back_click_attempts:
                    raise RuntimeError(
                        "Trois clics Retour confirmés n'ont pas ramené à la palette "
                        "principale. Ajustez le repère Retour puis relancez la reprise."
                    )
                back_clicks += 1
                self.callbacks.status(
                    "Nuances ouvertes — retour à la palette principale "
                    f"({back_clicks}/{self._back_click_attempts})"
                )
                self._click_at(self.calibration.back_button, palette_delay=True)
                continue

            unknown_reads += 1
            if unknown_reads >= self._unknown_read_attempts:
                raise RuntimeError(
                    "Impossible de retrouver la palette principale après 5 lectures. "
                    "Ouvrez la planche Heartopia ou ajustez les repères Palette et Retour."
                )
            self.callbacks.status(
                "Palette incertaine — nouvelle lecture "
                f"{unknown_reads + 1}/{self._unknown_read_attempts}"
            )
            self._sleep(self._retry_delay())

    @staticmethod
    def _is_main(observation: PaletteObservation | bool) -> bool:
        if isinstance(observation, bool):
            return observation
        return observation.state is PaletteState.MAIN and observation.matches_expected

    def _click_at(self, point: Point, *, palette_delay: bool = False) -> None:
        assert self._mouse is not None
        self._checkpoint()
        target = randomized_click_point(
            point,
            self.random,
            ui_click_radius(self.calibration),
        )
        self._mouse.move_to(target, self._varied(self.timing.cursor_move_seconds))
        self._mouse.click()
        delay = (
            self.timing.palette_settle_seconds
            if palette_delay
            else self.timing.click_settle_seconds
        )
        self._sleep(delay)

    def _retry_delay(self) -> float:
        return min(0.35, self.timing.palette_settle_seconds * 0.35)

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
