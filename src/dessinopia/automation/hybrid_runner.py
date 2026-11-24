from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from dessinopia.automation.runner import (
    AutomationCallbacks,
    AutomationRunner,
    AutomationState,
    AutomationStopped,
    MouseController,
)
from dessinopia.core.fill_planner import (
    FillOperation,
    HybridPaintingPlan,
    HybridShadePass,
    PencilOperation,
)
from dessinopia.core.models import CalibrationProfile
from dessinopia.core.palette import PALETTE_BY_INDEX
from dessinopia.core.path_planner import Cell, Stroke, build_continuous_strokes
from dessinopia.core.timing import TimingProfile
from dessinopia.vision.fill_verifier import FillCheck, FillCheckState
from dessinopia.vision.palette_observer import PaletteObservation
from dessinopia.vision.shade_observer import ShadeSelectionObservation
from dessinopia.vision.tool_observer import DrawingTool, ToolObservation


class FillVerificationPhase(StrEnum):
    BEFORE = "avant"
    AFTER = "après"


ToolVerifier = Callable[[DrawingTool], bool | ToolObservation]
FillVerifier = Callable[
    [FillOperation, FillVerificationPhase],
    FillCheck | FillCheckState,
]


class HybridAutomationRunner(AutomationRunner):
    """Execute a pencil/bucket plan with physical safety checks.

    Progress is committed only at complete hybrid-operation boundaries. Pencil
    repairs are checked up to three times before committing the operation.
    """

    def __init__(
        self,
        plan: HybridPaintingPlan,
        calibration: CalibrationProfile,
        timing: TimingProfile,
        callbacks: AutomationCallbacks | None = None,
        verify_group: Callable[[int], bool | PaletteObservation] | None = None,
        verify_tool: ToolVerifier | None = None,
        verify_shade: Callable[[int], bool | ShadeSelectionObservation] | None = None,
        verify_fill: FillVerifier | None = None,
        mouse_factory: Callable[[Callable[[], None]], MouseController] | None = None,
        seed: int | None = None,
        resume_from_pixels: int = 0,
        force_verification_failures: bool = False,
    ) -> None:
        self.verify_fill = verify_fill
        self.manual_recovery_required = False
        super().__init__(
            plan,  # type: ignore[arg-type]
            calibration,
            timing,
            callbacks=callbacks,
            verify_group=verify_group,
            verify_tool=verify_tool,
            verify_shade=verify_shade,
            mouse_factory=mouse_factory,
            seed=seed,
            resume_from_pixels=resume_from_pixels,
            force_verification_failures=force_verification_failures,
        )
        self.plan: HybridPaintingPlan
        if self.plan.has_fills and (
            verify_group is None
            or verify_tool is None
            or verify_shade is None
            or verify_fill is None
        ):
            raise ValueError(
                "Un plan avec pot exige les vérifications de palette, d'outil, "
                "de nuance et de remplissage"
            )

    @property
    def total_pixels(self) -> int:
        return self.plan.metrics.target_pixels

    def run(self) -> None:
        with self._state_lock:
            if self._state not in (AutomationState.IDLE, AutomationState.STOPPED):
                raise RuntimeError("Cette exécution est déjà active")
            stopped_before_start = (
                self._state is AutomationState.IDLE and self._stop.is_set()
            )
            if not stopped_before_start:
                self._stop.clear()
                self._run_gate.set()
        if stopped_before_start:
            self.callbacks.status("Dessin interrompu avant son lancement")
            self._set_state(AutomationState.STOPPED)
            return

        self._painted = self.resume_from_pixels
        self._checked_resume_revision = 0
        self._mouse = None
        self._active_tool = DrawingTool.UNKNOWN
        try:
            self._validate_tool_markers()
            self._mouse = self.mouse_factory(self._motion_checkpoint)
            self._set_state(AutomationState.RUNNING)
            self.callbacks.progress(self._painted, self.total_pixels)
            pixels_to_skip = self.resume_from_pixels
            brush_selected = False

            if self.plan.has_fills:
                if pixels_to_skip < self.total_pixels:
                    self._tool_handshake()
                else:
                    self.callbacks.status(
                        "Checkpoint complet — remise de sécurité sur le crayon"
                    )
                    self._select_and_verify_tool(DrawingTool.PENCIL)

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
                self._ensure_tool(DrawingTool.PENCIL)
                if not brush_selected:
                    self.callbacks.status("Sélection du pinceau le plus fin")
                    self._click_at(self.calibration.small_brush)
                    brush_selected = True

                for shade_number, shade in enumerate(family.shades, start=1):
                    self._checkpoint()
                    if pixels_to_skip >= shade.pixel_count:
                        pixels_to_skip -= shade.pixel_count
                        continue
                    color = PALETTE_BY_INDEX[shade.palette_index]
                    self.callbacks.status(
                        f"{color.name} — nuance {shade_number}/{len(family.shades)}"
                    )
                    self._ensure_tool(DrawingTool.PENCIL)
                    self._select_and_verify_shade(shade.palette_index)
                    pixels_to_skip = self._execute_shade(shade, pixels_to_skip)
                    # Palette navigation is never attempted while the bucket is
                    # active, including between two shades of the same family.
                    self._ensure_tool(DrawingTool.PENCIL)

                self._ensure_tool(DrawingTool.PENCIL)
                self._click_at(self.calibration.back_button, palette_delay=True)

            if pixels_to_skip:
                raise RuntimeError("Le point de reprise dépasse les opérations restantes")
            if self._active_tool is not DrawingTool.UNKNOWN:
                self._ensure_tool(DrawingTool.PENCIL)
            self.callbacks.progress(self.total_pixels, self.total_pixels)
            self.callbacks.status("Dessin hybride terminé")
            self._set_state(AutomationState.COMPLETED)
        except AutomationStopped:
            # Deliberately no tool/palette cleanup click here: F12 must stop all
            # automation immediately. The next run performs a fresh handshake.
            self.callbacks.status("Dessin interrompu en toute sécurité")
            self._set_state(AutomationState.STOPPED)
        except Exception as exc:
            self.callbacks.status(str(exc))
            self._set_state(AutomationState.ERROR)
            raise
        finally:
            if self._mouse:
                self._mouse.release_all()

    def _tool_handshake(self) -> None:
        self.callbacks.status("Test de sécurité des outils — pot puis crayon")
        self._select_and_verify_tool(DrawingTool.FILL)
        self._select_and_verify_tool(DrawingTool.PENCIL)

    def _ensure_tool(self, expected: DrawingTool) -> None:
        if self._active_tool is expected:
            return
        self._select_and_verify_tool(expected)

    def _execute_shade(self, shade: HybridShadePass, pixels_to_skip: int) -> int:
        # This explicit two-loop order is a safety invariant: every planned fence
        # is physically drawn before the first bucket click of this shade.
        for operation in shade.pencil_operations:
            if pixels_to_skip >= operation.pixel_count:
                pixels_to_skip -= operation.pixel_count
                continue
            if pixels_to_skip:
                raise RuntimeError(
                    "Le point de reprise ne correspond pas à une fin d'opération"
                )
            self._draw_pencil_operation(operation)

        for operation in shade.fill_operations:
            if pixels_to_skip >= operation.filled_pixel_count:
                pixels_to_skip -= operation.filled_pixel_count
                continue
            if pixels_to_skip:
                raise RuntimeError(
                    "Le point de reprise ne correspond pas à une fin d'opération"
                )
            self._execute_fill_operation(operation)
        return pixels_to_skip

    def _execute_fill_operation(self, operation: FillOperation) -> None:
        while True:
            self._checkpoint()
            revision = self._resume_revision
            before, missing_cells = self._verify_fill_operation(
                operation, FillVerificationPhase.BEFORE
            )
            self._checkpoint()
            if revision != self._resume_revision:
                continue
            if before is FillCheckState.ALREADY_FILLED:
                self.callbacks.status("Zone déjà remplie — reprise validée sans nouveau clic")
                self._commit_pixels(operation.filled_pixel_count)
                return
            if before is FillCheckState.FALLBACK:
                self._repair_and_verify_atomically(operation, missing_cells)
                return
            if before is not FillCheckState.READY:
                raise RuntimeError(
                    "Le contrôle avant remplissage a renvoyé un état incohérent : "
                    f"{before.value}."
                )

            self._ensure_tool(DrawingTool.FILL)
            self._prepare_paint(operation.palette_index, DrawingTool.FILL)
            if revision == self._resume_revision:
                break
            # Une pause peut avoir modifié la toile après le contrôle du contour.
        seed = self.calibration.canvas.cell_center(
            operation.seed.column,
            operation.seed.row,
            self.plan.width,
            self.plan.height,
        )
        self._click_at(seed, jitter=False)
        after, missing_cells = self._verify_fill_operation(
            operation, FillVerificationPhase.AFTER
        )
        self._checkpoint()
        if after is FillCheckState.SUCCESS:
            self._commit_pixels(operation.filled_pixel_count)
            return
        if after is FillCheckState.INCOMPLETE:
            self.callbacks.status("Remplissage incomplet — finition atomique au crayon")
            self._repair_and_verify_atomically(operation, missing_cells)
            return
        if after is FillCheckState.OVERFLOW:
            self.manual_recovery_required = True
            raise RuntimeError(
                "Débordement du pot de peinture détecté. Automatisation arrêtée ; "
                "inspectez la toile avant toute reprise."
            )
        raise RuntimeError(
            "Le résultat du remplissage est visuellement incertain. Automatisation "
            "arrêtée avant toute autre action sur la toile."
        )

    def _verify_fill_operation(
        self,
        operation: FillOperation,
        phase: FillVerificationPhase,
    ) -> tuple[FillCheckState, tuple[Cell, ...] | None]:
        if self.verify_fill is None:
            return (
                FillCheckState.READY
                if phase is FillVerificationPhase.BEFORE
                else FillCheckState.SUCCESS,
                None,
            )
        self._move_cursor_off_canvas()
        result = self.verify_fill(operation, phase)
        if isinstance(result, FillCheck):
            self.callbacks.status(
                f"Contrôle du remplissage {phase.value} — {result.summary}"
            )
            return result.state, result.missing_cells
        if isinstance(result, FillCheckState):
            self.callbacks.status(
                f"Contrôle du remplissage {phase.value} — {result.value}"
            )
            return result, None
        raise TypeError("La vérification de remplissage a renvoyé un résultat invalide")

    def _draw_pencil_operation(self, operation: PencilOperation) -> None:
        self._ensure_tool(DrawingTool.PENCIL)
        self._trace_stroke(operation.stroke)
        self._commit_pixels(operation.pixel_count)
        self._sleep(self.timing.between_strokes_seconds)

    def _repair_and_verify_atomically(
        self,
        operation: FillOperation,
        missing_cells: tuple[Cell, ...] | None,
    ) -> None:
        self._ensure_tool(DrawingTool.PENCIL)
        for attempt in range(1, 4):
            self._select_and_verify_shade(operation.palette_index)
            if missing_cells:
                count = len(missing_cells)
                detail = "1 case à reprendre" if count == 1 else f"{count} cases à reprendre"
                strokes = build_continuous_strokes(missing_cells, operation.palette_index)
            else:
                detail = "zone entière"
                strokes = operation.repair_strokes
            self.callbacks.status(f"Retouche au crayon ({attempt}/3) — {detail}")
            for stroke in strokes:
                self._trace_stroke(stroke)
                self._sleep(self.timing.between_strokes_seconds)
            self._checkpoint()
            result, observed_missing = self._verify_fill_operation(
                operation, FillVerificationPhase.AFTER
            )
            self._checkpoint()
            if result is FillCheckState.SUCCESS:
                self._commit_pixels(operation.filled_pixel_count)
                return
            if result is FillCheckState.OVERFLOW:
                self.manual_recovery_required = True
                raise RuntimeError(
                    "Une cellule extérieure a pris la couleur cible pendant la "
                    "retouche au crayon. Inspectez la toile avant de reprendre."
                )
            if observed_missing is not None:
                missing_cells = observed_missing
        self.callbacks.status(
            "Zone non confirmée après trois retouches au crayon — poursuite du dessin"
        )
        self._commit_pixels(operation.filled_pixel_count)

    def _trace_stroke(self, stroke: Stroke) -> None:
        self._drag_stroke(stroke)

    def _commit_pixels(self, count: int) -> None:
        with self._state_lock:
            self._painted += count
            painted = self._painted
        if painted > self.total_pixels:
            raise RuntimeError("Le moteur hybride a dépassé le nombre de pixels cible")
        self.callbacks.progress(painted, self.total_pixels)

    def _validate_resume_point(self) -> None:
        total = self.plan.metrics.target_pixels
        if not 0 <= self.resume_from_pixels <= total:
            raise ValueError("Le point de reprise est hors du plan de peinture")
        if self.resume_from_pixels == 0:
            return
        completed = 0
        for operation in self.plan.operations:
            completed += (
                operation.pixel_count
                if isinstance(operation, PencilOperation)
                else operation.filled_pixel_count
            )
            if completed == self.resume_from_pixels:
                return
            if completed > self.resume_from_pixels:
                break
        if self.resume_from_pixels != total:
            raise ValueError(
                "Le point de reprise doit se situer à la fin d'une opération hybride"
            )
