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
from dessinopia.core.path_planner import Stroke
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
    """Execute a lossless pencil/bucket plan with physical safety checks.

    Progress is committed only at complete hybrid-operation boundaries. A pencil
    fallback for one bucket operation is consequently atomic: interruption may
    redraw cells, but it can never cause a logical cell to be skipped on resume.
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
        self.verify_tool = verify_tool
        self.verify_fill = verify_fill
        self._active_tool = DrawingTool.UNKNOWN
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

    def _open_verified_family(
        self,
        column: int,
        row: int,
        expected_group: int,
        family_name: str,
    ) -> None:
        """Use the same bounded, state-aware recovery as the pencil runner."""

        super()._open_verified_family(
            column,
            row,
            expected_group,
            family_name,
        )

    def _tool_handshake(self) -> None:
        self.callbacks.status("Test de sécurité des outils — pot puis crayon")
        self._select_and_verify_tool(DrawingTool.FILL)
        self._select_and_verify_tool(DrawingTool.PENCIL)

    def _ensure_tool(self, expected: DrawingTool) -> None:
        if self._active_tool is expected:
            return
        self._select_and_verify_tool(expected)

    def _select_and_verify_tool(self, expected: DrawingTool) -> None:
        point = (
            self.calibration.pencil_tool
            if expected is DrawingTool.PENCIL
            else self.calibration.fill_tool
        )
        safe_region = (
            self.calibration.pencil_region
            if expected is DrawingTool.PENCIL
            else self.calibration.fill_region
        )
        for attempt in range(1, self._tool_selection_attempts + 1):
            self.callbacks.status(
                f"Activation de l'outil {expected.value} — contrôle "
                f"{attempt}/{self._tool_selection_attempts}"
            )
            self._click_at(point, safe_region=safe_region)
            if self.verify_tool is None:
                self._active_tool = expected
                return
            result = self.verify_tool(expected)
            if isinstance(result, ToolObservation):
                self.callbacks.status(f"Outil observé — {result.summary}")
                accepted = result.matches(expected)
            elif isinstance(result, bool):
                accepted = result
            else:
                raise TypeError("La vérification d'outil a renvoyé un résultat invalide")
            if accepted:
                self._active_tool = expected
                return
        if self.force_verification_failures:
            self._active_tool = expected
            self.callbacks.status(
                f"Mode Sans échec — outil {expected.value} accepté sans confirmation"
            )
            return
        self._active_tool = DrawingTool.UNKNOWN
        raise RuntimeError(
            f"L'outil {expected.value} n'a pas pu être confirmé. "
            "Aucun clic n'a été envoyé sur la toile."
        )

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
        self._checkpoint()
        before = self._verify_fill_operation(operation, FillVerificationPhase.BEFORE)
        self._checkpoint()
        if before is FillCheckState.ALREADY_FILLED:
            self.callbacks.status("Zone déjà remplie — reprise validée sans nouveau clic")
            self._commit_pixels(operation.filled_pixel_count)
            return
        if before is FillCheckState.FALLBACK:
            before = self._redraw_fence_and_recheck(operation)
            if before is FillCheckState.ALREADY_FILLED:
                self.callbacks.status(
                    "Zone déjà remplie après reprise du contour — validation sans clic"
                )
                self._commit_pixels(operation.filled_pixel_count)
                return
            if before is FillCheckState.FALLBACK:
                self.callbacks.status(
                    "Frontière toujours incertaine — réparation complète au crayon"
                )
                self._repair_and_verify_atomically(operation)
                return
        if before is FillCheckState.OVERFLOW:
            self.manual_recovery_required = True
            raise RuntimeError(
                "Une cellule extérieure porte déjà la couleur cible avant le "
                "remplissage. Automatisation arrêtée pour éviter de masquer une fuite."
            )
        if before is not FillCheckState.READY:
            raise RuntimeError(
                "Le contrôle avant remplissage a renvoyé un état incohérent : "
                f"{before.value}."
            )

        self._ensure_tool(DrawingTool.FILL)
        seed = self.calibration.canvas.cell_center(
            operation.seed.column,
            operation.seed.row,
            self.plan.width,
            self.plan.height,
        )
        self._click_at(seed, jitter=False)
        after = self._verify_fill_operation(operation, FillVerificationPhase.AFTER)
        self._checkpoint()
        if after is FillCheckState.SUCCESS:
            self._commit_pixels(operation.filled_pixel_count)
            return
        if after is FillCheckState.INCOMPLETE:
            self.callbacks.status("Remplissage incomplet — finition atomique au crayon")
            self._repair_and_verify_atomically(operation)
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

    def _redraw_fence_and_recheck(self, operation: FillOperation) -> FillCheckState:
        """Repair a missed contour without abandoning an otherwise safe bucket."""

        self._ensure_tool(DrawingTool.PENCIL)
        for attempt in range(1, 3):
            self._select_and_verify_shade(operation.palette_index)
            self.callbacks.status(
                f"Contour incertain — reprise sécurisée de la frontière ({attempt}/2)"
            )
            for stroke in operation.fence_strokes:
                self._trace_stroke(stroke)
                self._sleep(self.timing.between_strokes_seconds)
            self._checkpoint()
            result = self._verify_fill_operation(
                operation,
                FillVerificationPhase.BEFORE,
            )
            self._checkpoint()
            if result is not FillCheckState.FALLBACK:
                return result
        return FillCheckState.FALLBACK

    def _verify_fill_operation(
        self,
        operation: FillOperation,
        phase: FillVerificationPhase,
    ) -> FillCheckState:
        if self.verify_fill is None:
            return (
                FillCheckState.READY
                if phase is FillVerificationPhase.BEFORE
                else FillCheckState.SUCCESS
            )
        result = self.verify_fill(operation, phase)
        if isinstance(result, FillCheck):
            self.callbacks.status(
                f"Contrôle du remplissage {phase.value} — {result.summary}"
            )
            return result.state
        if isinstance(result, FillCheckState):
            self.callbacks.status(
                f"Contrôle du remplissage {phase.value} — {result.value}"
            )
            return result
        raise TypeError("La vérification de remplissage a renvoyé un résultat invalide")

    def _draw_pencil_operation(self, operation: PencilOperation) -> None:
        self._ensure_tool(DrawingTool.PENCIL)
        self._trace_stroke(operation.stroke)
        self._commit_pixels(operation.pixel_count)
        self._sleep(self.timing.between_strokes_seconds)

    def _repair_and_verify_atomically(self, operation: FillOperation) -> None:
        """Redraw the complete fence and interior, then prove the result.

        A failed BEFORE check may mean that a previously committed fence stroke
        was physically lost. Repainting only the interior would therefore leave
        a wrong image. The repair strokes deliberately cover both the fence and
        this fill region; only the still-uncommitted interior advances progress.
        """

        self._ensure_tool(DrawingTool.PENCIL)
        for attempt in range(1, 4):
            # A lost shade-selection click can be the reason the fence did not
            # match. Reassert the exact shade before each physical repair pass.
            self._select_and_verify_shade(operation.palette_index)
            self.callbacks.status(
                "Réparation par traits continus de la frontière et de la zone "
                f"({attempt}/3)"
            )
            for stroke in operation.repair_strokes:
                self._trace_stroke(stroke)
                self._sleep(self.timing.between_strokes_seconds)
            # No stroke above updates _painted. If F12 interrupts any of them,
            # the complete repair operation is replayed idempotently on resume.
            self._checkpoint()
            result = self._verify_fill_operation(operation, FillVerificationPhase.AFTER)
            self._checkpoint()
            if result is FillCheckState.SUCCESS:
                self._commit_pixels(operation.filled_pixel_count)
                return
            if result is FillCheckState.OVERFLOW:
                self.manual_recovery_required = True
                raise RuntimeError(
                    "Une cellule sentinelle porte la couleur du remplissage après la "
                    "réparation au crayon. Inspectez la toile avant toute reprise."
                )
            if attempt < 3:
                self.callbacks.status(
                    "Réparation au crayon non confirmée — nouvelle passe complète "
                    f"({attempt + 1}/3)"
                )
        if self.force_verification_failures:
            self.callbacks.status(
                "Mode Sans échec — zone entièrement repassée au crayon et validée de force"
            )
            self._commit_pixels(operation.filled_pixel_count)
            return
        raise RuntimeError(
            "La zone réparée au crayon n'a pas pu être confirmée après trois passes. "
            "Automatisation arrêtée avant de poursuivre."
        )

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
