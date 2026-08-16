from __future__ import annotations

import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from dessinopia import __display_version__
from dessinopia.automation.calibration_cleanup import CalibrationCleanupRunner
from dessinopia.automation.calibration_test import (
    CalibrationTestPattern,
    build_five_point_pattern,
)
from dessinopia.automation.hybrid_runner import (
    FillVerificationPhase,
    HybridAutomationRunner,
)
from dessinopia.automation.palette_recovery import PaletteRecoveryRunner
from dessinopia.automation.resume_journal import (
    PlanResumeJournal,
    RecoveryState,
    ResumeJournalStore,
    ResumeMode,
)
from dessinopia.automation.runner import (
    AutomationCallbacks,
    AutomationRunner,
    AutomationState,
)
from dessinopia.core.converter import ConversionOptions, ConversionResult, ImageConverter
from dessinopia.core.fill_planner import (
    FillOperation,
    HybridPaintingPlan,
    build_hybrid_fill_plan,
)
from dessinopia.core.models import (
    PRECISIONS,
    AspectRatio,
    CalibrationProfile,
    DitherMode,
    FitMode,
    Precision,
    ScreenRect,
    canvas_size,
    reference_profile,
)
from dessinopia.core.palette import PALETTE
from dessinopia.core.path_planner import PaintingPlan, build_painting_plan
from dessinopia.core.timing import TIMING_PRESETS
from dessinopia.settings import SettingsStore
from dessinopia.ui.calibration_dialog import CalibrationDialog
from dessinopia.ui.overlay import GridOverlay, OverlayToolbar
from dessinopia.ui.overlay_editor import CalibrationOverlayEditor
from dessinopia.ui.widgets import ImageDropLabel, PixelPreview, rgba_to_qimage
from dessinopia.vision.capture import (
    HeartopiaWindowLocator,
    ScreenCapture,
    localize_profile,
    monitor_containing_point,
)
from dessinopia.vision.detector import (
    HeartopiaDetector,
    PaletteState,
    canvas_matches_calibration,
    canvas_matches_grid_aspect,
)
from dessinopia.vision.fill_verifier import FillCanvasVerifier, FillCheckState
from dessinopia.vision.hole_detector import audit_canvas_holes, repair_indices
from dessinopia.vision.palette_observer import PaletteObservation, PaletteObserver
from dessinopia.vision.shade_observer import (
    ShadeSelectionObservation,
    ShadeSelectionObserver,
)
from dessinopia.vision.tool_observer import DrawingTool, ToolObservation, ToolObserver

AutomationWorker = AutomationRunner | CalibrationCleanupRunner | PaletteRecoveryRunner


class AutomationThread(QThread):
    state_changed = Signal(str)
    status_changed = Signal(str)
    progress_changed = Signal(int, int)
    failed = Signal(str)

    def __init__(self, runner: AutomationWorker) -> None:
        super().__init__()
        self.runner = runner
        self.runner.callbacks = AutomationCallbacks(
            state=lambda state: self.state_changed.emit(state.value),
            status=self.status_changed.emit,
            progress=self.progress_changed.emit,
        )

    def run(self) -> None:
        try:
            self.runner.run()
        except Exception as exc:
            self.failed.emit(str(exc) or type(exc).__name__)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            f"Dessinopia {__display_version__} — Heartopia Painting Assistant"
        )
        self.resize(1320, 820)
        self.setMinimumSize(1100, 700)
        self.store = SettingsStore()
        self.settings = self.store.load()
        self.converter = ImageConverter()
        self.detector = HeartopiaDetector()
        self.palette_observer = PaletteObserver()
        self.shade_observer = ShadeSelectionObserver()
        self.tool_observer = ToolObserver()
        self.fill_verifier = FillCanvasVerifier()
        self.capture = ScreenCapture()
        self.locator = HeartopiaWindowLocator()
        self._calibration_was_remapped = False
        self._adapt_saved_calibration_to_display()
        self.source_path: Path | None = None
        self.result: ConversionResult | None = None
        self.plan: PaintingPlan | None = None
        self.hybrid_plan: HybridPaintingPlan | None = None
        self.runner: AutomationWorker | None = None
        self.runner_thread: AutomationThread | None = None
        self._run_mode = "idle"
        self._resume_pixels = 0
        self._resume_fill_mode: bool | None = None
        self._manual_recovery_required = False
        self._project_controls_locked = False
        self._resume_journal: PlanResumeJournal | None = None
        self._last_resume_journal_save = 0.0
        self._calibration_pattern: CalibrationTestPattern | None = None
        self._validated_calibration_signature: tuple[object, ...] | None = None
        self._palette_recovery_continuation: str | None = None
        self._diagnostic_log_path: Path | None = None
        self._diagnostic_lock = threading.Lock()
        self._force_continue_unlocked = False
        self.overlay = GridOverlay()
        self.overlay_toolbar = OverlayToolbar()
        self.calibration_editor = CalibrationOverlayEditor()
        self._return_to_overlay_after_calibration = False
        self._build_ui()
        self._connect_overlay()
        self._apply_settings()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(22, 18, 22, 16)
        root.setSpacing(14)

        header = QHBoxLayout()
        brand = QVBoxLayout()
        title = QLabel("Dessinopia")
        title.setObjectName("title")
        subtitle = QLabel(
            "Convertissez une image, calibrez Heartopia, puis peignez couleur par couleur."
        )
        subtitle.setObjectName("subtitle")
        brand.addWidget(title)
        brand.addWidget(subtitle)
        header.addLayout(brand)
        header.addStretch()
        self.detection_label = QLabel("Heartopia non calibré")
        self.detection_label.setObjectName("statusPill")
        self.overlay_button = QPushButton("Détecter / afficher l'overlay")
        self.overlay_button.setToolTip(
            "Tenter la détection puis ouvrir l'overlay, même avec des repères à ajuster"
        )
        self.overlay_button.clicked.connect(self.show_overlay)
        calibration_button = QPushButton("Ajuster sur l’overlay")
        calibration_button.clicked.connect(self.open_calibration)
        numeric_button = QPushButton("Valeurs…")
        numeric_button.setToolTip("Réglage numérique avancé de la calibration")
        numeric_button.clicked.connect(self.open_numeric_calibration)
        header.addWidget(self.detection_label)
        header.addWidget(self.overlay_button)
        header.addWidget(calibration_button)
        header.addWidget(numeric_button)
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        left = self._build_settings_card()
        center = self._build_preview_card()
        right = self._build_palette_card()
        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setSizes([300, 710, 260])
        root.addWidget(splitter, 1)

        footer = QHBoxLayout()
        self.status_label = QLabel("Importez une image pour commencer.")
        self.status_label.setObjectName("muted")
        footer.addWidget(self.status_label)
        footer.addStretch()
        version = QLabel(f"Version {__display_version__}")
        version.setObjectName("muted")
        footer.addWidget(version)
        credit = QLabel(
            "Créé par <b>Slyreus</b> · "
            '<a href="https://paypal.me/FlorianButel">Soutenir via PayPal</a>'
        )
        credit.setOpenExternalLinks(True)
        credit.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        footer.addWidget(credit)
        root.addLayout(footer)
        self.setCentralWidget(central)

    def _card(self, title_text: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(11)
        title = QLabel(title_text)
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        return frame, layout

    def _build_settings_card(self) -> QFrame:
        frame, layout = self._card("1. Image et format")
        self.drop_area = ImageDropLabel()
        self.drop_area.file_dropped.connect(self.load_image)
        layout.addWidget(self.drop_area)
        import_button = QPushButton("Importer une image")
        import_button.clicked.connect(self.choose_image)
        layout.addWidget(import_button)

        form = QFormLayout()
        form.setSpacing(9)
        self.ratio_combo = QComboBox()
        for ratio in AspectRatio:
            self.ratio_combo.addItem(ratio.value, ratio.value)
        self.ratio_combo.currentIndexChanged.connect(self._ratio_changed)
        form.addRow("Format", self.ratio_combo)
        self.precision_combo = QComboBox()
        for precision in PRECISIONS:
            self.precision_combo.addItem(precision.value, precision.value)
        self.precision_combo.currentIndexChanged.connect(self._settings_changed)
        form.addRow("Définition", self.precision_combo)
        self.grid_label = QLabel("")
        self.grid_label.setObjectName("statusPill")
        form.addRow("Grille logique", self.grid_label)
        self.fit_combo = QComboBox()
        for mode in FitMode:
            self.fit_combo.addItem(mode.value, mode.value)
        self.fit_combo.currentIndexChanged.connect(self._settings_changed)
        form.addRow("Adaptation", self.fit_combo)
        self.dither_combo = QComboBox()
        for mode in DitherMode:
            self.dither_combo.addItem(mode.value, mode.value)
        self.dither_combo.currentIndexChanged.connect(self._settings_changed)
        form.addRow("Tramage", self.dither_combo)
        self.tempo_combo = QComboBox()
        self.tempo_combo.addItem("Très lent (anti-surcharge)", "tres_lent")
        self.tempo_combo.addItem("Lent", "lent")
        self.tempo_combo.addItem("Prudent", "prudent")
        self.tempo_combo.addItem("Équilibré", "equilibre")
        self.tempo_combo.addItem("Rapide", "rapide")
        self.tempo_combo.setToolTip(
            "Temporisation des déplacements et des clics. Le mode Très lent réduit "
            "fortement le rythme envoyé à Heartopia et convient aux systèmes chargés."
        )
        self.tempo_combo.currentIndexChanged.connect(self._settings_changed)
        form.addRow("Tempo", self.tempo_combo)
        layout.addLayout(form)

        alpha_label = QLabel("Seuil de transparence")
        alpha_label.setObjectName("muted")
        layout.addWidget(alpha_label)
        alpha_row = QHBoxLayout()
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 255)
        self.alpha_slider.valueChanged.connect(self._alpha_changed)
        self.alpha_value = QLabel("96")
        alpha_row.addWidget(self.alpha_slider)
        alpha_row.addWidget(self.alpha_value)
        layout.addLayout(alpha_row)
        self.grid_checkbox = QCheckBox("Afficher la grille sur l'aperçu")
        self.grid_checkbox.toggled.connect(self._grid_toggled)
        layout.addWidget(self.grid_checkbox)
        self.fill_checkbox = QCheckBox("Autoriser le remplissage sécurisé")
        self.fill_checkbox.setToolTip(
            "Utiliser le pot uniquement pour les zones dont les frontières sont prouvées fermées"
        )
        self.fill_checkbox.toggled.connect(self._fill_mode_changed)
        layout.addWidget(self.fill_checkbox)
        self.force_continue_checkbox = QCheckBox("Sans échec (forcer les vérifications)")
        self.force_continue_checkbox.setEnabled(False)
        self.force_continue_checkbox.setToolTip(
            "Se débloque après une première erreur. Force les contrôles non concluants, "
            "mais ne neutralise ni F12 ni un débordement du pot confirmé."
        )
        self.force_continue_checkbox.toggled.connect(self._force_mode_changed)
        layout.addWidget(self.force_continue_checkbox)
        layout.addStretch()
        self.convert_button = QPushButton("Convertir l'image")
        self.convert_button.setObjectName("primary")
        self.convert_button.setEnabled(False)
        self.convert_button.clicked.connect(self.convert_image)
        layout.addWidget(self.convert_button)
        return frame

    def _build_preview_card(self) -> QFrame:
        frame, layout = self._card("2. Prévisualisation")
        self.preview_tabs = QTabWidget()
        self.source_preview = PixelPreview(pixelated=False)
        self.converted_preview = PixelPreview(pixelated=True)
        self.preview_tabs.addTab(self.source_preview, "Image source")
        self.preview_tabs.addTab(self.converted_preview, "Conversion Heartopia")
        layout.addWidget(self.preview_tabs, 1)
        self.stats_label = QLabel("Aucune conversion")
        self.stats_label.setObjectName("muted")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.stats_label)
        action_row = QHBoxLayout()
        self.export_button = QPushButton("Exporter l'aperçu PNG")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_preview)
        action_row.addStretch()
        action_row.addWidget(self.export_button)
        action_row.addStretch()
        layout.addLayout(action_row)
        return frame

    def _build_palette_card(self) -> QFrame:
        frame, layout = self._card("3. Plan de peinture")
        help_label = QLabel(
            "Couleurs triées par utilisation. L'exécution les regroupe par famille."
        )
        help_label.setWordWrap(True)
        help_label.setObjectName("muted")
        layout.addWidget(help_label)
        self.color_list = QListWidget()
        layout.addWidget(self.color_list, 1)
        self.plan_label = QLabel("Aucun plan")
        self.plan_label.setWordWrap(True)
        self.plan_label.setObjectName("muted")
        layout.addWidget(self.plan_label)
        warning = QLabel("F12 : arrêt d'urgence global")
        warning.setObjectName("statusPill")
        warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(warning)
        return frame

    def _connect_overlay(self) -> None:
        self.overlay_toolbar.start_requested.connect(self.start_automation)
        self.overlay_toolbar.test_requested.connect(self.start_calibration_test)
        self.overlay_toolbar.pause_requested.connect(self.pause_automation)
        self.overlay_toolbar.resume_requested.connect(self.resume_automation)
        self.overlay_toolbar.reset_requested.connect(self.reset_resume)
        self.overlay_toolbar.stop_requested.connect(self.stop_automation)
        self.overlay_toolbar.calibrate_requested.connect(self.open_calibration)
        self.overlay_toolbar.repair_requested.connect(self.start_hole_repair)
        self.overlay_toolbar.close_requested.connect(self.close_overlay)
        self.calibration_editor.save.connect(self._save_overlay_calibration)
        self.calibration_editor.cancel.connect(self._cancel_overlay_calibration)

    def _apply_settings(self) -> None:
        controls = (
            self.ratio_combo,
            self.precision_combo,
            self.fit_combo,
            self.dither_combo,
            self.tempo_combo,
            self.alpha_slider,
            self.grid_checkbox,
            self.fill_checkbox,
        )
        previous_signal_states = [control.blockSignals(True) for control in controls]
        try:
            self._select_combo(self.ratio_combo, self.settings.ratio)
            self._select_combo(self.precision_combo, self.settings.precision)
            self._select_combo(self.fit_combo, self.settings.fit_mode)
            self._select_combo(self.dither_combo, self.settings.dither_mode)
            self._select_combo(self.tempo_combo, self.settings.tempo)
            self.alpha_slider.setValue(self.settings.alpha_threshold)
            self.grid_checkbox.setChecked(self.settings.show_grid)
            self.fill_checkbox.setChecked(self.settings.allow_fill)
        finally:
            for control, was_blocked in zip(controls, previous_signal_states, strict=True):
                control.blockSignals(was_blocked)
        self.converted_preview.set_grid_visible(self.settings.show_grid)
        self.alpha_value.setText(str(self.alpha_slider.value()))
        self._update_grid_label()
        if self.settings.calibration:
            confidence = round(self.settings.calibration.confidence * 100)
            if self._calibration_was_remapped:
                bounds = self.settings.calibration.reference_bounds
                self.detection_label.setText(
                    f"Calibré · ajusté à {round(bounds.width)}×{round(bounds.height)}"
                )
            else:
                self.detection_label.setText(f"Calibré · {confidence}%")

    def _adapt_saved_calibration_to_display(self) -> None:
        """Remap persisted manual markers when the game/display bounds changed."""

        profile = self.settings.calibration
        if profile is None:
            return
        try:
            window = self.locator.find()
            monitors = self.capture.monitor_bounds()
        except Exception:
            return

        source = profile.reference_bounds
        # Profiles written before Beta 4 did not persist their virtual-desktop
        # origin. Recover it from the monitor that still contains the saved canvas.
        if not (
            source.left <= profile.canvas.center.x < source.right
            and source.top <= profile.canvas.center.y < source.bottom
        ):
            legacy_monitor = monitor_containing_point(monitors, profile.canvas.center)
            if legacy_monitor is not None:
                source = ScreenRect(
                    legacy_monitor.left,
                    legacy_monitor.top,
                    source.width,
                    source.height,
                )
                profile = replace(
                    profile,
                    reference_left=source.left,
                    reference_top=source.top,
                )

        if window is not None:
            target = window.rect
        else:
            target = monitor_containing_point(monitors, profile.canvas.center)
            if target is None and monitors:
                target = min(
                    monitors,
                    key=lambda bounds: (
                        bounds.center.x - profile.canvas.center.x
                    ) ** 2
                    + (
                        bounds.center.y - profile.canvas.center.y
                    ) ** 2,
                )
        if target is None:
            return

        changed = any(
            abs(current - expected) > 0.5
            for current, expected in zip(
                (
                    source.left,
                    source.top,
                    source.width,
                    source.height,
                ),
                (
                    target.left,
                    target.top,
                    target.width,
                    target.height,
                ),
                strict=True,
            )
        )
        if changed:
            profile = profile.remapped_to(target)
            self._calibration_was_remapped = True
        elif profile != self.settings.calibration:
            # Persist the recovered origin for a legacy multi-monitor profile.
            profile = replace(
                profile,
                reference_left=target.left,
                reference_top=target.top,
            )
        else:
            return
        self.settings.calibration = profile
        self.store.save(self.settings)

    @staticmethod
    def _select_combo(combo: QComboBox, value: object) -> None:
        expected = getattr(value, "value", value)
        for index in range(combo.count()):
            if combo.itemData(index) == expected:
                combo.setCurrentIndex(index)
                return

    def choose_image(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Choisir une image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
        )
        if path:
            self.load_image(path)

    def load_image(self, path: str) -> None:
        if self._automation_is_active():
            self.status_label.setText(
                "Image inchangée : arrêtez ou terminez le dessin avant d'en charger une autre."
            )
            return
        candidate = Path(path)
        try:
            with Image.open(candidate) as image:
                source = ImageOps.exif_transpose(image).convert("RGBA")
                rgba = __import__("numpy").asarray(source, dtype="uint8")
        except Exception as exc:
            QMessageBox.critical(self, "Image illisible", str(exc))
            return
        self.source_path = candidate
        self.source_preview.set_image(rgba_to_qimage(rgba))
        self.drop_area.setText(f"{candidate.name}\n{source.width} × {source.height}")
        self.convert_button.setEnabled(True)
        self.status_label.setText("Image chargée. Conversion en cours…")
        self.convert_image()

    def convert_image(self) -> None:
        if self._automation_is_active():
            self.status_label.setText(
                "Conversion différée : le plan actif est verrouillé jusqu'à la fin du dessin."
            )
            return
        if not self.source_path:
            return
        options = self._current_conversion_options()
        try:
            self.result = self.converter.convert_path(self.source_path, options)
            self.plan = build_painting_plan(self.result.indices)
            self.hybrid_plan = build_hybrid_fill_plan(
                self.result.indices,
                enable_fill=self.fill_checkbox.isChecked(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Conversion impossible", str(exc))
            return
        self._resume_pixels = 0
        self._resume_fill_mode = None
        self._manual_recovery_required = False
        self._validated_calibration_signature = None
        self._restore_resume_journal()
        self._update_resume_ui()
        self.converted_preview.set_image(rgba_to_qimage(self.result.preview_rgba))
        self.converted_preview.set_grid_visible(self.grid_checkbox.isChecked())
        self.preview_tabs.setCurrentWidget(self.converted_preview)
        self.stats_label.setText(
            f"{self.result.width} × {self.result.height} · "
            + (
                f"{self.result.unique_colors} couleurs · "
                f"{self.result.painted_pixels:,} pixels à peindre"
            ).replace(",", " ")
        )
        assert self.plan is not None
        self._update_plan_summary()
        self._populate_colors()
        self.overlay_button.setEnabled(True)
        self.export_button.setEnabled(True)
        if self._resume_pixels or self._manual_recovery_required:
            self.status_label.setText(
                "Conversion terminée · point de reprise compatible retrouvé sur disque."
            )
        else:
            self.status_label.setText(
                "Conversion terminée. Vérifiez l'aperçu puis calibrez Heartopia."
            )
        self._persist_controls()

    def _update_plan_summary(self) -> None:
        if self.plan is None:
            return
        hybrid = self.hybrid_plan
        if hybrid is not None and hybrid.fill_enabled and hybrid.has_fills:
            metrics = hybrid.metrics
            seconds = self.settings.timing.estimate_seconds(
                metrics.direct_pixels,
                metrics.pencil_strokes,
                hybrid.shade_count,
                len(hybrid.families),
            )
            # A bucket operation includes tool navigation and two visual safety
            # controls; include a conservative allowance in the displayed estimate.
            seconds += metrics.fill_clicks * (
                2 * self.settings.timing.cursor_move_seconds
                + self.settings.timing.click_settle_seconds
                + 0.35
            )
            saved = (
                f"{metrics.saved_pixel_actions:,} gestes pixel évités"
            ).replace(",", " ")
            self.plan_label.setText(
                f"{len(hybrid.families)} familles · {hybrid.shade_count} nuances\n"
                f"{metrics.pencil_strokes} contours/traits continus + "
                f"{metrics.fill_clicks} remplissages sûrs\n"
                f"{saved} ({metrics.savings_ratio:.0%})\n"
                f"Durée estimée : {seconds / 60:.1f} min"
            )
            return
        seconds = self.settings.timing.estimate_seconds(
            self.plan.pixel_count,
            self.plan.stroke_count,
            self.plan.shade_count,
            len(self.plan.families),
        )
        fill_note = ""
        if hybrid is not None and hybrid.fill_enabled:
            metrics = hybrid.metrics
            fill_note = (
                "\nAucune zone remplissable sûre · crayon uniquement"
                f"\n{metrics.candidate_components} candidates analysées · "
                f"{metrics.unsafe_connectivity_components} refusées par la sécurité"
            )
        self.plan_label.setText(
            f"{len(self.plan.families)} familles\n{self.plan.shade_count} nuances\n"
            f"{self.plan.stroke_count} gestes crayon continus{fill_note}\n"
            f"Durée estimée : {seconds / 60:.1f} min"
        )

    def _populate_colors(self) -> None:
        self.color_list.clear()
        if not self.result:
            return
        for palette_index, count in sorted(
            self.result.color_counts().items(), key=lambda item: item[1], reverse=True
        ):
            color = PALETTE[palette_index]
            pixmap = QPixmap(22, 22)
            pixmap.fill(QColor(color.hex_value))
            item = QListWidgetItem(QIcon(pixmap), f"{color.name}  ·  {count}")
            item.setToolTip(color.hex_value.upper())
            self.color_list.addItem(item)

    def export_preview(self) -> None:
        if not self.result:
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, "Exporter la conversion", "dessinopia.png", "PNG (*.png)"
        )
        if path:
            Image.fromarray(self.result.preview_rgba, "RGBA").save(path)
            self.status_label.setText(f"Aperçu exporté : {path}")

    def _grab_for_vision(self, bounds):  # type: ignore[no-untyped-def]
        """Capture Heartopia without permanently hiding the overlay from streams."""

        windows = (self.overlay, self.overlay_toolbar)
        for window in windows:
            window.set_capture_excluded(True)
        try:
            return self.capture.grab(bounds)
        finally:
            for window in windows:
                window.set_capture_excluded(False)

    def detect_heartopia(self, *, show_warning: bool = True) -> bool:
        if self._automation_is_active():
            self.status_label.setText(
                "Détection différée : arrêtez ou terminez le dessin avant de recalibrer."
            )
            return False
        window = self.locator.find()
        profile = None
        try:
            if window:
                screenshot = self._grab_for_vision(window.rect)
                expected = self._expected_canvas_for_capture(window.rect, screenshot)
                profile = self.detector.auto_profile(
                    screenshot,
                    expected_canvas=expected,
                )
                if profile:
                    profile = profile.translated(window.rect.left, window.rect.top)
            else:
                monitors = self.capture.monitor_bounds()
                regions = monitors or (self.capture.virtual_bounds(),)
                for bounds in regions:
                    screenshot = self._grab_for_vision(bounds)
                    expected = self._expected_canvas_for_capture(bounds, screenshot)
                    candidate = self.detector.auto_profile(
                        screenshot,
                        expected_canvas=expected,
                    )
                    if candidate is None:
                        continue
                    candidate = candidate.translated(bounds.left, bounds.top)
                    if profile is None or candidate.confidence > profile.confidence:
                        profile = candidate
        except Exception as exc:
            if show_warning:
                QMessageBox.warning(self, "Capture impossible", str(exc))
            return False
        if profile is None:
            self.detection_label.setText("Détection à ajuster")
            if show_warning:
                QMessageBox.warning(
                    self,
                    "Toile non détectée",
                    "Ouvrez Heartopia en mode fenêtré sans bordure, affichez une toile "
                    "avec son cadre complet, puis réessayez. "
                    "Vous pouvez aussi utiliser la calibration manuelle.",
                )
            return False
        self.settings.calibration = profile
        self.store.save(self.settings)
        self._validated_calibration_signature = None
        self._rebind_resume_journal_after_calibration()
        self.detection_label.setText(f"Heartopia détecté · {profile.confidence:.0%}")
        self.status_label.setText("La toile et les commandes Heartopia ont été calibrées.")
        self._refresh_overlay()
        return True

    def _expected_canvas_for_capture(self, bounds, image):  # type: ignore[no-untyped-def]
        if self.settings.calibration is not None:
            return localize_profile(
                self.settings.calibration,
                bounds,
                image.shape[1],
                image.shape[0],
            ).canvas
        return (
            reference_profile(AspectRatio(self.ratio_combo.currentData()))
            .scaled_to(image.shape[1], image.shape[0])
            .canvas
        )

    def open_calibration(self) -> None:
        if self.runner_thread and self.runner_thread.isRunning():
            QMessageBox.information(
                self.overlay_toolbar,
                "Dessin en cours",
                "Arrêtez le dessin avant de modifier les repères.",
            )
            return
        profile = self._calibration_profile_for_editing()
        if profile is None:
            return
        self._return_to_overlay_after_calibration = (
            self.overlay.isVisible() or self.overlay_toolbar.isVisible()
        )
        self.overlay.hide()
        self.overlay_toolbar.hide()
        self.showMinimized()
        try:
            self.calibration_editor.begin(profile, self.current_canvas_size())
        except Exception as exc:
            self._finish_overlay_calibration()
            QMessageBox.warning(self, "Calibration impossible", str(exc))

    def open_numeric_calibration(self) -> None:
        """Keep the former coordinate form as an advanced fallback."""

        if self.runner_thread and self.runner_thread.isRunning():
            QMessageBox.information(
                self,
                "Dessin en cours",
                "Arrêtez le dessin avant de modifier la calibration.",
            )
            return
        profile = self._calibration_profile_for_editing()
        if profile is None:
            return
        dialog = CalibrationDialog(profile, self)
        if dialog.exec():
            self.settings.calibration = dialog.result_profile()
            self.store.save(self.settings)
            self._validated_calibration_signature = None
            self._rebind_resume_journal_after_calibration()
            self.detection_label.setText("Calibration numérique · enregistrée")
            self.status_label.setText(
                "Calibration numérique enregistrée. Vérifiez-la avec l’overlay visuel."
            )
            self._refresh_overlay()

    def _calibration_profile_for_editing(self) -> CalibrationProfile | None:
        if self.settings.calibration:
            return self.settings.calibration
        try:
            window = self.locator.find()
            if window:
                bounds = window.rect
            else:
                monitors = self.capture.monitor_bounds()
                bounds = monitors[0] if monitors else self.capture.virtual_bounds()
        except Exception as exc:
            QMessageBox.warning(self, "Calibration impossible", str(exc))
            return None
        return (
            reference_profile(AspectRatio(self.ratio_combo.currentData()))
            .scaled_to(round(bounds.width), round(bounds.height))
            .translated(bounds.left, bounds.top)
        )

    def _save_overlay_calibration(self, profile: CalibrationProfile) -> None:
        self.settings.calibration = profile
        self.store.save(self.settings)
        self._validated_calibration_signature = None
        self._rebind_resume_journal_after_calibration()
        self.detection_label.setText("Calibration visuelle · enregistrée")
        self.status_label.setText("Les repères visuels ont été enregistrés.")
        self.calibration_editor.set_calibration_mode(False)
        self.calibration_editor.hide()
        self._refresh_overlay()
        self._finish_overlay_calibration()

    def _cancel_overlay_calibration(self) -> None:
        self.calibration_editor.set_calibration_mode(False)
        self.calibration_editor.hide()
        self._finish_overlay_calibration()

    def _finish_overlay_calibration(self) -> None:
        return_to_overlay = self._return_to_overlay_after_calibration
        self._return_to_overlay_after_calibration = False
        if return_to_overlay and self.settings.calibration:
            self._refresh_overlay()
            self.overlay.show()
            self.overlay_toolbar.show()
            self.overlay_toolbar.raise_()
            return
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def show_overlay(self) -> None:
        if self._automation_is_active():
            self.status_label.setText(
                "L'overlay est déjà utilisé par l'automatisation en cours."
            )
            return
        if self.result is not None and not self._conversion_is_current():
            self.status_label.setText(
                "Le format de conversion a changé : reconvertissez l'image avant l'overlay."
            )
            return
        detected = self.detect_heartopia(show_warning=False)
        fallback_created = False
        if not detected and self.settings.calibration is None:
            profile = self._calibration_profile_for_editing()
            if profile is None:
                self.status_label.setText(
                    "Impossible de préparer les repères de référence pour l'overlay."
                )
                return
            self.settings.calibration = profile
            fallback_created = True
            self.detection_label.setText("Repères de référence à ajuster")
        if self.settings.calibration is None:
            return
        self._refresh_overlay()
        self.overlay.show()
        self.overlay_toolbar.show()
        target_screen = self.overlay.target_screen or self.screen()
        screen = target_screen.availableGeometry()
        self.overlay_toolbar.place_at_top_center(screen)
        self.overlay_toolbar.raise_()
        ready = self.result is not None and self.plan is not None
        self.overlay_toolbar.start_button.setEnabled(ready)
        self.overlay_toolbar.test_button.setEnabled(ready)
        self.overlay_toolbar.repair_button.setEnabled(ready)
        if detected:
            self.overlay_toolbar.status_label.setText(
                "Heartopia détecté · vérifiez les repères avant de démarrer"
            )
        elif fallback_created:
            self.overlay_toolbar.status_label.setText(
                "Toile non reconnue · repères de référence affichés et ajustables"
            )
        else:
            self.overlay_toolbar.status_label.setText(
                "Détection non concluante · repères sauvegardés conservés"
            )
        self.showMinimized()

    def _refresh_overlay(self) -> None:
        if self.settings.calibration:
            self.overlay.configure(self.settings.calibration, self.current_canvas_size())

    def close_overlay(self) -> None:
        if self.runner and self.runner.state in (AutomationState.RUNNING, AutomationState.PAUSED):
            self.stop_automation()
        self.overlay.hide()
        self.overlay_toolbar.hide()
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def start_automation(self) -> None:
        if not self.plan or not self.settings.calibration:
            self.overlay_toolbar.status_label.setText("Conversion ou calibration manquante")
            return
        if self.result is not None and not self._conversion_is_current():
            self.overlay_toolbar.status_label.setText(
                "Paramètres modifiés : reconvertissez l'image avant de dessiner"
            )
            return
        if self.runner_thread and self.runner_thread.isRunning():
            return
        if self._manual_recovery_required:
            response = QMessageBox.question(
                self.overlay_toolbar,
                "Confirmer la correction dans Heartopia",
                "Un débordement avait été confirmé. Dans Heartopia, utilisez Annuler une fois "
                "(ou corrigez manuellement la cellule extérieure), puis affichez de nouveau la "
                "palette principale.\n\nDessinopia conservera le checkpoint et revérifiera la "
                "zone avant tout nouveau pot. Avez-vous effectué cette correction ?",
            )
            if response is not QMessageBox.StandardButton.Yes:
                self.overlay_toolbar.status_label.setText(
                    "Reprise bloquée jusqu'à la correction du débordement dans Heartopia"
                )
                return
            self._manual_recovery_required = False
            self._save_resume_checkpoint(
                self._resume_pixels,
                recovery=RecoveryState.PALETTE_RECHECK_REQUIRED,
                force=True,
            )
        self._start_diagnostic_session("dessin")
        if (
            self._resume_pixels == 0
            and not self._calibration_test_is_valid()
            and not self._force_mode_active()
        ):
            self._prepare_calibration_test()
            return
        self._prepare_drawing()

    def start_calibration_test(self) -> None:
        if not self.plan or not self.settings.calibration:
            self.overlay_toolbar.status_label.setText("Conversion ou calibration manquante")
            return
        if self.runner_thread and self.runner_thread.isRunning():
            return
        if self._resume_pixels:
            QMessageBox.information(
                self.overlay_toolbar,
                "Dessin déjà commencé",
                "Le test des cinq points n'est pas relancé au milieu d'un dessin. "
                "Utilisez Reprendre, ou reconvertissez l'image pour repartir de zéro.",
            )
            return
        self._start_diagnostic_session("test-calibration")
        self._prepare_calibration_test()

    def _prepare_calibration_test(self) -> None:
        if not self.plan or not self.settings.calibration:
            return
        canvas_is_current, canvas_message, override_allowed = self._validate_canvas_before_start()
        if not canvas_is_current and not self._confirm_canvas_override(
            canvas_message,
            allow_continue=override_allowed,
        ):
            return
        if not self._ensure_main_palette("calibration_test"):
            return
        response = QMessageBox.question(
            self.overlay_toolbar,
            "Tester la calibration",
            "Dessinopia va peindre temporairement cinq pixels : rouge en haut à gauche, "
            "jaune en haut à droite, vert au centre, bleu en bas à gauche et violet en bas "
            "à droite.\n\nNe touchez plus à la souris. F12 arrête immédiatement. Continuer ?",
        )
        if response is not QMessageBox.StandardButton.Yes:
            return
        size = self.current_canvas_size()
        self._calibration_pattern = build_five_point_pattern(size.width, size.height)
        test_plan = build_painting_plan(self._calibration_pattern.to_index_matrix())
        active_calibration = self.settings.calibration
        runner = AutomationRunner(
            test_plan,
            active_calibration,
            self.settings.timing,
            verify_group=lambda group: self._observe_palette(
                group,
                active_calibration,
            )
            or False,
            verify_tool=lambda tool: self._verify_tool(
                tool,
                active_calibration,
            ),
            verify_shade=lambda palette_index: self._verify_shade(
                palette_index,
                active_calibration,
            ),
        )
        self._launch_worker(runner, "calibration_test")

    def _prepare_drawing(self, *, confirmation_already_given: bool = False) -> None:
        if not self.plan or not self.result or not self.settings.calibration:
            return
        requested_fill = self.settings.allow_fill
        resume_mode_is_fixed = bool(
            self._resume_pixels or self._manual_recovery_required
        )
        planning_fill = (
            self._resume_fill_mode
            if resume_mode_is_fixed and self._resume_fill_mode is not None
            else requested_fill
        )
        if (
            self.hybrid_plan is None
            or self.hybrid_plan.fill_enabled != planning_fill
        ):
            self.hybrid_plan = build_hybrid_fill_plan(
                self.result.indices,
                enable_fill=planning_fill,
            )
            self._update_plan_summary()
        use_fill = planning_fill and self.hybrid_plan.has_fills
        force_mode = self._force_mode_active()
        canvas_is_current, canvas_message, override_allowed = self._validate_canvas_before_start()
        if not canvas_is_current:
            if force_mode:
                self._append_diagnostic(
                    f"Mode Sans échec — contrôle de toile ignoré : {canvas_message}"
                )
                self.overlay_toolbar.status_label.setText(
                    "Mode Sans échec — position calibrée utilisée sans confirmation visuelle"
                )
            elif not self._confirm_canvas_override(
                canvas_message,
                allow_continue=override_allowed,
            ):
                return
        continuation = "draw_confirmed" if confirmation_already_given else "draw"
        if not self._ensure_main_palette(continuation):
            return
        if not confirmation_already_given:
            checkpoint = ""
            if self._resume_pixels:
                percent = round(self._resume_pixels * 100 / max(1, self.plan.pixel_count))
                retained_mode = (
                    " Le mode de dessin du lancement initial est conservé pour garantir "
                    "le point de reprise."
                    if requested_fill != planning_fill
                    else ""
                )
                checkpoint = (
                    f"\n\nReprise sécurisée à {percent}% : les {self._resume_pixels} pixels "
                    f"déjà validés seront ignorés.{retained_mode}"
                )
            response = QMessageBox.question(
                self.overlay_toolbar,
                "Reprendre le dessin" if self._resume_pixels else "Lancer le dessin",
                "Heartopia doit afficher la palette principale, le même format et la même "
                "définition que Dessinopia."
                f"{checkpoint}\n\nNe touchez plus à la souris pendant le dessin. "
                + (
                    "Le pot sera utilisé uniquement après contrôle de chaque frontière ; "
                    "un doute provoquera un repli au crayon. "
                    if use_fill
                    else ""
                )
                + (
                    "Mode Sans échec actif : les contrôles non concluants seront forcés. "
                    if force_mode
                    else ""
                )
                + "F12 arrête immédiatement. Continuer ?",
            )
            if response is not QMessageBox.StandardButton.Yes:
                return
        if use_fill:
            active_plan = self.hybrid_plan
            active_calibration = self.settings.calibration
            runner = HybridAutomationRunner(
                active_plan,
                active_calibration,
                self.settings.timing,
                verify_group=lambda group: self._observe_palette(
                    group,
                    active_calibration,
                )
                or False,
                verify_tool=lambda tool: self._verify_tool(
                    tool,
                    active_calibration,
                ),
                verify_shade=lambda palette_index: self._verify_shade(
                    palette_index,
                    active_calibration,
                ),
                verify_fill=lambda operation, phase: self._verify_fill(
                    operation,
                    phase,
                    plan=active_plan,
                    calibration=active_calibration,
                ),
                resume_from_pixels=self._resume_pixels,
                force_verification_failures=force_mode,
            )
        else:
            active_calibration = self.settings.calibration
            runner = AutomationRunner(
                self.plan,
                active_calibration,
                self.settings.timing,
                verify_group=lambda group: self._observe_palette(
                    group,
                    active_calibration,
                )
                or False,
                verify_tool=lambda tool: self._verify_tool(
                    tool,
                    active_calibration,
                ),
                verify_shade=lambda palette_index: self._verify_shade(
                    palette_index,
                    active_calibration,
                ),
                resume_from_pixels=self._resume_pixels,
                force_verification_failures=force_mode,
            )
        self._launch_worker(runner, "draw")

    def start_hole_repair(self) -> None:
        """Audit the visible board and repaint background-like target cells."""

        if not self.result or not self.settings.calibration:
            self.overlay_toolbar.status_label.setText(
                "Conversion ou calibration manquante pour l'audit"
            )
            return
        if self.runner_thread and self.runner_thread.isRunning():
            return
        self._start_diagnostic_session("réparation-trous")
        self._prepare_hole_repair()

    def _prepare_hole_repair(self) -> None:
        if self.result is None or self.settings.calibration is None:
            return
        canvas_is_current, canvas_message, override_allowed = self._validate_canvas_before_start()
        if not canvas_is_current and not self._confirm_canvas_override(
            canvas_message,
            allow_continue=override_allowed,
        ):
            return
        if not self._ensure_main_palette("hole_repair"):
            return
        image, profile = self._capture_for_detection()
        if image is None or profile is None:
            self._automation_failed("La toile n'a pas pu être capturée pour l'audit des trous.")
            return
        audit = audit_canvas_holes(image, profile, self.result.indices)
        self._append_diagnostic(
            f"Audit des trous — candidats={audit.hole_count}/{audit.sampled_cells}"
        )
        if not audit.cells:
            self.overlay_toolbar.status_label.setText("Audit terminé — aucun trou détecté")
            QMessageBox.information(
                self.overlay_toolbar,
                "Audit terminé",
                "Aucune cellule ressemblant au fond ivoire n'a été détectée.",
            )
            return
        response = QMessageBox.question(
            self.overlay_toolbar,
            "Réparer les trous",
            f"Dessinopia a trouvé {audit.hole_count} cellules ressemblant encore au "
            "fond ivoire. Les nuances très pâles peuvent être repeintes elles aussi, "
            "ce qui est sans danger.\n\nLa réparation utilisera un clic prudent par "
            "cellule. Continuer ?",
        )
        if response is not QMessageBox.StandardButton.Yes:
            return
        sparse = repair_indices(self.result.indices, audit)
        repair_plan = build_painting_plan(sparse)
        calibration = self.settings.calibration
        runner = AutomationRunner(
            repair_plan,
            calibration,
            TIMING_PRESETS["prudent"],
            verify_group=lambda group: self._observe_palette(group, calibration) or False,
            verify_tool=lambda tool: self._verify_tool(tool, calibration),
            verify_shade=lambda palette_index: self._verify_shade(
                palette_index,
                calibration,
            ),
        )
        self._append_diagnostic(
            f"Réparation des trous — cellules={repair_plan.pixel_count}; "
            "mode=prudent-point-par-point"
        )
        self._launch_worker(runner, "hole_repair")

    def _ensure_main_palette(self, continuation: str) -> bool:
        observation = self._observe_palette()
        if observation is not None and observation.matches_expected:
            self.overlay_toolbar.status_label.setText("Palette principale confirmée")
            return True
        force_mode = self._force_mode_active()
        if force_mode and (
            observation is None or observation.state is PaletteState.UNKNOWN
        ):
            self._append_diagnostic(
                "Mode Sans échec — palette principale supposée visible"
            )
            self.overlay_toolbar.status_label.setText(
                "Mode Sans échec — palette principale supposée visible"
            )
            return True
        self._palette_recovery_continuation = continuation
        assert self.settings.calibration is not None
        runner = PaletteRecoveryRunner(
            self.settings.calibration,
            self.settings.timing,
            observe_main=lambda: self._observe_palette() or False,
        )
        self._launch_worker(runner, "palette_recovery")
        return False

    def _resume_binding(
        self,
    ) -> tuple[PaintingPlan | HybridPaintingPlan, ResumeMode] | None:
        if self.plan is None:
            return None
        if (
            self.hybrid_plan is not None
            and self.hybrid_plan.fill_enabled
            and self.hybrid_plan.has_fills
        ):
            return self.hybrid_plan, ResumeMode.SAFE_FILL
        return self.plan, ResumeMode.PENCIL

    def _restore_resume_journal(self) -> None:
        binding = self._resume_binding()
        calibration = self.settings.calibration
        if binding is None or calibration is None:
            self._resume_journal = None
            return
        plan, mode = binding
        journal = PlanResumeJournal(
            plan,
            mode=mode,
            calibration=calibration,
        )
        self._resume_journal = journal
        entry = journal.load()
        if entry is None:
            return
        self._resume_pixels = entry.validated_pixels
        self._resume_fill_mode = entry.mode is ResumeMode.SAFE_FILL
        self._manual_recovery_required = (
            entry.recovery is RecoveryState.MANUAL_INTERVENTION_REQUIRED
        )
        self.status_label.setText(
            "Point de reprise retrouvé sur disque ; la palette sera revérifiée avant "
            "de continuer."
        )

    def _bind_resume_journal(self, runner: AutomationRunner) -> None:
        if isinstance(runner, HybridAutomationRunner):
            plan: PaintingPlan | HybridPaintingPlan = runner.plan
            mode = ResumeMode.SAFE_FILL
        else:
            plan = runner.plan
            mode = ResumeMode.PENCIL
        self._resume_journal = PlanResumeJournal(
            plan,
            mode=mode,
            calibration=runner.calibration,
        )
        self._last_resume_journal_save = 0.0

    def _rebind_resume_journal_after_calibration(self) -> None:
        """Keep an in-memory checkpoint usable after an explicit visual edit."""

        calibration = self.settings.calibration
        if (
            calibration is None
            or self.plan is None
            or not (self._resume_pixels or self._manual_recovery_required)
        ):
            return
        if (
            self._resume_fill_mode is True
            and self.hybrid_plan is not None
            and self.hybrid_plan.has_fills
        ):
            plan: PaintingPlan | HybridPaintingPlan = self.hybrid_plan
            mode = ResumeMode.SAFE_FILL
        else:
            plan = self.plan
            mode = ResumeMode.PENCIL
        self._resume_journal = PlanResumeJournal(
            plan,
            mode=mode,
            calibration=calibration,
        )
        self._last_resume_journal_save = 0.0
        recovery = (
            RecoveryState.MANUAL_INTERVENTION_REQUIRED
            if self._manual_recovery_required
            else RecoveryState.PALETTE_RECHECK_REQUIRED
        )
        self._save_resume_checkpoint(
            self._resume_pixels,
            recovery=recovery,
            force=True,
        )

    def _save_resume_checkpoint(
        self,
        pixels: int,
        *,
        recovery: RecoveryState,
        force: bool = False,
    ) -> None:
        journal = self._resume_journal
        if journal is None:
            return
        now = time.monotonic()
        # An atomic fsync after every one-pixel stroke would noticeably slow a
        # large drawing. At most 0.75 s of idempotent work is replayed after a
        # process crash; explicit stops and safety events are always flushed.
        if not force and now - self._last_resume_journal_save < 0.75:
            return
        try:
            journal.save_progress(pixels, recovery=recovery)
            self._last_resume_journal_save = now
        except (OSError, TypeError, ValueError) as exc:
            # Persistence must never interrupt painting. The in-memory
            # checkpoint remains available and the diagnostic explains why a
            # restart might replay more work.
            self._append_diagnostic(f"Journal de reprise non enregistré — {exc}")

    def _launch_worker(self, runner: AutomationWorker, mode: str) -> None:
        if self.runner_thread is not None and self.runner_thread.isRunning():
            self.overlay_toolbar.status_label.setText(
                "Une opération Dessinopia est déjà active"
            )
            return
        self.runner = runner
        self._run_mode = mode
        self._set_project_controls_locked(True)
        self._append_diagnostic(f"Lancement du mode {mode}")
        self.runner_thread = AutomationThread(runner)
        self.runner_thread.state_changed.connect(self._automation_state_changed)
        self.runner_thread.status_changed.connect(self._automation_status_changed)
        self.runner_thread.progress_changed.connect(self._automation_progress_changed)
        self.runner_thread.failed.connect(self._automation_failed)
        self.runner_thread.finished.connect(self._automation_thread_finished)
        if mode == "draw" and isinstance(runner, AutomationRunner):
            self._bind_resume_journal(runner)
            self._save_resume_checkpoint(
                runner.painted_pixels,
                recovery=RecoveryState.PALETTE_RECHECK_REQUIRED,
                force=True,
            )
        self.runner_thread.start()

    def _confirm_canvas_override(self, message: str, *, allow_continue: bool = True) -> bool:
        """Let the user override an uncertain vision check without hiding the risk."""

        dialog = QMessageBox(self.overlay_toolbar)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Vérification de la toile")
        dialog.setText(message)
        dialog.setInformativeText(
            (
                "La détection peut varier de quelques pixels. Vous pouvez ajuster les repères "
                "directement sur l’overlay ou continuer avec la calibration affichée."
            )
            if allow_continue
            else (
                "Ouvrez la planche à dessin Heartopia et sa palette principale. Continuer est "
                "désactivé tant qu’aucune toile n’est visible ; vous pouvez ajuster les repères."
            )
        )
        adjust_button = dialog.addButton(
            "Ajuster sur l’overlay",
            QMessageBox.ButtonRole.ActionRole,
        )
        continue_button = None
        if allow_continue:
            continue_button = dialog.addButton(
                "Continuer quand même",
                QMessageBox.ButtonRole.AcceptRole,
            )
        dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(adjust_button)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked is adjust_button:
            self.open_calibration()
            return False
        if continue_button is not None and clicked is continue_button:
            self.overlay_toolbar.status_label.setText(
                "Contrôle visuel ignoré à votre demande — calibration conservée"
            )
            return True
        return False

    def pause_automation(self) -> None:
        if self.runner:
            self.runner.pause()

    def resume_automation(self) -> None:
        if self.runner:
            self.runner.resume()

    def reset_resume(self) -> None:
        """Explicitly abandon the checkpoint and restart the immutable plan at zero."""

        if self._automation_is_active():
            return
        if not (self._resume_pixels or self._manual_recovery_required):
            self.overlay_toolbar.status_label.setText("Aucun point de reprise à réinitialiser")
            return
        response = QMessageBox.question(
            self.overlay_toolbar,
            "Réinitialiser la reprise",
            "Abandonner définitivement le checkpoint et recommencer le dessin depuis 0 % ?\n\n"
            "Cette action n'efface pas la toile dans Heartopia. Les zones déjà peintes "
            "seront simplement redessinées. Corrigez d'abord tout débordement éventuel.",
        )
        if response is not QMessageBox.StandardButton.Yes:
            return
        try:
            if self._resume_journal is not None:
                self._resume_journal.clear()
            else:
                ResumeJournalStore().clear()
        except OSError as exc:
            self._append_diagnostic(f"Journal de reprise non supprimé — {exc}")
        self._resume_journal = None
        self._resume_pixels = 0
        self._resume_fill_mode = None
        self._manual_recovery_required = False
        self._validated_calibration_signature = None
        self.overlay_toolbar.set_progress(0, self.plan.pixel_count if self.plan else 0)
        self._update_resume_ui()
        self.overlay_toolbar.status_label.setText(
            "Reprise réinitialisée — le prochain lancement recommencera à 0 %"
        )

    def stop_automation(self) -> None:
        if self.runner:
            self.runner.stop()

    def _automation_state_changed(self, value: str) -> None:
        self.overlay_toolbar.state_label.setText(value)
        state = AutomationState(value)
        if self._run_mode == "draw" and isinstance(self.runner, AutomationRunner):
            if state in (AutomationState.STOPPED, AutomationState.ERROR):
                self._resume_pixels = max(self._resume_pixels, self.runner.painted_pixels)
                self._resume_fill_mode = isinstance(
                    self.runner,
                    HybridAutomationRunner,
                )
                if isinstance(self.runner, HybridAutomationRunner):
                    self._manual_recovery_required = (
                        self.runner.manual_recovery_required
                    )
                recovery = (
                    RecoveryState.MANUAL_INTERVENTION_REQUIRED
                    if self._manual_recovery_required
                    else RecoveryState.PALETTE_RECHECK_REQUIRED
                )
                self._save_resume_checkpoint(
                    self.runner.painted_pixels,
                    recovery=recovery,
                    force=True,
                )
            elif state is AutomationState.COMPLETED:
                self._resume_pixels = 0
                self._resume_fill_mode = None
                self._manual_recovery_required = False
        running = state in (AutomationState.RUNNING, AutomationState.PAUSED)
        self.overlay_toolbar.set_running(
            running,
            paused=state is AutomationState.PAUSED,
            stopping=state is AutomationState.STOPPING,
        )
        if not running and state is not AutomationState.STOPPING:
            self._update_resume_ui()

    def _automation_status_changed(self, message: str) -> None:
        self.overlay_toolbar.status_label.setText(message)
        self._append_diagnostic(message)

    def _automation_progress_changed(self, done: int, total: int) -> None:
        self.overlay_toolbar.set_progress(done, total)
        if self._run_mode != "draw":
            return
        self._save_resume_checkpoint(
            done,
            recovery=RecoveryState.PALETTE_RECHECK_REQUIRED,
            force=done == total,
        )

    def _automation_thread_finished(self) -> None:
        mode = self._run_mode
        state = self.runner.state if self.runner is not None else AutomationState.ERROR
        self.runner_thread = None
        self._set_project_controls_locked(False)
        if state is not AutomationState.COMPLETED:
            self._run_mode = "idle"
            return
        if mode == "palette_recovery":
            continuation = self._palette_recovery_continuation
            self._palette_recovery_continuation = None
            self._run_mode = "idle"
            if continuation == "calibration_test":
                self._prepare_calibration_test()
            elif continuation in ("draw", "draw_confirmed"):
                self._prepare_drawing(
                    confirmation_already_given=continuation == "draw_confirmed"
                )
            elif continuation == "hole_repair":
                self._prepare_hole_repair()
            return
        if mode == "calibration_test":
            self._run_mode = "idle"
            self._ask_calibration_test_result()
            return
        if mode == "calibration_cleanup":
            self._validated_calibration_signature = self._calibration_signature()
            self._run_mode = "idle"
            self.overlay_toolbar.status_label.setText(
                "Calibration validée — lancement du dessin"
            )
            self._prepare_drawing(confirmation_already_given=True)
            return
        if mode == "draw":
            self._run_mode = "idle"
            if self._resume_journal is not None:
                self._resume_journal.clear()
            self._resume_journal = None
            self._append_diagnostic("Dessin terminé sans erreur")
            return
        if mode == "hole_repair":
            self._run_mode = "idle"
            self._append_diagnostic("Réparation des trous terminée sans erreur")
            self.overlay_toolbar.status_label.setText(
                "Réparation terminée — vous pouvez relancer un audit de contrôle"
            )

    def _automation_is_active(self) -> bool:
        """Return True while a worker owns the immutable drawing session."""

        return self._project_controls_locked or (
            self.runner_thread is not None and self.runner_thread.isRunning()
        )

    def _set_project_controls_locked(self, locked: bool) -> None:
        """Prevent a running worker and the UI from referring to different plans."""

        self._project_controls_locked = locked
        central = self.centralWidget()
        if central is not None:
            central.setEnabled(not locked)
        self.calibration_editor.setEnabled(not locked)

    def _ask_calibration_test_result(self) -> None:
        dialog = QMessageBox(self.overlay_toolbar)
        dialog.setIcon(QMessageBox.Icon.Question)
        dialog.setWindowTitle("Résultat du test des cinq points")
        dialog.setText(
            "Les cinq pixels sont-ils exactement dans les quatre coins et au centre de la toile ?"
        )
        dialog.setInformativeText(
            "Attendu : rouge ↖, jaune ↗, vert au centre, bleu ↙ et violet ↘. "
            "Si vous confirmez, Dessinopia les efface avec la transparence puis lance le dessin."
        )
        accept_button = dialog.addButton(
            "Oui, lancer le dessin",
            QMessageBox.ButtonRole.AcceptRole,
        )
        adjust_button = dialog.addButton(
            "Non, ajuster les repères",
            QMessageBox.ButtonRole.ActionRole,
        )
        dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(accept_button)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked is accept_button:
            self._start_calibration_cleanup()
            return
        self._validated_calibration_signature = None
        if clicked is adjust_button:
            self.overlay_toolbar.status_label.setText(
                "Test refusé — corrigez les repères puis remettez la toile à blanc "
                "avant de retester"
            )
            self.open_calibration()
        else:
            self.overlay_toolbar.status_label.setText(
                "Test non validé — vous gardez la main"
            )

    def _start_calibration_cleanup(self) -> None:
        if self._calibration_pattern is None or self.settings.calibration is None:
            return
        active_calibration = self.settings.calibration
        runner = CalibrationCleanupRunner(
            self._calibration_pattern,
            active_calibration,
            self.settings.timing,
            verify_tool=lambda tool: self._verify_tool(
                tool,
                active_calibration,
            ),
            verify_shade=lambda column, row: self._verify_shade_cell(
                column,
                row,
                active_calibration,
                label="transparence",
                allow_outline_tiebreaker=True,
            ),
        )
        self._launch_worker(runner, "calibration_cleanup")

    def _calibration_signature(self) -> tuple[object, ...] | None:
        if self.settings.calibration is None:
            return None
        size = self.current_canvas_size()
        return (self.settings.calibration, size.width, size.height)

    def _calibration_test_is_valid(self) -> bool:
        signature = self._calibration_signature()
        return signature is not None and signature == self._validated_calibration_signature

    def _update_resume_ui(self) -> None:
        if self.plan and self._manual_recovery_required:
            percent = round(self._resume_pixels * 100 / max(1, self.plan.pixel_count))
            self.overlay_toolbar.start_button.setText(
                f"Vérifier puis reprendre à {percent}%"
            )
            self.overlay_toolbar.status_label.setText(
                "Débordement confirmé : utilisez Annuler dans Heartopia, puis reprenez"
            )
            self.overlay_toolbar.set_resume_available(True)
            return
        if self.plan and self._resume_pixels:
            percent = round(self._resume_pixels * 100 / max(1, self.plan.pixel_count))
            self.overlay_toolbar.start_button.setText(f"Reprendre à {percent}%")
            self.overlay_toolbar.status_label.setText(
                f"Point de reprise conservé : {self._resume_pixels}/{self.plan.pixel_count} pixels"
            )
            self.overlay_toolbar.set_resume_available(True)
            return
        self.overlay_toolbar.start_button.setText("Démarrer")
        self.overlay_toolbar.set_resume_available(False)

    def _automation_failed(self, details: str) -> None:
        self._force_continue_unlocked = True
        self.force_continue_checkbox.setEnabled(True)
        if self._run_mode == "draw" and isinstance(self.runner, AutomationRunner):
            self._resume_pixels = max(self._resume_pixels, self.runner.painted_pixels)
            self._resume_fill_mode = isinstance(
                self.runner,
                HybridAutomationRunner,
            )
            if isinstance(self.runner, HybridAutomationRunner):
                self._manual_recovery_required = self.runner.manual_recovery_required
            recovery = (
                RecoveryState.MANUAL_INTERVENTION_REQUIRED
                if self._manual_recovery_required
                else RecoveryState.PALETTE_RECHECK_REQUIRED
            )
            self._save_resume_checkpoint(
                self.runner.painted_pixels,
                recovery=recovery,
                force=True,
            )
            self._update_resume_ui()
        self._append_diagnostic(f"ERREUR — {details}")
        checkpoint = ""
        if self.plan and (self._resume_pixels or self._manual_recovery_required):
            percent = round(self._resume_pixels * 100 / max(1, self.plan.pixel_count))
            checkpoint = (
                f"\n\nReprise disponible à {percent}% ({self._resume_pixels}/"
                f"{self.plan.pixel_count} pixels)."
            )
        log_hint = (
            f"\n\nDiagnostic enregistré dans :\n{self._diagnostic_log_path}"
            if self._diagnostic_log_path
            else ""
        )
        issue_title, recovery_steps = self._failure_guidance(details)
        self.overlay_toolbar.set_running(False)
        dialog = QMessageBox(self.overlay_toolbar)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle(f"Mise en sécurité — {issue_title}")
        dialog.setText(
            f"Pourquoi Dessinopia s'est mis en sécurité :\n{details}\n\n"
            f"Comment reprendre rapidement :\n{recovery_steps}{checkpoint}\n\n"
            "Le point de reprise s'arrête à la dernière opération entièrement validée. "
            "Le mode Sans échec est maintenant débloqué pour forcer les contrôles "
            "non concluants lors de la reprise."
            f"{log_hint}",
        )
        force_button = dialog.addButton(
            "Activer Sans échec pour la reprise",
            QMessageBox.ButtonRole.AcceptRole,
        )
        dialog.addButton(QMessageBox.StandardButton.Ok)
        dialog.exec()
        if dialog.clickedButton() is force_button:
            self.force_continue_checkbox.setChecked(True)
            self.overlay_toolbar.status_label.setText(
                "Mode Sans échec armé — cliquez sur Reprendre"
            )

    @staticmethod
    def _failure_guidance(details: str) -> tuple[str, str]:
        normalized = details.casefold()
        if "transparence" in normalized or "n6" in normalized:
            return (
                "transparence non confirmée",
                "1. Le clic N6 a été envoyé, mais Dessinopia n'a pas reconnu son "
                "contour blanc.\n2. Vérifiez que N6 est entourée de blanc dans la palette "
                "Neutres ; corrigez uniquement son repère si le clic tombe ailleurs.\n"
                "3. Relancez le nettoyage : aucun des cinq pixels de test n'a été "
                "modifié après ce refus.",
            )
        if any(
            marker in normalized
            for marker in ("débordement", "cellule extérieure", "sentinelle")
        ):
            return (
                "débordement confirmé",
                "1. Dans Heartopia, utilisez Annuler une fois ou corrigez la cellule "
                "extérieure.\n2. Affichez la palette principale.\n3. Cliquez sur "
                "« Vérifier puis reprendre » ; la zone sera recontrôlée avant tout pot.",
            )
        if any(
            marker in normalized
            for marker in ("remplissage", "frontière", "zone réparée")
        ):
            return (
                "zone non confirmée",
                "1. Laissez Heartopia au premier plan quelques secondes et affichez la "
                "palette principale.\n2. Vérifiez que le cadre Toile et sa grille sont "
                "alignés dans « Repères détectés ».\n3. Cliquez sur Reprendre ; la "
                "zone entière sera rejouée au crayon puis recontrôlée avant de continuer.",
            )
        if "nuance" in normalized:
            return (
                "nuance non confirmée",
                "1. Vérifiez si la case indiquée porte bien le contour blanc de sélection "
                "dans Heartopia.\n2. Corrigez son repère dans « Repères détectés » "
                "uniquement si le clic tombe réellement ailleurs.\n3. Cliquez sur "
                "Reprendre ; aucun pixel de cette nuance n'a été envoyé après l'échec.",
            )
        if any(marker in normalized for marker in ("famille", "palette")):
            return (
                "palette non confirmée",
                "1. Remettez Heartopia au premier plan et affichez sa palette "
                "principale.\n2. Si nécessaire, alignez P1…P16, Nuances et Retour dans "
                "« Repères détectés ».\n3. Cliquez sur Reprendre ; Dessinopia "
                "relira la palette avant de toucher la toile.",
            )
        if any(marker in normalized for marker in ("crayon", "outil", "pot")):
            return (
                "outil non confirmé",
                "1. Ouvrez « Repères détectés ».\n2. Alignez les cadres Crayon "
                "et Pot sur leurs boutons à gauche de Heartopia.\n3. Reprenez ; l'outil sera "
                "recliqué et vérifié avant la toile.",
            )
        if any(marker in normalized for marker in ("toile", "capture", "visible")):
            return (
                "Heartopia non lisible",
                "1. Rouvrez Heartopia en fenêtré sans bordure avec la toile visible.\n"
                "2. Vérifiez le cadre Toile dans « Repères détectés ».\n3. Affichez "
                "la palette principale puis cliquez sur Reprendre.",
            )
        return (
            "contrôle non concluant",
            "1. Laissez la toile et la palette principale visibles dans Heartopia.\n"
            "2. Consultez le diagnostic indiqué ci-dessous et vérifiez les repères.\n"
            "3. Cliquez sur Reprendre : le checkpoint sera revérifié.",
        )

    def _start_diagnostic_session(self, label: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        log_directory = self.store.path.parent / "logs"
        try:
            log_directory.mkdir(parents=True, exist_ok=True)
            self._diagnostic_log_path = log_directory / f"automation-{timestamp}.log"
            size = self.current_canvas_size()
            self._diagnostic_log_path.write_text(
                f"Dessinopia — session {label}\n"
                f"Date locale : {datetime.now().isoformat(timespec='seconds')}\n"
                f"Grille : {size.width}x{size.height}\n",
                encoding="utf-8",
            )
            if self.hybrid_plan is not None:
                metrics = self.hybrid_plan.metrics
                self._append_diagnostic(
                    "Plan — "
                    f"remplissage={self.hybrid_plan.fill_enabled}; "
                    f"zones={metrics.fill_clicks}; directs={metrics.direct_pixels}; "
                    f"économie={metrics.saved_pixel_actions}"
                )
        except OSError:
            self._diagnostic_log_path = None

    def _append_diagnostic(self, message: str) -> None:
        path = self._diagnostic_log_path
        if path is None:
            return
        line = f"{datetime.now().isoformat(timespec='milliseconds')} | {message}\n"
        try:
            with self._diagnostic_lock, path.open("a", encoding="utf-8") as stream:
                stream.write(line)
        except OSError:
            pass

    def _capture_for_detection(
        self,
        calibration: CalibrationProfile | None = None,
    ):  # type: ignore[no-untyped-def]
        active_calibration = calibration or self.settings.calibration
        if active_calibration is None:
            return None, None
        window = self.locator.find()
        if window:
            bounds = window.rect
        else:
            monitors = self.capture.monitor_bounds()
            bounds = monitor_containing_point(monitors, active_calibration.canvas.center)
            if bounds is None:
                bounds = self.capture.virtual_bounds()
        image = self._grab_for_vision(bounds)
        profile = localize_profile(
            active_calibration,
            bounds,
            image.shape[1],
            image.shape[0],
        )
        return image, profile

    def _observe_palette(
        self,
        expected_group: int | None = None,
        calibration: CalibrationProfile | None = None,
    ) -> PaletteObservation | None:
        frames = []
        profile = None
        try:
            for index in range(3):
                image, current_profile = self._capture_for_detection(calibration)
                if image is None or current_profile is None:
                    return None
                frames.append(image)
                profile = current_profile
                if index < 2:
                    time.sleep(0.045)
            assert profile is not None
            observation = self.palette_observer.observe_many(
                frames,
                profile,
                expected_group=expected_group,
            )
            diagnostics = observation.diagnostics
            self._append_diagnostic(
                "Palette — "
                f"attendue={expected_group}; {observation.summary}; "
                f"retour={diagnostics.back_button_score:.3f}; "
                f"principale={diagnostics.main_palette_score:.3f}; "
                f"nuances={diagnostics.shade_palette_score:.3f}; "
                f"ΔE={diagnostics.best_group_delta_e}/{diagnostics.second_group_delta_e}; "
                f"correspondances={diagnostics.matching_shades}/{diagnostics.sampled_shades}"
            )
            return observation
        except Exception as exc:
            self._append_diagnostic(f"Lecture de palette impossible — {exc}")
            return None

    def _verify_tool(
        self,
        expected: DrawingTool,
        calibration: CalibrationProfile | None = None,
    ) -> bool | ToolObservation:
        frames = []
        profile = None
        try:
            for index in range(3):
                image, current_profile = self._capture_for_detection(calibration)
                if image is None or current_profile is None:
                    return False
                frames.append(image)
                profile = current_profile
                if index < 2:
                    time.sleep(0.045)
            assert profile is not None
            observation = self.tool_observer.observe_many(frames, profile)
            self._append_diagnostic(
                f"Outil — attendu={expected.value}; {observation.summary}"
            )
            return observation
        except Exception as exc:
            self._append_diagnostic(f"Lecture de l'outil impossible — {exc}")
            return False

    def _verify_shade(
        self,
        palette_index: int,
        calibration: CalibrationProfile | None = None,
    ) -> bool | ShadeSelectionObservation:
        color = PALETTE[palette_index]
        return self._verify_shade_cell(
            color.shade_column,
            color.shade_row,
            calibration,
            label=color.name,
            allow_outline_tiebreaker=True,
        )

    def _verify_shade_cell(
        self,
        column: int,
        row: int,
        calibration: CalibrationProfile | None = None,
        *,
        label: str = "commande",
        allow_outline_tiebreaker: bool = False,
    ) -> bool | ShadeSelectionObservation:
        frames = []
        profile = None
        try:
            for index in range(3):
                image, current_profile = self._capture_for_detection(calibration)
                if image is None or current_profile is None:
                    return False
                frames.append(image)
                profile = current_profile
                if index < 2:
                    time.sleep(0.045)
            assert profile is not None
            observation = self.shade_observer.observe_many(
                frames,
                profile,
                expected_column=column,
                expected_row=row,
                allow_outline_tiebreaker=allow_outline_tiebreaker,
            )
            self._append_diagnostic(
                f"Nuance — attendue={label}; {observation.summary}"
            )
            return observation
        except Exception as exc:
            self._append_diagnostic(f"Lecture de nuance impossible — {exc}")
            return False

    def _verify_fill(
        self,
        operation: FillOperation,
        phase: FillVerificationPhase,
        *,
        plan: HybridPaintingPlan | None = None,
        calibration: CalibrationProfile | None = None,
    ) -> FillCheckState:
        active_plan = plan or self.hybrid_plan
        if active_plan is None:
            return FillCheckState.UNKNOWN
        checks = []
        try:
            for index in range(3):
                image, profile = self._capture_for_detection(calibration)
                if image is None or profile is None:
                    return (
                        FillCheckState.FALLBACK
                        if phase is FillVerificationPhase.BEFORE
                        else FillCheckState.UNKNOWN
                    )
                if phase is FillVerificationPhase.BEFORE:
                    check = self.fill_verifier.check_before(
                        image,
                        profile,
                        active_plan.width,
                        active_plan.height,
                        operation,
                    )
                else:
                    check = self.fill_verifier.check_after(
                        image,
                        profile,
                        active_plan.width,
                        active_plan.height,
                        operation,
                    )
                checks.append(check)
                self._append_diagnostic(
                    f"Pot {phase.value} — composante={operation.component_id}; {check.summary}"
                )
                if index < 2:
                    time.sleep(0.045)
        except Exception as exc:
            self._append_diagnostic(f"Contrôle du remplissage impossible — {exc}")
            return (
                FillCheckState.FALLBACK
                if phase is FillVerificationPhase.BEFORE
                else FillCheckState.UNKNOWN
            )

        overflow_count = sum(
            check.state is FillCheckState.OVERFLOW for check in checks
        )
        if phase is FillVerificationPhase.BEFORE:
            if overflow_count >= 2:
                return FillCheckState.OVERFLOW
            if all(check.state is FillCheckState.ALREADY_FILLED for check in checks):
                return FillCheckState.ALREADY_FILLED
            if all(check.state is FillCheckState.READY for check in checks):
                return FillCheckState.READY
            return FillCheckState.FALLBACK
        success_count = sum(
            check.state is FillCheckState.SUCCESS for check in checks
        )
        if overflow_count >= 2:
            return FillCheckState.OVERFLOW
        if success_count >= 2:
            return FillCheckState.SUCCESS
        # Any non-majority uncertainty is recoverable: the runner switches back
        # to the pencil, redraws the complete fence/zone and checks it again.
        return FillCheckState.INCOMPLETE

    def _validate_canvas_before_start(self) -> tuple[bool, str, bool]:
        try:
            image, profile = self._capture_for_detection()
            if image is None or profile is None:
                return False, "Aucune calibration exploitable n'est disponible.", False
            assessment = self.detector.assess_canvas(image, profile.canvas)
            detection = assessment.detection
            self._append_diagnostic(
                "Toile — "
                f"preuve={assessment.evidence.value}; confiance={assessment.confidence:.3f}; "
                f"rectangle={detection.rect if detection else None}"
            )
            if detection is None:
                # Once a drawing covers enough cells, the formerly uniform light
                # canvas is no longer one connected component.  A positively
                # recognized Heartopia palette is then stronger evidence that the
                # calibrated editor is still open than the canvas fill itself.
                palette = self.detector.detect_palette_state(image, profile)
                if palette.state in (PaletteState.MAIN, PaletteState.SHADES):
                    return True, "", True
                return (
                    False,
                    "La toile de dessin n'est pas visible à sa position calibrée.",
                    False,
                )
            if not canvas_matches_calibration(
                detection.rect,
                profile.canvas,
                tolerance_ratio=0.08,
                minimum_tolerance=24.0,
            ):
                return (
                    False,
                    "La toile détectée a changé de position ou de dimensions depuis "
                    "la calibration.",
                    True,
                )
            if not canvas_matches_grid_aspect(
                detection.rect,
                self.current_canvas_size(),
                tolerance_ratio=0.08,
            ):
                return (
                    False,
                    "Le format de la toile Heartopia ne correspond pas au format "
                    "sélectionné dans Dessinopia.",
                    True,
                )
            self._adopt_detected_canvas(profile, detection.rect, assessment.confidence)
            return True, "", True
        except Exception as exc:
            return False, f"La capture de contrôle a échoué : {exc}", False

    def _adopt_detected_canvas(
        self,
        localized_profile: CalibrationProfile,
        detected_canvas: ScreenRect,
        confidence: float,
    ) -> None:
        """Snap the saved logical grid to the frame pixels observed at launch."""

        calibration = self.settings.calibration
        if calibration is None or confidence < 0.90:
            return
        dx = calibration.canvas.left - localized_profile.canvas.left
        dy = calibration.canvas.top - localized_profile.canvas.top
        desktop_canvas = detected_canvas.translated(dx, dy)
        current = calibration.canvas
        edge_delta = max(
            abs(desktop_canvas.left - current.left),
            abs(desktop_canvas.right - current.right),
            abs(desktop_canvas.top - current.top),
            abs(desktop_canvas.bottom - current.bottom),
        )
        if edge_delta < 0.25:
            return
        self.settings.calibration = replace(
            calibration,
            canvas=desktop_canvas,
            confidence=max(calibration.confidence, confidence),
            source="automatic-pixel-refinement",
        )
        self.store.save(self.settings)
        self._validated_calibration_signature = None
        self._rebind_resume_journal_after_calibration()
        self._refresh_overlay()
        self._append_diagnostic(
            "Toile — grille recalée automatiquement sur les bords observés : "
            f"ancien={current}; nouveau={desktop_canvas}; delta-max={edge_delta:.2f}px"
        )

    def _detect_current_palette_state(self) -> PaletteState | None:
        observation = self._observe_palette()
        return observation.state if observation is not None else None

    def _verify_group(self, expected_group: int) -> bool | PaletteObservation:
        return self._observe_palette(expected_group) or False

    def current_canvas_size(self):  # type: ignore[no-untyped-def]
        return canvas_size(
            AspectRatio(self.ratio_combo.currentData()),
            Precision(self.precision_combo.currentData()),
        )

    def _current_conversion_options(self) -> ConversionOptions:
        return ConversionOptions(
            size=self.current_canvas_size(),
            fit_mode=FitMode(self.fit_combo.currentData()),
            dither_mode=DitherMode(self.dither_combo.currentData()),
            alpha_threshold=self.alpha_slider.value(),
        )

    def _conversion_is_current(self) -> bool:
        return (
            self.result is not None
            and self.result.options == self._current_conversion_options()
        )

    def _ratio_changed(self) -> None:
        self._update_grid_label()
        self._settings_changed()

    def _update_grid_label(self) -> None:
        self.grid_label.setText(self.current_canvas_size().label)

    def _alpha_changed(self, value: int) -> None:
        self.alpha_value.setText(str(value))
        self._settings_changed()

    def _grid_toggled(self, checked: bool) -> None:
        self.converted_preview.set_grid_visible(checked)
        self.settings.show_grid = checked
        self.store.save(self.settings)

    def _fill_mode_changed(self, checked: bool) -> None:
        """Rebuild the hybrid plan immediately when the checkbox changes."""

        self._settings_changed()
        if (
            self.result is None
            or self.plan is None
            or not self._conversion_is_current()
            or self._automation_is_active()
        ):
            return
        self.hybrid_plan = build_hybrid_fill_plan(
            self.result.indices,
            enable_fill=checked,
        )
        self._resume_pixels = 0
        self._resume_fill_mode = None
        self._manual_recovery_required = False
        self._restore_resume_journal()
        self._update_resume_ui()
        self._update_plan_summary()
        if checked and self.hybrid_plan.has_fills:
            metrics = self.hybrid_plan.metrics
            self.status_label.setText(
                f"Remplissage sécurisé actif · {metrics.fill_clicks} zones prévues · "
                f"{metrics.saved_pixel_actions} gestes pixel évités."
            )
        elif checked:
            self.status_label.setText(
                "Remplissage actif, mais aucune zone assez grande et fermée n'a été trouvée."
            )
        else:
            self.status_label.setText("Remplissage désactivé · plan crayon recalculé.")

    def _force_mode_active(self) -> bool:
        return (
            self._force_continue_unlocked
            and self.force_continue_checkbox.isChecked()
        )

    def _force_mode_changed(self, checked: bool) -> None:
        if checked and self._force_continue_unlocked:
            self.status_label.setText(
                "Mode Sans échec armé · les contrôles non concluants seront forcés."
            )
        elif not checked and self._force_continue_unlocked:
            self.status_label.setText("Mode Sans échec désactivé · contrôles normaux restaurés.")

    def _settings_changed(self) -> None:
        self._update_grid_label()
        self._persist_controls()
        if self.source_path:
            if self.result is not None and self._conversion_is_current():
                self.status_label.setText("Paramètres d'exécution enregistrés.")
            else:
                self.status_label.setText("Paramètres modifiés — cliquez sur Convertir.")
        self.overlay_button.setEnabled(not self._automation_is_active())

    def _persist_controls(self) -> None:
        self.settings.ratio = AspectRatio(self.ratio_combo.currentData())
        self.settings.precision = Precision(self.precision_combo.currentData())
        self.settings.fit_mode = FitMode(self.fit_combo.currentData())
        self.settings.dither_mode = DitherMode(self.dither_combo.currentData())
        self.settings.tempo = self.tempo_combo.currentData()
        self.settings.timing = TIMING_PRESETS[self.settings.tempo]
        self.settings.alpha_threshold = self.alpha_slider.value()
        self.settings.show_grid = self.grid_checkbox.isChecked()
        self.settings.allow_fill = self.fill_checkbox.isChecked()
        self.store.save(self.settings)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.stop_automation()
        self.overlay.close()
        self.overlay_toolbar.close()
        self.calibration_editor.close()
        if (
            self.runner_thread
            and self.runner_thread.isRunning()
            and not self.runner_thread.wait(3000)
        ):
            event.ignore()
            QMessageBox.warning(
                self,
                "Arrêt en cours",
                "Dessinopia attend encore le relâchement sécurisé de la souris. "
                "Réessayez de fermer l'application dans quelques secondes.",
            )
            return
        event.accept()
