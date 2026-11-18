from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

from dessinopia.core.fill_planner import (
    FillOperation,
    HybridPaintingPlan,
    PencilOperation,
)
from dessinopia.core.models import CalibrationProfile
from dessinopia.core.path_planner import PaintingPlan

_JOURNAL_VERSION = 2
_FINGERPRINT_VERSION = 1
_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}")
_EMPTY_CALIBRATION_FINGERPRINT = hashlib.sha256(b"no-calibration").hexdigest()

PaintPlan = PaintingPlan | HybridPaintingPlan


def resume_journal_path() -> Path:
    """Return the per-user journal path used by the desktop application."""

    base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    return base / "Dessinopia" / "resume-journal.json"


class ResumeMode(StrEnum):
    """Execution engine whose atomic boundaries produced the checkpoint."""

    PENCIL = "pencil"
    SAFE_FILL = "safe-fill"


class RecoveryState(StrEnum):
    """Recovery required before the recorded checkpoint may be resumed."""

    READY = "ready"
    PALETTE_RECHECK_REQUIRED = "palette-recheck-required"
    MANUAL_INTERVENTION_REQUIRED = "manual-intervention-required"


def plan_fingerprint(plan: PaintPlan) -> str:
    """Return a deterministic SHA-256 fingerprint of all plan safety data.

    Hybrid fingerprints deliberately include fences, sentinels and repair
    strokes, not just the target pixels. A checkpoint can therefore never be
    reused after a planner or safety-geometry change that happens to produce
    the same final image.
    """

    if isinstance(plan, HybridPaintingPlan):
        kind = "hybrid"
    elif isinstance(plan, PaintingPlan):
        kind = "pencil"
    else:
        raise TypeError("Le journal de reprise exige un plan de peinture Dessinopia")
    payload = {
        "fingerprint_version": _FINGERPRINT_VERSION,
        "kind": kind,
        "plan": _canonical_value(plan),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def calibration_fingerprint(calibration: CalibrationProfile | None) -> str:
    """Fingerprint every coordinate and recognition ROI used by mouse/vision."""

    if calibration is None:
        return _EMPTY_CALIBRATION_FINGERPRINT
    if not isinstance(calibration, CalibrationProfile):
        raise TypeError("L'empreinte exige une calibration Dessinopia")
    payload = {
        "fingerprint_version": _FINGERPRINT_VERSION,
        "calibration": _canonical_value(calibration),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _compatible_calibration_fingerprints(
    calibration: CalibrationProfile | None,
) -> set[str]:
    if calibration is None:
        return {_EMPTY_CALIBRATION_FINGERPRINT}
    # Windows can supply integer origins that are read back from settings as floats.
    left = float(calibration.reference_left)
    top = float(calibration.reference_top)
    left_values = (left, int(left)) if left.is_integer() else (left,)
    top_values = (top, int(top)) if top.is_integer() else (top,)
    return {
        calibration_fingerprint(replace(calibration, reference_left=x, reference_top=y))
        for x in left_values
        for y in top_values
    }


def is_atomic_progress(plan: PaintPlan, validated_pixels: int) -> bool:
    """Whether progress lies exactly after a complete stroke/fill operation."""

    if isinstance(validated_pixels, bool) or not isinstance(validated_pixels, int):
        return False
    if not 0 <= validated_pixels <= plan.pixel_count:
        return False
    if validated_pixels in (0, plan.pixel_count):
        return True

    completed = 0
    if isinstance(plan, PaintingPlan):
        sizes = (
            stroke.length
            for family in plan.families
            for shade in family.shades
            for stroke in shade.strokes
        )
    elif isinstance(plan, HybridPaintingPlan):
        sizes = (_hybrid_operation_size(operation) for operation in plan.operations)
    else:
        raise TypeError("Le journal de reprise exige un plan de peinture Dessinopia")
    for size in sizes:
        completed += size
        if completed == validated_pixels:
            return True
        if completed > validated_pixels:
            return False
    return False


@dataclass(frozen=True, slots=True)
class ResumeJournalEntry:
    """Validated, immutable payload persisted by :class:`ResumeJournalStore`."""

    journal_version: int
    plan_fingerprint: str
    mode: ResumeMode
    validated_pixels: int
    total_pixels: int
    recovery: RecoveryState
    calibration_fingerprint: str = _EMPTY_CALIBRATION_FINGERPRINT

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def for_plan(
        cls,
        plan: PaintPlan,
        *,
        mode: ResumeMode,
        validated_pixels: int,
        recovery: RecoveryState = RecoveryState.READY,
        calibration: CalibrationProfile | None = None,
    ) -> ResumeJournalEntry:
        """Build an entry only if progress is an atomic boundary of ``plan``."""

        if not isinstance(mode, ResumeMode):
            raise TypeError("Le mode de reprise est invalide")
        if not isinstance(recovery, RecoveryState):
            raise TypeError("L'état de récupération est invalide")
        if not is_atomic_progress(plan, validated_pixels):
            raise ValueError(
                "Le progrès validé doit se situer à la fin d'une opération atomique"
            )
        return cls(
            journal_version=_JOURNAL_VERSION,
            plan_fingerprint=plan_fingerprint(plan),
            mode=mode,
            validated_pixels=validated_pixels,
            total_pixels=plan.pixel_count,
            recovery=recovery,
            calibration_fingerprint=calibration_fingerprint(calibration),
        )

    @classmethod
    def from_dict(cls, data: object) -> ResumeJournalEntry:
        if not isinstance(data, dict):
            raise TypeError("Le journal doit être un objet JSON")
        expected_keys = {
            "journal_version",
            "plan_fingerprint",
            "mode",
            "validated_pixels",
            "total_pixels",
            "recovery",
            "calibration_fingerprint",
        }
        if set(data) != expected_keys:
            raise ValueError("Le schéma du journal est incomplet ou inconnu")
        journal_version = _strict_int(data["journal_version"], "journal_version")
        validated_pixels = _strict_int(data["validated_pixels"], "validated_pixels")
        total_pixels = _strict_int(data["total_pixels"], "total_pixels")
        fingerprint = data["plan_fingerprint"]
        calibration_digest = data["calibration_fingerprint"]
        if not isinstance(fingerprint, str):
            raise TypeError("L'empreinte du plan doit être une chaîne")
        if not isinstance(calibration_digest, str):
            raise TypeError("L'empreinte de calibration doit être une chaîne")
        try:
            mode = ResumeMode(data["mode"])
            recovery = RecoveryState(data["recovery"])
        except ValueError as exc:
            raise ValueError("Le mode ou l'état de récupération est inconnu") from exc
        return cls(
            journal_version=journal_version,
            plan_fingerprint=fingerprint,
            mode=mode,
            validated_pixels=validated_pixels,
            total_pixels=total_pixels,
            recovery=recovery,
            calibration_fingerprint=calibration_digest,
        )

    def to_dict(self) -> dict[str, int | str]:
        return {
            "journal_version": self.journal_version,
            "plan_fingerprint": self.plan_fingerprint,
            "mode": self.mode.value,
            "validated_pixels": self.validated_pixels,
            "total_pixels": self.total_pixels,
            "recovery": self.recovery.value,
            "calibration_fingerprint": self.calibration_fingerprint,
        }

    def matches(
        self,
        plan: PaintPlan,
        *,
        mode: ResumeMode | None = None,
        calibration: CalibrationProfile | None = None,
    ) -> bool:
        """Check plan identity, engine mode, total and atomic progress."""

        return (
            (mode is None or self.mode is mode)
            and self.plan_fingerprint == plan_fingerprint(plan)
            and self.total_pixels == plan.pixel_count
            and is_atomic_progress(plan, self.validated_pixels)
            and (
                calibration is None
                or self.calibration_fingerprint
                in _compatible_calibration_fingerprints(calibration)
            )
        )

    def _validate(self) -> None:
        if self.journal_version != _JOURNAL_VERSION:
            raise ValueError("La version du journal n'est pas prise en charge")
        if not _FINGERPRINT_PATTERN.fullmatch(self.plan_fingerprint):
            raise ValueError("L'empreinte du plan est invalide")
        if not _FINGERPRINT_PATTERN.fullmatch(self.calibration_fingerprint):
            raise ValueError("L'empreinte de calibration est invalide")
        if not isinstance(self.mode, ResumeMode):
            raise TypeError("Le mode de reprise est invalide")
        if isinstance(self.validated_pixels, bool) or not isinstance(
            self.validated_pixels, int
        ):
            raise TypeError("Le progrès validé doit être un entier")
        if isinstance(self.total_pixels, bool) or not isinstance(self.total_pixels, int):
            raise TypeError("Le total de pixels doit être un entier")
        if not 0 <= self.validated_pixels <= self.total_pixels:
            raise ValueError("Le progrès validé est hors des limites du plan")
        if not isinstance(self.recovery, RecoveryState):
            raise TypeError("L'état de récupération est invalide")


class ResumeJournalStore:
    """Read and atomically replace one crash-safe resume journal."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or resume_journal_path()

    def save(self, entry: ResumeJournalEntry) -> None:
        if not isinstance(entry, ResumeJournalEntry):
            raise TypeError("Le journal exige une entrée de reprise validée")
        # Re-parse even a manually constructed entry before touching the disk.
        payload = ResumeJournalEntry.from_dict(entry.to_dict()).to_dict()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            # The temporary file is closed and lives in the destination folder:
            # os.replace consequently uses MoveFileEx semantics on Windows and
            # cannot expose a partially written JSON document.
            os.replace(temporary, self.path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def load(
        self,
        *,
        expected_plan: PaintPlan | None = None,
        expected_mode: ResumeMode | None = None,
        expected_calibration: CalibrationProfile | None = None,
    ) -> ResumeJournalEntry | None:
        """Return a valid matching entry, or ``None`` for any unusable journal."""

        try:
            raw = self.path.read_text(encoding="utf-8")
            entry = ResumeJournalEntry.from_dict(json.loads(raw))
            if expected_mode is not None and not isinstance(expected_mode, ResumeMode):
                raise TypeError("Le mode attendu est invalide")
            if expected_plan is not None and not entry.matches(
                expected_plan,
                mode=expected_mode,
                calibration=expected_calibration,
            ):
                return None
            if (
                expected_plan is None
                and expected_mode is not None
                and entry.mode is not expected_mode
            ):
                return None
            return entry
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            return None

    def clear(self) -> None:
        """Remove the journal after a completed or explicitly abandoned drawing."""

        self.path.unlink(missing_ok=True)


class PlanResumeJournal:
    """Efficient journal facade bound to one immutable plan and runner mode.

    The potentially large hybrid fingerprint and all atomic boundaries are
    computed once. Calling :meth:`save_progress` after every committed operation
    therefore only validates a set membership and performs the atomic disk write.
    """

    def __init__(
        self,
        plan: PaintPlan,
        *,
        mode: ResumeMode,
        calibration: CalibrationProfile | None = None,
        store: ResumeJournalStore | None = None,
    ) -> None:
        if not isinstance(mode, ResumeMode):
            raise TypeError("Le mode de reprise est invalide")
        self.store = store or ResumeJournalStore()
        self.mode = mode
        self.plan_fingerprint = plan_fingerprint(plan)
        self.calibration_fingerprint = calibration_fingerprint(calibration)
        self._calibration_fingerprints = _compatible_calibration_fingerprints(calibration)
        self.total_pixels = plan.pixel_count
        self._boundaries = frozenset(_atomic_progress_boundaries(plan))

    def load(self) -> ResumeJournalEntry | None:
        """Load the entry only when it belongs exactly to this bound plan."""

        entry = self.store.load(expected_mode=self.mode)
        if entry is None:
            return None
        if (
            entry.plan_fingerprint != self.plan_fingerprint
            or entry.calibration_fingerprint not in self._calibration_fingerprints
            or entry.total_pixels != self.total_pixels
            or entry.validated_pixels not in self._boundaries
        ):
            return None
        return entry

    def save_progress(
        self,
        validated_pixels: int,
        *,
        recovery: RecoveryState = RecoveryState.READY,
    ) -> ResumeJournalEntry:
        """Atomically persist one progress value already committed by the runner."""

        if isinstance(validated_pixels, bool) or not isinstance(validated_pixels, int):
            raise TypeError("Le progrès validé doit être un entier")
        if validated_pixels not in self._boundaries:
            raise ValueError(
                "Le progrès validé doit se situer à la fin d'une opération atomique"
            )
        if not isinstance(recovery, RecoveryState):
            raise TypeError("L'état de récupération est invalide")
        entry = ResumeJournalEntry(
            journal_version=_JOURNAL_VERSION,
            plan_fingerprint=self.plan_fingerprint,
            mode=self.mode,
            validated_pixels=validated_pixels,
            total_pixels=self.total_pixels,
            recovery=recovery,
            calibration_fingerprint=self.calibration_fingerprint,
        )
        self.store.save(entry)
        return entry

    def clear(self) -> None:
        self.store.clear()


def _hybrid_operation_size(operation: PencilOperation | FillOperation) -> int:
    if isinstance(operation, PencilOperation):
        return operation.pixel_count
    if isinstance(operation, FillOperation):
        return operation.filled_pixel_count
    raise TypeError("Le plan hybride contient une opération inconnue")


def _atomic_progress_boundaries(plan: PaintPlan) -> tuple[int, ...]:
    boundaries = [0]
    completed = 0
    if isinstance(plan, PaintingPlan):
        sizes = (
            stroke.length
            for family in plan.families
            for shade in family.shades
            for stroke in shade.strokes
        )
    elif isinstance(plan, HybridPaintingPlan):
        sizes = (_hybrid_operation_size(operation) for operation in plan.operations)
    else:
        raise TypeError("Le journal de reprise exige un plan de peinture Dessinopia")
    for size in sizes:
        completed += size
        boundaries.append(completed)
    if completed != plan.pixel_count:
        raise ValueError("Les opérations du plan ne correspondent pas à son total de pixels")
    return tuple(boundaries)


def _strict_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} doit être un entier")
    return value


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list | tuple):
        return [_canonical_value(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"Valeur de plan non sérialisable : {type(value).__name__}")
