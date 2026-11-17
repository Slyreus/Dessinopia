from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dessinopia.core.models import (
    DEFAULT_DITHER_STRENGTH,
    DEFAULT_MAX_COLORS,
    AspectRatio,
    CalibrationProfile,
    DitherMode,
    FitMode,
    Precision,
    ResizeMode,
)
from dessinopia.core.palette import PALETTE
from dessinopia.core.timing import TIMING_PRESETS, TimingProfile


def settings_path() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    return base / "Dessinopia" / "settings.json"


def _bounded_int(value: Any, default: int, lower: int, upper: int) -> int:
    if type(value) is not int:
        return default
    return min(upper, max(lower, value))


@dataclass(slots=True)
class AppSettings:
    ratio: AspectRatio = AspectRatio.LANDSCAPE_16_9
    precision: Precision = Precision.SMALL
    fit_mode: FitMode = FitMode.CONTAIN
    dither_mode: DitherMode = DitherMode.NONE
    alpha_threshold: int = 96
    show_grid: bool = True
    allow_fill: bool = False
    tempo: str = "equilibre"
    calibration: CalibrationProfile | None = None
    timing: TimingProfile = TimingProfile()
    resize_mode: ResizeMode = ResizeMode.PIXEL
    max_colors: int = DEFAULT_MAX_COLORS
    dither_strength: int = DEFAULT_DITHER_STRENGTH
    reduce_noise: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "ratio": self.ratio.value,
            "precision": self.precision.value,
            "fit_mode": self.fit_mode.value,
            "resize_mode": self.resize_mode.value,
            "max_colors": self.max_colors,
            "dither_mode": self.dither_mode.value,
            "dither_strength": self.dither_strength,
            "reduce_noise": self.reduce_noise,
            "alpha_threshold": self.alpha_threshold,
            "show_grid": self.show_grid,
            "allow_fill": self.allow_fill,
            "tempo": self.tempo,
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "timing": asdict(self.timing),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppSettings:
        if not isinstance(data, dict):
            raise TypeError("Les réglages doivent être un objet JSON")
        calibration = data.get("calibration")
        tempo = str(data.get("tempo", "equilibre"))
        timing_data = data.get("timing")
        resize_mode = data.get("resize_mode", ResizeMode.PIXEL.value)
        return cls(
            ratio=AspectRatio(data.get("ratio", AspectRatio.LANDSCAPE_16_9.value)),
            precision=Precision(data.get("precision", Precision.SMALL.value)),
            fit_mode=FitMode(data.get("fit_mode", FitMode.CONTAIN.value)),
            resize_mode=ResizeMode(resize_mode) if resize_mode in ResizeMode else ResizeMode.PIXEL,
            max_colors=_bounded_int(data.get("max_colors"), DEFAULT_MAX_COLORS, 1, len(PALETTE)),
            dither_mode=DitherMode(data.get("dither_mode", DitherMode.NONE.value)),
            dither_strength=_bounded_int(
                data.get("dither_strength"), DEFAULT_DITHER_STRENGTH, 0, 100
            ),
            reduce_noise=bool(data.get("reduce_noise", True)),
            alpha_threshold=int(data.get("alpha_threshold", 96)),
            show_grid=bool(data.get("show_grid", True)),
            allow_fill=bool(data.get("allow_fill", False)),
            tempo=tempo if tempo in TIMING_PRESETS else "equilibre",
            calibration=CalibrationProfile.from_dict(calibration) if calibration else None,
            timing=TimingProfile(**timing_data)
            if timing_data
            else TIMING_PRESETS.get(tempo, TimingProfile()),
        )


class SettingsStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings_path()

    def load(self) -> AppSettings:
        if not self.path.exists():
            return AppSettings()
        try:
            return AppSettings.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return AppSettings()

    def save(self, settings: AppSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(settings.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(self.path)
