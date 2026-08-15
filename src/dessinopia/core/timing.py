from __future__ import annotations

from dataclasses import dataclass
from random import Random


@dataclass(frozen=True, slots=True)
class TimingProfile:
    cursor_move_seconds: float = 0.09
    click_settle_seconds: float = 0.42
    palette_settle_seconds: float = 0.72
    stroke_cell_seconds: float = 0.028
    between_strokes_seconds: float = 0.050
    jitter_ratio: float = 0.18

    def varied(self, value: float, random: Random) -> float:
        spread = max(0.0, min(self.jitter_ratio, 0.5))
        return max(0.0, value * random.uniform(1 - spread, 1 + spread))

    def estimate_seconds(self, pixels: int, strokes: int, shades: int, families: int) -> float:
        internal_cell_moves = max(0, pixels - strokes)
        drawing = internal_cell_moves * self.stroke_cell_seconds + strokes * (
            self.cursor_move_seconds + self.between_strokes_seconds
        )
        shade_navigation = shades * (self.cursor_move_seconds + self.click_settle_seconds)
        family_navigation = families * (
            3 * self.cursor_move_seconds
            + self.click_settle_seconds
            + 2 * self.palette_settle_seconds
        )
        brush_selection = self.cursor_move_seconds + self.click_settle_seconds if families else 0.0
        return drawing + shade_navigation + family_navigation + brush_selection


TIMING_PRESETS: dict[str, TimingProfile] = {
    # Mode volontairement lent pour les machines modestes ou lorsque Heartopia
    # a besoin de davantage de temps pour assimiler les gestes et animations.
    "tres_lent": TimingProfile(0.40, 1.40, 2.40, 0.120, 0.300, 0.12),
    "prudent": TimingProfile(0.16, 0.72, 1.00, 0.038, 0.080, 0.22),
    "equilibre": TimingProfile(),
    "rapide": TimingProfile(0.045, 0.20, 0.42, 0.020, 0.030, 0.10),
}
