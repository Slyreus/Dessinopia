from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from dessinopia.core.color_spaces import delta_e_ciede2000, srgb_to_lab
from dessinopia.core.models import CanvasSize, DitherMode, FitMode
from dessinopia.core.palette import PALETTE, PALETTE_RGB


@dataclass(frozen=True, slots=True)
class ConversionOptions:
    size: CanvasSize
    fit_mode: FitMode = FitMode.CONTAIN
    dither_mode: DitherMode = DitherMode.NONE
    alpha_threshold: int = 96
    ordered_strength: float = 28.0


@dataclass(slots=True)
class ConversionResult:
    indices: np.ndarray
    preview_rgba: np.ndarray
    source_preview_rgba: np.ndarray
    options: ConversionOptions

    @property
    def width(self) -> int:
        return int(self.indices.shape[1])

    @property
    def height(self) -> int:
        return int(self.indices.shape[0])

    @property
    def painted_pixels(self) -> int:
        return int(np.count_nonzero(self.indices >= 0))

    @property
    def transparent_pixels(self) -> int:
        return int(np.count_nonzero(self.indices < 0))

    @property
    def unique_colors(self) -> int:
        opaque = self.indices[self.indices >= 0]
        return int(np.unique(opaque).size) if opaque.size else 0

    def color_counts(self) -> dict[int, int]:
        opaque = self.indices[self.indices >= 0]
        if not opaque.size:
            return {}
        values, counts = np.unique(opaque, return_counts=True)
        return {int(value): int(count) for value, count in zip(values, counts, strict=True)}


class ImageConverter:
    def __init__(self) -> None:
        self._palette_lab = srgb_to_lab(PALETTE_RGB)

    def convert_path(self, path: str | Path, options: ConversionOptions) -> ConversionResult:
        with Image.open(path) as loaded:
            image = ImageOps.exif_transpose(loaded).convert("RGBA")
        return self.convert_image(image, options)

    def convert_image(self, image: Image.Image, options: ConversionOptions) -> ConversionResult:
        prepared = self._prepare(image.convert("RGBA"), options)
        rgba = np.asarray(prepared, dtype=np.uint8)
        rgb = rgba[..., :3]
        alpha = rgba[..., 3]
        # Un pixel sans aucune opacité reste toujours une absence de peinture,
        # y compris lorsque l'utilisateur place le seuil sur zéro.
        opaque = (alpha > 0) & (alpha >= options.alpha_threshold)

        if options.dither_mode is DitherMode.FLOYD_STEINBERG:
            indices = self._floyd_steinberg(rgb, opaque)
        else:
            candidates = rgb.astype(np.float64)
            if options.dither_mode is DitherMode.ORDERED:
                candidates = self._ordered(candidates, options.ordered_strength)
            indices = self._nearest_indices(candidates.astype(np.uint8), opaque)

        preview = np.zeros((*indices.shape, 4), dtype=np.uint8)
        preview[..., 3] = 0
        painted = indices >= 0
        preview[painted, :3] = PALETTE_RGB[indices[painted]]
        preview[painted, 3] = 255
        return ConversionResult(
            indices=indices,
            preview_rgba=preview,
            source_preview_rgba=rgba.copy(),
            options=options,
        )

    @staticmethod
    def _prepare(image: Image.Image, options: ConversionOptions) -> Image.Image:
        target = (options.size.width, options.size.height)
        if options.fit_mode is FitMode.STRETCH:
            return image.resize(target, Image.Resampling.LANCZOS)
        if options.fit_mode is FitMode.COVER:
            return ImageOps.fit(
                image, target, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
            )

        image_ratio = image.width / image.height
        target_ratio = target[0] / target[1]
        contained_size = target
        if image_ratio > target_ratio:
            contained_size = (
                target[0],
                max(1, round(image.height / image.width * target[0])),
            )
        elif image_ratio < target_ratio:
            contained_size = (
                max(1, round(image.width / image.height * target[1])),
                target[1],
            )
        contained = image.resize(contained_size, Image.Resampling.LANCZOS)
        output = Image.new("RGBA", target, (0, 0, 0, 0))
        position = ((target[0] - contained.width) // 2, (target[1] - contained.height) // 2)
        output.alpha_composite(contained, position)
        return output

    @staticmethod
    def _ordered(rgb: np.ndarray, strength: float) -> np.ndarray:
        bayer = np.asarray(
            (
                (0, 8, 2, 10),
                (12, 4, 14, 6),
                (3, 11, 1, 9),
                (15, 7, 13, 5),
            ),
            dtype=np.float64,
        )
        height, width = rgb.shape[:2]
        tiled = np.tile(bayer, ((height + 3) // 4, (width + 3) // 4))[:height, :width]
        offset = ((tiled + 0.5) / 16.0 - 0.5) * strength
        return np.clip(rgb + offset[..., None], 0, 255)

    def _nearest_indices(self, rgb: np.ndarray, opaque: np.ndarray) -> np.ndarray:
        result = np.full(opaque.shape, -1, dtype=np.int16)
        if not np.any(opaque):
            return result
        pixels_lab = srgb_to_lab(rgb[opaque])
        # Les lots gardent la consommation mémoire stable pour les grilles 150×150.
        nearest = np.empty(pixels_lab.shape[0], dtype=np.int16)
        for start in range(0, pixels_lab.shape[0], 4096):
            chunk = pixels_lab[start : start + 4096]
            distances = delta_e_ciede2000(chunk[:, None, :], self._palette_lab[None, :, :])
            nearest[start : start + len(chunk)] = np.argmin(distances, axis=1).astype(np.int16)
        result[opaque] = nearest
        return result

    def _floyd_steinberg(self, rgb: np.ndarray, opaque: np.ndarray) -> np.ndarray:
        work = rgb.astype(np.float64)
        height, width = opaque.shape
        result = np.full((height, width), -1, dtype=np.int16)

        for y in range(height):
            left_to_right = y % 2 == 0
            xs = range(width) if left_to_right else range(width - 1, -1, -1)
            direction = 1 if left_to_right else -1
            for x in xs:
                if not opaque[y, x]:
                    continue
                current = np.clip(work[y, x], 0, 255)
                lab = srgb_to_lab(current.reshape(1, 3))
                distances = delta_e_ciede2000(lab, self._palette_lab)
                palette_index = int(np.argmin(distances))
                result[y, x] = palette_index
                error = current - PALETTE_RGB[palette_index].astype(np.float64)

                self._diffuse(work, opaque, x + direction, y, error, 7 / 16)
                self._diffuse(work, opaque, x - direction, y + 1, error, 3 / 16)
                self._diffuse(work, opaque, x, y + 1, error, 5 / 16)
                self._diffuse(work, opaque, x + direction, y + 1, error, 1 / 16)
        return result

    @staticmethod
    def _diffuse(
        work: np.ndarray,
        opaque: np.ndarray,
        x: int,
        y: int,
        error: np.ndarray,
        weight: float,
    ) -> None:
        if 0 <= y < work.shape[0] and 0 <= x < work.shape[1] and opaque[y, x]:
            work[y, x] += error * weight


def palette_description(result: ConversionResult) -> list[tuple[str, str, int]]:
    counts = result.color_counts()
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [(PALETTE[index].name, PALETTE[index].hex_value, count) for index, count in ordered]
