from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from dessinopia.core.color_spaces import delta_e_ciede2000, srgb_to_lab
from dessinopia.core.models import (
    DEFAULT_DITHER_STRENGTH,
    DEFAULT_MAX_COLORS,
    CanvasSize,
    DitherMode,
    FitMode,
    ResizeMode,
)
from dessinopia.core.palette import PALETTE, PALETTE_RGB


@dataclass(frozen=True, slots=True)
class ConversionOptions:
    size: CanvasSize
    fit_mode: FitMode = FitMode.CONTAIN
    dither_mode: DitherMode = DitherMode.NONE
    alpha_threshold: int = 96
    ordered_strength: float = 28.0
    resize_mode: ResizeMode = ResizeMode.PIXEL
    max_colors: int = DEFAULT_MAX_COLORS
    dither_strength: int = DEFAULT_DITHER_STRENGTH
    reduce_noise: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.max_colors <= len(PALETTE):
            raise ValueError(f"Le nombre de couleurs doit être compris entre 1 et {len(PALETTE)}.")
        if not 0 <= self.dither_strength <= 100:
            raise ValueError("L’intensité du tramage doit être comprise entre 0 et 100 %.")


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
        self._all_colors = np.arange(len(PALETTE), dtype=np.int16)
        self._palette_distances = delta_e_ciede2000(
            self._palette_lab[:, None, :], self._palette_lab[None, :, :]
        )

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

        indices = self._convert_colors(rgb, opaque, options)
        if options.reduce_noise:
            self._clean_isolated(indices)

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
        if options.fit_mode is FitMode.NATIVE:
            left = (image.width - target[0]) // 2
            top = (image.height - target[1]) // 2
            return image.crop((left, top, left + target[0], top + target[1]))
        resample = (
            Image.Resampling.NEAREST
            if options.resize_mode is ResizeMode.PIXEL
            else Image.Resampling.LANCZOS
        )
        if options.fit_mode is FitMode.STRETCH:
            return image.resize(target, resample)
        if options.fit_mode is FitMode.COVER:
            return ImageOps.fit(image, target, method=resample, centering=(0.5, 0.5))

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
        contained = image.resize(contained_size, resample)
        output = Image.new("RGBA", target, (0, 0, 0, 0))
        position = ((target[0] - contained.width) // 2, (target[1] - contained.height) // 2)
        output.alpha_composite(contained, position)
        return output

    def _convert_colors(
        self, rgb: np.ndarray, opaque: np.ndarray, options: ConversionOptions
    ) -> np.ndarray:
        if not opaque.any():
            return np.full(opaque.shape, -1, dtype=np.int16)

        mode = options.dither_mode if options.dither_strength else DitherMode.NONE
        colors = self._all_colors
        if options.max_colors < len(PALETTE):
            indices = self._nearest_indices(rgb, opaque, colors)
            used, counts = np.unique(indices[opaque], return_counts=True)
            if mode is DitherMode.NONE and used.size <= options.max_colors:
                return indices
            colors = self._select_colors(used, counts, options.max_colors)
            if colors.size == 1:
                indices[opaque] = colors[0]
                return indices

        strength = options.dither_strength / 100
        if mode is DitherMode.FLOYD_STEINBERG:
            return self._floyd_steinberg(rgb, opaque, colors, strength)
        if mode is DitherMode.ORDERED:
            rgb = self._ordered(rgb, options.ordered_strength * strength)
        return self._nearest_indices(rgb, opaque, colors)

    def _select_colors(self, used: np.ndarray, counts: np.ndarray, limit: int) -> np.ndarray:
        if used.size <= limit:
            # Le tramage peut employer les places libres pour les teintes voisines.
            distance_to_used = self._palette_distances[used].min(axis=0)
            return np.sort(np.argsort(distance_to_used, kind="stable")[:limit])
        # Chaque choix réduit l'erreur perceptuelle cumulée. Le carré de la
        # distance donne aussi du poids aux petits détails très contrastés.
        errors = self._palette_distances[np.ix_(used, used)] ** 2 * counts[:, None]
        best = np.full(used.size, np.inf)
        selected = np.zeros(used.size, dtype=bool)
        for _ in range(limit):
            costs = np.minimum(best[:, None], errors).sum(axis=0)
            costs[selected] = np.inf
            choice = int(np.argmin(costs))
            selected[choice] = True
            best = np.minimum(best, errors[:, choice])
        return used[selected]

    def _clean_isolated(self, indices: np.ndarray) -> None:
        height, width = indices.shape
        padded = np.pad(indices, 1, constant_values=-1)
        neighbors = np.stack(
            [
                padded[y : y + height, x : x + width]
                for y in range(3)
                for x in range(3)
                if (y, x) != (1, 1)
            ]
        )
        # Une majorité de cinq voisins sur huit contient forcément la médiane.
        majority = np.partition(neighbors, 4, axis=0)[4]
        isolated = (
            (indices >= 0)
            & (majority >= 0)
            & ~np.any(neighbors == indices, axis=0)
            & (np.count_nonzero(neighbors == majority, axis=0) >= 5)
        )
        # Conserver les accents contrastés, même lorsqu'ils ne font qu'un pixel.
        original = indices[isolated]
        replacement = majority[isolated]
        close = self._palette_distances[original, replacement] <= 15
        indices[isolated] = np.where(close, replacement, original)

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
        return np.clip(rgb + offset[..., None], 0, 255).astype(np.uint8)

    def _nearest_indices(
        self, rgb: np.ndarray, opaque: np.ndarray, colors: np.ndarray
    ) -> np.ndarray:
        result = np.full(opaque.shape, -1, dtype=np.int16)
        unique_rgb, inverse = np.unique(rgb[opaque], axis=0, return_inverse=True)
        pixels_lab = srgb_to_lab(unique_rgb)
        palette_lab = self._palette_lab[colors]
        # Les lots gardent la consommation mémoire stable pour les grilles 150×150.
        nearest = np.empty(pixels_lab.shape[0], dtype=np.int16)
        for start in range(0, pixels_lab.shape[0], 4096):
            chunk = pixels_lab[start : start + 4096]
            distances = delta_e_ciede2000(chunk[:, None, :], palette_lab[None, :, :])
            nearest[start : start + len(chunk)] = colors[np.argmin(distances, axis=1)]
        result[opaque] = nearest[inverse]
        return result

    def _floyd_steinberg(
        self, rgb: np.ndarray, opaque: np.ndarray, colors: np.ndarray, strength: float
    ) -> np.ndarray:
        work = rgb.astype(np.float64)
        palette_lab = self._palette_lab[colors]
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
                distances = delta_e_ciede2000(lab, palette_lab)
                palette_index = int(colors[np.argmin(distances)])
                result[y, x] = palette_index
                error = (current - PALETTE_RGB[palette_index]) * strength

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
