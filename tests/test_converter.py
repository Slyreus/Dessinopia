from dataclasses import replace
from unittest import TestCase

import numpy as np
from PIL import Image

from dessinopia.core.converter import ConversionOptions, ImageConverter
from dessinopia.core.models import CanvasSize, DitherMode, FitMode
from dessinopia.core.palette import PALETTE_RGB


class ConversionTests(TestCase):
    def test_crisp_edges_and_isolated_shades(self) -> None:
        pixels = np.full((7, 7), 12, dtype=np.int16)
        pixels[2, 2] = 22  # Nuance proche du fond.
        pixels[4, 2] = 0  # Point sombre volontaire.
        pixels[1:6, 5] = 0
        image = Image.fromarray(PALETTE_RGB[pixels])
        converter = ImageConverter()
        enlarged = converter.convert_image(image, ConversionOptions(CanvasSize(21, 21)))
        np.testing.assert_array_equal(enlarged.indices, pixels.repeat(3, 0).repeat(3, 1))
        native = replace(enlarged.options, fit_mode=FitMode.NATIVE, reduce_noise=False)
        centered = converter.convert_image(image, native)
        np.testing.assert_array_equal(centered.indices, np.pad(pixels, 7, constant_values=-1))

        cleaned = converter.convert_image(image, ConversionOptions(CanvasSize(7, 7)))
        expected = pixels.copy()
        expected[2, 2] = 12
        np.testing.assert_array_equal(cleaned.indices, expected)

    def test_dither_respects_color_limit_and_transparency(self) -> None:
        rgba = np.full((8, 64, 4), 255, dtype=np.uint8)
        rgba[..., :3] = np.linspace(0, 255, 64, dtype=np.uint8)[None, :, None]
        rgba[2:5, 30:33, 3] = 0
        image = Image.fromarray(rgba)
        converter = ImageConverter()
        options = ConversionOptions(CanvasSize(64, 8), max_colors=3, reduce_noise=False)
        plain = converter.convert_image(image, options)
        for mode in (DitherMode.ORDERED, DitherMode.FLOYD_STEINBERG):
            with self.subTest(mode=mode):
                off = replace(options, dither_mode=mode, dither_strength=0)
                np.testing.assert_array_equal(
                    converter.convert_image(image, off).indices, plain.indices
                )
                dithered = converter.convert_image(image, replace(off, dither_strength=100))
                self.assertLessEqual(dithered.unique_colors, options.max_colors)
                self.assertTrue(np.any(dithered.indices != plain.indices))
                np.testing.assert_array_equal(dithered.indices < 0, rgba[..., 3] == 0)
