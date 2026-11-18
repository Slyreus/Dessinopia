from unittest import TestCase

import numpy as np

from dessinopia.core.models import ScreenRect
from dessinopia.vision.detector import HeartopiaDetector


class CalibrationTests(TestCase):
    def test_painted_canvas_excludes_frame_and_margins(self) -> None:
        image = np.full((270, 480, 3), (80, 64, 53), dtype=np.uint8)
        matte = (246, 239, 213)
        image[50:251, 65:416] = matte
        image[62:242, 85:395] = (184, 166, 148)
        image[64:240, 87:393] = matte
        image[68:236, 89:391] = (50, 70, 90)

        detector = HeartopiaDetector()
        previous = ScreenRect(88, 68, 304, 180)
        profile = detector.auto_profile(image, expected_canvas=previous)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.canvas, ScreenRect(89, 68, 302, 168))

        # Le jeu peut être centré dans un écran ultralarge ou une fenêtre verticale.
        for horizontal, vertical in ((480, 0), (0, 360)):
            padded = np.pad(image, ((vertical, vertical), (horizontal, horizontal), (0, 0)))
            measured = detector.auto_profile(padded)
            self.assertEqual(measured.canvas, profile.canvas.translated(horizontal, vertical))

        image[60:120, 70:120] = (245, 245, 245)
        self.assertIsNone(detector.auto_profile(image, expected_canvas=previous))
