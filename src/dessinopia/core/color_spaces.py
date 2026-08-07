from __future__ import annotations

import numpy as np


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64) / 255.0
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    encoded = np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1 / 2.4) - 0.055,
    )
    return np.clip(np.rint(encoded * 255), 0, 255).astype(np.uint8)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convertit du sRGB 8 bits vers CIE L*a*b* (illuminant D65)."""

    linear = srgb_to_linear(rgb)
    matrix = np.asarray(
        (
            (0.4124564, 0.3575761, 0.1804375),
            (0.2126729, 0.7151522, 0.0721750),
            (0.0193339, 0.1191920, 0.9503041),
        )
    )
    xyz = linear @ matrix.T
    xyz /= np.asarray((0.95047, 1.0, 1.08883))
    delta = 6 / 29
    fxyz = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4 / 29)
    fx, fy, fz = np.moveaxis(fxyz, -1, 0)
    return np.stack((116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)), axis=-1)


def delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Distance CIEDE2000 vectorisée, avec diffusion NumPy standard."""

    a = np.asarray(lab1, dtype=np.float64)
    b = np.asarray(lab2, dtype=np.float64)
    l1, aa1, bb1 = np.moveaxis(a, -1, 0)
    l2, aa2, bb2 = np.moveaxis(b, -1, 0)

    c1 = np.hypot(aa1, bb1)
    c2 = np.hypot(aa2, bb2)
    c_bar = (c1 + c2) / 2
    g = 0.5 * (1 - np.sqrt(c_bar**7 / (c_bar**7 + 25**7)))
    ap1 = (1 + g) * aa1
    ap2 = (1 + g) * aa2
    cp1 = np.hypot(ap1, bb1)
    cp2 = np.hypot(ap2, bb2)
    hp1 = np.mod(np.degrees(np.arctan2(bb1, ap1)), 360)
    hp2 = np.mod(np.degrees(np.arctan2(bb2, ap2)), 360)

    dl = l2 - l1
    dc = cp2 - cp1
    dh = hp2 - hp1
    dh = np.where((cp1 * cp2) == 0, 0, dh)
    dh = np.where(dh > 180, dh - 360, dh)
    dh = np.where(dh < -180, dh + 360, dh)
    d_h = 2 * np.sqrt(cp1 * cp2) * np.sin(np.radians(dh / 2))

    l_bar = (l1 + l2) / 2
    c_bar_p = (cp1 + cp2) / 2
    h_sum = hp1 + hp2
    h_diff = np.abs(hp1 - hp2)
    h_bar = np.where((cp1 * cp2) == 0, h_sum, h_sum / 2)
    h_bar = np.where(((cp1 * cp2) != 0) & (h_diff > 180) & (h_sum < 360), (h_sum + 360) / 2, h_bar)
    h_bar = np.where(((cp1 * cp2) != 0) & (h_diff > 180) & (h_sum >= 360), (h_sum - 360) / 2, h_bar)

    t = (
        1
        - 0.17 * np.cos(np.radians(h_bar - 30))
        + 0.24 * np.cos(np.radians(2 * h_bar))
        + 0.32 * np.cos(np.radians(3 * h_bar + 6))
        - 0.20 * np.cos(np.radians(4 * h_bar - 63))
    )
    sl = 1 + (0.015 * (l_bar - 50) ** 2) / np.sqrt(20 + (l_bar - 50) ** 2)
    sc = 1 + 0.045 * c_bar_p
    sh = 1 + 0.015 * c_bar_p * t
    delta_theta = 30 * np.exp(-(((h_bar - 275) / 25) ** 2))
    rc = 2 * np.sqrt(c_bar_p**7 / (c_bar_p**7 + 25**7))
    rt = -rc * np.sin(np.radians(2 * delta_theta))

    return np.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + (d_h / sh) ** 2 + rt * (dc / sc) * (d_h / sh))
