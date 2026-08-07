from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PaletteColor:
    index: int
    group: int
    shade: int
    name: str
    hex_value: str
    main_column: int
    main_row: int
    shade_column: int
    shade_row: int
    transparent: bool = False

    @property
    def rgb(self) -> tuple[int, int, int]:
        value = self.hex_value.lstrip("#")
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


GROUP_NAMES = (
    "Neutres",
    "Rouges",
    "Corails",
    "Oranges",
    "Jaunes",
    "Verts anis",
    "Verts",
    "Turquoises",
    "Cyans",
    "Bleus",
    "Indigos",
    "Violets",
    "Roses",
)

# Position représentative de chaque famille dans la palette principale (2 × 8).
MAIN_CELLS = (
    (0, 0),
    (0, 2),
    (1, 2),
    (0, 3),
    (1, 3),
    (0, 4),
    (1, 4),
    (0, 5),
    (1, 5),
    (0, 6),
    (1, 6),
    (0, 7),
    (1, 7),
)

# Valeurs fixes de la palette Heartopia. Les variantes visuellement proches
# restent distinctes afin de reproduire fidèlement les choix disponibles en jeu.
_GROUP_HEX: tuple[tuple[str, ...], ...] = (
    ("#051616", "#414545", "#808282", "#bebfbf", "#feffff"),
    (
        "#cf354d",
        "#ee6f72",
        "#a6263d",
        "#f5aca6",
        "#c98483",
        "#a35d5e",
        "#69313b",
        "#e7d5d5",
        "#c0acab",
        "#755e5e",
    ),
    (
        "#e95e2b",
        "#f98358",
        "#ab4226",
        "#feba9f",
        "#d9937c",
        "#af6c58",
        "#753b31",
        "#e9d5d0",
        "#c1aca6",
        "#755e59",
    ),
    (
        "#f49e16",
        "#feae3b",
        "#b16f16",
        "#fece92",
        "#daa76d",
        "#b3814b",
        "#795126",
        "#f5e4ce",
        "#cdbca9",
        "#806f5e",
    ),
    (
        "#edca16",
        "#f9d838",
        "#b39416",
        "#fae791",
        "#d3be6f",
        "#ab954b",
        "#756326",
        "#eee7c7",
        "#c6bfa2",
        "#787259",
    ),
    (
        "#a8bc16",
        "#b6c931",
        "#758616",
        "#d8df93",
        "#adb76d",
        "#85914b",
        "#535e2b",
        "#e6e9c7",
        "#bcc2a3",
        "#6e745d",
    ),
    (
        "#05a25d",
        "#41b97b",
        "#057447",
        "#9cdaad",
        "#76b28b",
        "#4f8969",
        "#245640",
        "#c3e0cc",
        "#9db7a6",
        "#53695d",
    ),
    (
        "#058781",
        "#05aba0",
        "#056966",
        "#7ecdc2",
        "#55a49c",
        "#2b7e78",
        "#054b4b",
        "#bee0da",
        "#98b7b2",
        "#4e6b66",
    ),
    (
        "#05729c",
        "#0599ba",
        "#055878",
        "#79bbca",
        "#5193a5",
        "#246d7f",
        "#05495b",
        "#c6dde2",
        "#9eb5ba",
        "#4f676f",
    ),
    (
        "#055ea6",
        "#2b83c1",
        "#054782",
        "#83a8c9",
        "#5d80a1",
        "#365b7f",
        "#193b56",
        "#c1cdd5",
        "#9ba6b0",
        "#4c5967",
    ),
    (
        "#534da1",
        "#7577bd",
        "#3e387e",
        "#a2a0c7",
        "#787aa1",
        "#55567e",
        "#333555",
        "#c9cad5",
        "#a2a3b0",
        "#565869",
    ),
    (
        "#813d8b",
        "#a167a9",
        "#602b6c",
        "#b89bb9",
        "#907395",
        "#6c4d73",
        "#432e4b",
        "#cfc9d1",
        "#aba1ac",
        "#605665",
    ),
    (
        "#ad356f",
        "#cf6b8f",
        "#862658",
        "#d9a1b4",
        "#b47a8c",
        "#8b5367",
        "#60354b",
        "#e4d5da",
        "#bcadb1",
        "#725e66",
    ),
)


def _build_palette() -> tuple[PaletteColor, ...]:
    colors: list[PaletteColor] = []
    for group_index, values in enumerate(_GROUP_HEX):
        main_column, main_row = MAIN_CELLS[group_index]
        for shade_index, hex_value in enumerate(values):
            colors.append(
                PaletteColor(
                    index=len(colors),
                    group=group_index,
                    shade=shade_index,
                    name=f"{GROUP_NAMES[group_index]} {shade_index + 1}",
                    hex_value=hex_value,
                    main_column=main_column,
                    main_row=main_row,
                    shade_column=shade_index % 2,
                    shade_row=shade_index // 2,
                )
            )
    return tuple(colors)


PALETTE: tuple[PaletteColor, ...] = _build_palette()
PALETTE_RGB = np.asarray([color.rgb for color in PALETTE], dtype=np.uint8)
PALETTE_BY_INDEX = {color.index: color for color in PALETTE}

# La transparence est la sixième case de la famille neutre dans le jeu. Elle n'est
# pas une couleur de quantification ; l'index -1 est réservé aux pixels à ignorer.
TRANSPARENT_SHADE_CELL = (1, 2)


def colors_for_group(group: int) -> tuple[PaletteColor, ...]:
    return tuple(color for color in PALETTE if color.group == group)
