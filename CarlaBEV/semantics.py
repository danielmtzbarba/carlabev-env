from __future__ import annotations

from enum import IntEnum

import numpy as np


class SemanticClass(IntEnum):
    NON_DRIVABLE = 0
    DRIVABLE = 1
    SIDEWALK = 2
    VEHICLE = 3
    PEDESTRIAN = 4
    ROUTE = 5
    TRAFFIC_LIGHT_RED = 6
    EGO = 7
    TRAFFIC_LIGHT_YELLOW = 8
    TRAFFIC_LIGHT_GREEN = 9


VOID_ID = 255
"""Pixel value of 'nothing rendered' (outside the rotated crop / masked corners). Renders black."""

SEMANTIC_COLOR_TUPLES = {
    SemanticClass.NON_DRIVABLE: (150, 150, 150),
    SemanticClass.DRIVABLE: (255, 255, 255),
    SemanticClass.SIDEWALK: (220, 220, 220),
    SemanticClass.VEHICLE: (0, 7, 175),
    SemanticClass.PEDESTRIAN: (255, 0, 0),
    SemanticClass.ROUTE: (0, 255, 0),
    SemanticClass.TRAFFIC_LIGHT_RED: (255, 64, 64),
    SemanticClass.EGO: (0, 200, 255),  # cyan: distinct from vehicles (blue), pedestrians (red), route (green), lights
    # Distinct colours: a yellow light must not read as the ego, a green light not as route.
    SemanticClass.TRAFFIC_LIGHT_YELLOW: (255, 200, 0),
    SemanticClass.TRAFFIC_LIGHT_GREEN: (0, 200, 0),
}

SEMANTIC_COLOR_ARRAYS = {
    cls: np.array(color, dtype=np.uint8) for cls, color in SEMANTIC_COLOR_TUPLES.items()
}

MAP_LABEL_TO_CLASS = {
    0: SemanticClass.NON_DRIVABLE,
    127: SemanticClass.DRIVABLE,
    255: SemanticClass.SIDEWALK,
}

DRIVABLE_CLASSES = {
    SemanticClass.DRIVABLE,
    SemanticClass.ROUTE,
}
OFFROAD_CLASSES = {
    SemanticClass.SIDEWALK,
}
BLOCKING_CLASSES = {
    SemanticClass.NON_DRIVABLE,
}

_CLASS_BY_COLOR = {
    color: semantic_class for semantic_class, color in SEMANTIC_COLOR_TUPLES.items()
}


def semantic_color_tuple(semantic_class: SemanticClass) -> tuple[int, int, int]:
    return SEMANTIC_COLOR_TUPLES[semantic_class]


def semantic_color_array(semantic_class: SemanticClass) -> np.ndarray:
    return SEMANTIC_COLOR_ARRAYS[semantic_class]


def build_palette_lut() -> np.ndarray:
    """256x3 uint8 lookup table: class id -> RGB (VOID and unused ids are black)."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for semantic_class, color in SEMANTIC_COLOR_TUPLES.items():
        lut[int(semantic_class)] = color
    return lut


PALETTE_LUT = build_palette_lut()


def ids_to_rgb(ids: np.ndarray) -> np.ndarray:
    """Map an (H, W) class-id raster to an (H, W, 3) RGB image through the palette."""
    return PALETTE_LUT[np.asarray(ids, dtype=np.uint8)]


def semantic_class_from_id(class_id) -> SemanticClass | None:
    try:
        return SemanticClass(int(class_id))
    except ValueError:
        return None


def semantic_class_from_rgb(tile_rgb) -> SemanticClass | None:
    if tile_rgb is None:
        return None
    color = tuple(int(channel) for channel in np.asarray(tile_rgb, dtype=np.uint8).tolist())
    return _CLASS_BY_COLOR.get(color)
