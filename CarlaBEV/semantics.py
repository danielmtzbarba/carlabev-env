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


SEMANTIC_COLOR_TUPLES = {
    SemanticClass.NON_DRIVABLE: (150, 150, 150),
    SemanticClass.DRIVABLE: (255, 255, 255),
    SemanticClass.SIDEWALK: (220, 220, 220),
    SemanticClass.VEHICLE: (0, 7, 175),
    SemanticClass.PEDESTRIAN: (255, 0, 0),
    SemanticClass.ROUTE: (0, 255, 0),
    SemanticClass.TRAFFIC_LIGHT_RED: (255, 64, 64),
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


def semantic_class_from_rgb(tile_rgb) -> SemanticClass | None:
    if tile_rgb is None:
        return None
    color = tuple(int(channel) for channel in np.asarray(tile_rgb, dtype=np.uint8).tolist())
    return _CLASS_BY_COLOR.get(color)
