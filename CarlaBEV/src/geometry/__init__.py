"""Actor geometry: metric footprints, oriented bodies, the collision system and rect helpers."""

from CarlaBEV.src.geometry.body import Body, Footprint, OrientedBox
from CarlaBEV.src.geometry.collision import CollisionSystem, ContactInfo
from CarlaBEV.src.geometry.rects import IntRect, body_surface_rect, rects_overlap

__all__ = [
    "Body",
    "Footprint",
    "OrientedBox",
    "CollisionSystem",
    "ContactInfo",
    "IntRect",
    "body_surface_rect",
    "rects_overlap",
]
