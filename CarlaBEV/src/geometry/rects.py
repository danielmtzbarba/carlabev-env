"""Integer surface rectangles without pygame (legacy trigger semantics).

``pygame.Rect.colliderect`` treats rectangles as half-open intervals: two
rects overlap only if they share a positive area (touching edges do not count).
These helpers reproduce that exactly on plain tuples so that domain code
(actors, targets) does not depend on pygame.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from CarlaBEV.src.geometry.body import Body


class IntRect(NamedTuple):
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center(self) -> tuple[int, int]:
        return self.left + self.width // 2, self.top + self.height // 2

    @staticmethod
    def from_center(cx: float, cy: float, width: int, height: int) -> "IntRect":
        """Same rounding as ``pygame.Rect(0, 0, w, h)`` with ``rect.center = (round(cx), round(cy))``."""
        width, height = int(width), int(height)
        return IntRect(int(round(cx)) - width // 2, int(round(cy)) - height // 2, width, height)


def rects_overlap(a: IntRect, b: IntRect) -> bool:
    return a.left < b.right and b.left < a.right and a.top < b.bottom and b.top < a.bottom


def body_surface_rect(body: Body, metric) -> IntRect:
    """Axis-aligned surface-px rect enclosing a body (legacy rect for squares)."""
    fp = body.footprint
    if not fp.oriented:
        size_px = max(1, int(round(metric.distance_meters_to_surface(fp.length_m))))
        width_px = max(1, int(round(metric.distance_meters_to_surface(fp.width_m))))
        cx, cy = body.center()
        sx, sy = metric.meters_to_surface((cx, cy))
        return IntRect.from_center(float(sx), float(sy), width_px, size_px)
    x0, y0, x1, y1 = body.aabb()
    p0 = metric.meters_to_surface((x0, y0))
    p1 = metric.meters_to_surface((x1, y1))
    left, top = int(round(min(p0[0], p1[0]))), int(round(min(p0[1], p1[1])))
    right, bottom = int(round(max(p0[0], p1[0]))), int(round(max(p0[1], p1[1])))
    return IntRect(left, top, max(1, right - left), max(1, bottom - top))


def distance_between_centers(a: IntRect, b: IntRect) -> float:
    (ax, ay), (bx, by) = a.center, b.center
    return math.hypot(ax - bx, ay - by)
