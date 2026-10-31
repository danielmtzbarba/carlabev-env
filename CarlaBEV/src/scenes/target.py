"""Route targets (checkpoints and goal): axis-aligned surface squares that trigger once."""

from __future__ import annotations

import numpy as np

from CarlaBEV.semantics import SemanticClass
from CarlaBEV.src.geometry.rects import IntRect, rects_overlap


class Target:
    semantic_class = SemanticClass.ROUTE

    def __init__(self, id, target_pos, color=None, size=1, scale=1):
        self.id = id
        self.color = color
        self.lenght = int(size)
        self.width = int(size)
        self.scale = scale
        self._visible = False
        self.x0, self.y0 = float(target_pos[0]), float(target_pos[1])
        self.position = np.array([self.x0, self.y0], dtype=float)

    def reset(self):
        self._visible = True
        self.position = np.array([self.x0, self.y0], dtype=float)

    def surface_rect(self) -> IntRect:
        return IntRect.from_center(self.position[0], self.position[1], self.width, self.lenght)

    def rect_px(self) -> tuple[float, float, float, float]:
        """(left, top, width, height) in surface px for the renderer."""
        rect = self.surface_rect()
        return rect.left, rect.top, rect.width, rect.height

    def step(self, t, dt):
        pass

    def trigger(self, hero_rect: IntRect):
        """Return (id, triggered). A visible target triggers once when the ego rect overlaps it."""
        result = None
        if self._visible:
            result = rects_overlap(hero_rect, self.surface_rect())
            if result:
                self._visible = False
        return self.id, result

    def isCollided(self, hero, offset=None):  # legacy name
        return (*self.trigger(hero.surface_rect()), 9999)

    @property
    def visible(self):
        return self._visible

    @property
    def pose(self):
        return np.array([self.position[0], self.position[1], 0.0], dtype=float)
