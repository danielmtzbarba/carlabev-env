from __future__ import annotations

import pygame


vec = pygame.math.Vector2


class SurfaceFrame:
    """
    Explicit transform between world coordinates and a pygame surface frame.

    Stage 1 uses the same numeric values for world and map-surface coordinates,
    but routing all conversions through this class keeps rendering concerns out
    of dynamics/control code and prepares the simulator for a metric world frame.
    """

    def __init__(
        self,
        pixels_per_world: float = 1.0,
        origin: tuple[float, float] = (0.0, 0.0),
        y_axis_down: bool = True,
    ) -> None:
        self.pixels_per_world = float(pixels_per_world)
        self.origin = vec(origin)
        self.y_axis_down = y_axis_down

    def world_to_surface(self, position) -> vec:
        x_w, y_w = float(position[0]), float(position[1])
        x_s = self.origin.x + x_w * self.pixels_per_world
        if self.y_axis_down:
            y_s = self.origin.y + y_w * self.pixels_per_world
        else:
            y_s = self.origin.y - y_w * self.pixels_per_world
        return vec(x_s, y_s)

    def surface_to_world(self, position) -> vec:
        x_s, y_s = float(position[0]), float(position[1])
        x_w = (x_s - self.origin.x) / self.pixels_per_world
        if self.y_axis_down:
            y_w = (y_s - self.origin.y) / self.pixels_per_world
        else:
            y_w = (self.origin.y - y_s) / self.pixels_per_world
        return vec(x_w, y_w)

    def rect_from_world_center(self, position, size) -> pygame.Rect:
        center = self.world_to_surface(position)
        width, height = int(size[0]), int(size[1])
        rect = pygame.Rect(0, 0, width, height)
        rect.center = (round(center.x), round(center.y))
        return rect

