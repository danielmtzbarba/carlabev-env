"""Layered world renderer on a class-id raster.

The world is an 8-bit palettized pygame surface whose pixel values are
:class:`~CarlaBEV.semantics.SemanticClass` ids (``VOID_ID`` = nothing). One
drawing pass therefore yields the authoritative semantic raster; RGB images
are derived through the palette LUT only where a human/video consumer needs
them.

Layers
------
* **static** — the padded map (drivable / sidewalk / non-drivable ids), never
  redrawn;
* **dynamic** — actor footprints, route targets and traffic-light strips,
  drawn every step on top of the static layer. Instead of re-blitting the whole
  map to erase the previous frame (the former ``scene.redraw_world`` phase,
  ≈2 ms per step), only the *dirty rects* of the previous frame are restored
  from the static layer.

The renderer is a pure observer: it reads bodies/poses and never mutates
simulation state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

from CarlaBEV.semantics import PALETTE_LUT, VOID_ID, SemanticClass
from CarlaBEV.src.geometry.body import Body


_PALETTE: list[tuple[int, int, int]] | None = None


def make_palette() -> list[tuple[int, int, int]]:
    """256-entry RGB palette for class-id surfaces (cached: building it costs ~1 ms)."""
    global _PALETTE
    if _PALETTE is None:
        _PALETTE = [tuple(int(c) for c in PALETTE_LUT[i]) for i in range(256)]
    return _PALETTE


def make_id_surface(ids: np.ndarray) -> pygame.Surface:
    """8-bit palettized surface from an (H, W) uint8 class-id array."""
    ids = np.ascontiguousarray(np.asarray(ids, dtype=np.uint8).T)  # pygame is (W, H)
    surface = pygame.surfarray.make_surface(ids)
    surface.set_palette(make_palette())
    return surface


def new_id_surface(size: tuple[int, int], fill: int = VOID_ID) -> pygame.Surface:
    surface = pygame.Surface(size, depth=8)
    surface.set_palette(make_palette())
    surface.fill(fill)
    return surface


_LUT_C = np.ascontiguousarray(PALETTE_LUT)


def surface_ids(surface: pygame.Surface) -> np.ndarray:
    """(H, W) uint8 class ids of an 8-bit surface (copy)."""
    return np.array(pygame.surfarray.pixels2d(surface), dtype=np.uint8).T


def surface_rgb(surface: pygame.Surface) -> np.ndarray:
    """(H, W, 3) uint8 RGB of an 8-bit surface through the palette (single gather, no temporaries)."""
    ids = np.asarray(pygame.surfarray.pixels2d(surface)).T  # (H, W) view
    out = np.empty(ids.shape + (3,), dtype=np.uint8)
    np.take(_LUT_C, ids, axis=0, out=out)
    return out


@dataclass
class Drawable:
    """What the world renderer needs from a dynamic object."""

    class_id: int
    body: Body | None = None  # oriented footprint (vehicles, pedestrians, ego)
    rect_px: tuple[float, float, float, float] | None = None  # (left, top, w, h) in surface px (targets, strips)


class WorldRenderer:
    """Static id layer + dirty-rect dynamic layer, in the padded render frame."""

    def __init__(self, static_ids: np.ndarray, *, padding: int, metric, frame):
        self.metric = metric
        self.frame = frame
        self.padding = int(padding)
        height, width = static_ids.shape
        padded = np.full((height + 2 * self.padding, width + 2 * self.padding), int(SemanticClass.NON_DRIVABLE), dtype=np.uint8)
        padded[self.padding : self.padding + height, self.padding : self.padding + width] = static_ids
        self.static = make_id_surface(padded)
        self.world = self.static.copy()
        self.size = self.world.get_size()
        self._dirty: list[pygame.Rect] = []
        self.last_frame_rects = 0

    # --- frame lifecycle ------------------------------------------------------
    def reset(self) -> None:
        self.world.blit(self.static, (0, 0))
        self._dirty.clear()

    def begin_frame(self) -> None:
        """Erase last frame's dynamic pixels by restoring them from the static layer."""
        for rect in self._dirty:
            self.world.blit(self.static, rect.topleft, rect)
        self.last_frame_rects = len(self._dirty)
        self._dirty.clear()

    def _remember(self, rect: pygame.Rect) -> pygame.Rect:
        if rect.width > 0 and rect.height > 0:
            self._dirty.append(rect.inflate(2, 2).clip(self.world.get_rect()))
        return rect

    # --- primitives (class ids, metric bodies) --------------------------------
    def _to_px(self, x_m: float, y_m: float) -> tuple[float, float]:
        k = self.frame.pixels_per_world / self.metric.meters_per_pixel
        sign = 1.0 if self.frame.y_axis_down else -1.0
        return self.frame.origin.x + x_m * k, self.frame.origin.y + sign * y_m * k

    def draw_body(self, body: Body, class_id: int) -> pygame.Rect:
        fp = body.footprint
        if not fp.oriented:
            size_px = max(1, int(round(self.metric.distance_meters_to_surface(fp.length_m))))
            width_px = max(1, int(round(self.metric.distance_meters_to_surface(fp.width_m))))
            cx, cy = self._to_px(*body.center())
            rect = pygame.Rect(0, 0, width_px, size_px)
            rect.center = (round(cx), round(cy))
            return self._remember(pygame.draw.rect(self.world, int(class_id), rect))
        if fp.shape == "circle":
            cx, cy = self._to_px(*body.center())
            radius_px = max(1, int(round(self.metric.distance_meters_to_surface(fp.radius_m))))
            return self._remember(pygame.draw.circle(self.world, int(class_id), (int(round(cx)), int(round(cy))), radius_px))
        points = [self._to_px(x, y) for x, y in body.corner_tuples()]
        return self._remember(pygame.draw.polygon(self.world, int(class_id), points))

    def draw_surface_rect(self, left: float, top: float, width: float, height: float, class_id: int) -> pygame.Rect:
        """Axis-aligned rectangle given in *surface* px (converted into the render frame)."""
        origin = self.frame.world_to_surface((left, top))
        rect = pygame.Rect(int(round(origin.x)), int(round(origin.y)), max(1, int(round(width))), max(1, int(round(height))))
        return self._remember(pygame.draw.rect(self.world, int(class_id), rect))

    def draw_surface_square_centered(self, cx: float, cy: float, size_px: int, class_id: int) -> pygame.Rect:
        rect = self.frame.rect_from_world_center((cx, cy), (size_px, size_px))
        return self._remember(pygame.draw.rect(self.world, int(class_id), rect))

    def draw(self, drawable: Drawable) -> pygame.Rect | None:
        if drawable.body is not None:
            return self.draw_body(drawable.body, drawable.class_id)
        if drawable.rect_px is not None:
            left, top, width, height = drawable.rect_px
            return self.draw_surface_rect(left, top, width, height, drawable.class_id)
        return None

    # --- queries ----------------------------------------------------------------
    def ids(self) -> np.ndarray:
        return surface_ids(self.world)

    def rgb(self) -> np.ndarray:
        return surface_rgb(self.world)
