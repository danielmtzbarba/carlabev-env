from __future__ import annotations

from dataclasses import dataclass
import math

import pygame

from CarlaBEV.semantics import VOID_ID, SemanticClass


@dataclass(frozen=True)
class FovRenderSpec:
    output_size: int
    ego_anchor_x_frac: float = 0.5
    ego_anchor_y_frac: float = 0.5
    mask_fov: bool = False
    mask_frac: float = 0.5


class FovRenderer:
    """Build the BEV observation (class-id raster) from the world-aligned id surface.

    The pipeline is crop → rotate (nearest neighbour, VOID fill) → compose →
    mask → ego overlay. All surfaces are 8-bit palettized class ids; RGB is a
    palette lookup performed by the consumer.
    """

    def __init__(self, spec: FovRenderSpec):
        self.spec = spec
        self.output_size = int(spec.output_size)
        self.output_resolution = (self.output_size, self.output_size)
        self.anchor_px = self._compute_anchor_px()
        self.crop_size = self._compute_crop_size()
        self.crop_resolution = (self.crop_size, self.crop_size)
        self._mask_polygons = self._build_mask_polygons() if spec.mask_fov else None
        self.ego_overlay_rect: pygame.Rect | None = None
        self._output: pygame.Surface | None = None  # preallocated class-id output surface

    def _compute_anchor_px(self) -> tuple[int, int]:
        max_idx = self.output_size - 1
        x = int(round(max_idx * self.spec.ego_anchor_x_frac))
        y = int(round(max_idx * self.spec.ego_anchor_y_frac))
        x = max(0, min(max_idx, x))
        y = max(0, min(max_idx, y))
        return x, y

    def _compute_crop_size(self) -> int:
        anchor_x, anchor_y = self.anchor_px
        max_x = max(anchor_x, (self.output_size - 1) - anchor_x)
        max_y = max(anchor_y, (self.output_size - 1) - anchor_y)
        radius = math.hypot(max_x, max_y)
        crop_size = int(math.ceil(2.0 * radius))
        return max(self.output_size, crop_size)

    def _build_mask_polygons(self):
        m = int(self.output_size * self.spec.mask_frac)
        n = self.output_size
        return (
            [(0, 0), (m, 0), (0, m)],
            [(n, 0), (n - m, 0), (n, m)],
            [(0, n), (0, n - m), (m, n)],
            [(n, n), (n - m, n), (n, n - m)],
        )

    def compute_crop_rect(self, crop_center: tuple[float, float], scene_size: tuple[int, int]) -> pygame.Rect:
        center_x = int(round(crop_center[0]))
        center_y = int(round(crop_center[1]))
        xmin = center_x - self.crop_size // 2
        ymin = center_y - self.crop_size // 2
        max_x = max(0, scene_size[0] - self.crop_size)
        max_y = max(0, scene_size[1] - self.crop_size)
        xmin = max(0, min(max_x, xmin))
        ymin = max(0, min(max_y, ymin))
        return pygame.Rect(xmin, ymin, self.crop_size, self.crop_size)

    def extract_crop(self, scene_surface: pygame.Surface, crop_rect: pygame.Rect) -> pygame.Surface:
        crop = scene_surface.subsurface(crop_rect)
        # rotate() fills the uncovered corners with the colorkey: make that VOID.
        # (The colorkey lives on the subsurface object; the parent is unaffected.)
        crop.set_colorkey(VOID_ID)
        return crop

    def rotate_crop(self, crop_surface: pygame.Surface, yaw_rad: float) -> tuple[pygame.Surface, pygame.Rect]:
        angle = math.degrees(yaw_rad) + 90
        rotated = pygame.transform.rotate(crop_surface, angle)
        rect = rotated.get_rect(center=self.anchor_px)
        return rotated, rect

    def compose_output(self, rotated_surface: pygame.Surface, rotated_rect: pygame.Rect) -> pygame.Surface:
        if self._output is None:
            from CarlaBEV.envs.world_renderer import new_id_surface

            self._output = new_id_surface(self.output_resolution, VOID_ID)
        output = self._output
        output.fill(VOID_ID)
        rotated_surface.set_colorkey(None)
        output.blit(rotated_surface, rotated_rect)
        return output

    def apply_mask(self, output_surface: pygame.Surface) -> pygame.Surface:
        if self._mask_polygons is not None:
            for points in self._mask_polygons:
                pygame.draw.polygon(output_surface, VOID_ID, points)
        return output_surface

    def draw_ego_overlay(self, output_surface: pygame.Surface, *, width_px: int, length_px: int, center_offset_px: float = 0.0) -> pygame.Rect:
        """Draw the ego footprint upright at the anchor (reference point on the anchor, body extending up)."""
        rect = pygame.Rect(0, 0, max(1, int(width_px)), max(1, int(length_px)))
        ax, ay = self.anchor_px
        rect.center = (int(round(ax)), int(round(ay - center_offset_px)))
        self.ego_overlay_rect = pygame.draw.rect(output_surface, int(SemanticClass.EGO), rect)
        return self.ego_overlay_rect

