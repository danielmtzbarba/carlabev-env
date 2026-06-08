from __future__ import annotations

from dataclasses import dataclass
import math

import pygame


@dataclass(frozen=True)
class FovRenderSpec:
    output_size: int
    ego_anchor_x_frac: float = 0.5
    ego_anchor_y_frac: float = 0.5
    mask_fov: bool = False
    mask_frac: float = 0.5


class FovRenderer:
    """Build the BEV observation image from a world-aligned scene surface."""

    def __init__(self, spec: FovRenderSpec):
        self.spec = spec
        self.output_size = int(spec.output_size)
        self.output_resolution = (self.output_size, self.output_size)
        self.anchor_px = self._compute_anchor_px()
        self.crop_size = self._compute_crop_size()
        self.crop_resolution = (self.crop_size, self.crop_size)
        self._mask_surface = self._build_mask_surface() if spec.mask_fov else None

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

    def _build_mask_surface(self) -> pygame.Surface:
        surface = pygame.Surface(self.output_resolution)
        surface.fill((0, 0, 0))
        m = int(self.output_size * self.spec.mask_frac)
        transparent = (0, 0, 0, 0)
        opaque = (0, 0, 0, 255)

        mask = pygame.Surface(self.output_resolution, flags=pygame.SRCALPHA)
        mask.fill(transparent)

        cutouts = (
            [(0, 0), (m, 0), (0, m)],
            [(self.output_size, 0), (self.output_size - m, 0), (self.output_size, m)],
            [(0, self.output_size), (0, self.output_size - m), (m, self.output_size)],
            [
                (self.output_size, self.output_size),
                (self.output_size - m, self.output_size),
                (self.output_size, self.output_size - m),
            ],
        )
        for points in cutouts:
            pygame.draw.polygon(mask, opaque, points)
        return mask

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
        return scene_surface.subsurface(crop_rect)

    def rotate_crop(self, crop_surface: pygame.Surface, yaw_rad: float) -> tuple[pygame.Surface, pygame.Rect]:
        angle = math.degrees(yaw_rad) + 90
        rotated = pygame.transform.rotate(crop_surface, angle)
        rect = rotated.get_rect(center=self.anchor_px)
        return rotated, rect

    def compose_output(self, rotated_surface: pygame.Surface, rotated_rect: pygame.Rect) -> pygame.Surface:
        output = pygame.Surface(self.output_resolution)
        output.fill((0, 0, 0))
        output.blit(rotated_surface, rotated_rect)
        return output

    def apply_mask(self, output_surface: pygame.Surface) -> pygame.Surface:
        if self._mask_surface is not None:
            output_surface.blit(self._mask_surface, (0, 0))
        return output_surface

