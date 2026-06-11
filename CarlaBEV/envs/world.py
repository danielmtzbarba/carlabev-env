from dataclasses import dataclass

import numpy as np
import pygame

from .utils import load_map
from .fov import FovRenderSpec, FovRenderer
from .transforms import SurfaceFrame
from CarlaBEV.src.scenes.scene import Scene
from CarlaBEV.semantics import (
    BLOCKING_CLASSES,
    SemanticClass,
    semantic_class_from_rgb,
    semantic_color_tuple,
)


@dataclass(frozen=True)
class RenderMapLayers:
    padding: int
    query_shape: tuple[int, int]
    render_shape: tuple[int, int]
    render_frame: SurfaceFrame


class BaseMap(Scene):
    """
    A modular map interface for BEV-based environments.

    Handles map loading, cropping the FOV, rotation, and hero/world rendering.
    """

    def __init__(self, cfg, agent_class=None):
        self.cfg = cfg
        self.mask_fov = self.cfg.fov_masked
        self.map_name = cfg.map_name
        self.AgentClass = agent_class
        self.size = cfg.size
        self.fov_renderer = FovRenderer(
            FovRenderSpec(
                output_size=self.size,
                ego_anchor_x_frac=getattr(self.cfg, "ego_anchor_x_frac", 0.5),
                ego_anchor_y_frac=getattr(self.cfg, "ego_anchor_y_frac", 0.5),
                mask_fov=self.mask_fov,
            )
        )

        # --- Load map and create query/render surfaces
        self._query_map_arr, base_map_img, _ = load_map(cfg.map_name, cfg.size)
        self._query_Y, self._query_X, _ = self._query_map_arr.shape
        self._render_layers = self._build_render_layers(base_map_img)
        self._map_img = self._build_padded_render_map(base_map_img)
        self._scene = pygame.Surface(self._render_layers.render_shape)
        self.surface_frame = SurfaceFrame()

        # --- Rendering surfaces
        self._fov_surface = pygame.Surface(self.fov_renderer.output_resolution)
        self.crop_resolution = self.fov_renderer.crop_resolution

        # --- Initialize base Scene
        action_space = getattr(cfg, "action_mode", getattr(cfg, "action_space", "discrete"))
        super().__init__(size=cfg.size, screen=self._map_img, action_space=action_space)
        self.render_frame = self._render_layers.render_frame

        # --- Internal state
        self._theta = 0.0
        self._scene.blit(self._map_img, (0, 0))

    def _build_render_layers(self, base_map_img: pygame.Surface) -> RenderMapLayers:
        padding = int(self.fov_renderer.crop_size)
        render_w = self._query_X + 2 * padding
        render_h = self._query_Y + 2 * padding
        return RenderMapLayers(
            padding=padding,
            query_shape=(self._query_X, self._query_Y),
            render_shape=(render_w, render_h),
            render_frame=SurfaceFrame(origin=(padding, padding)),
        )

    def _build_padded_render_map(self, base_map_img: pygame.Surface) -> pygame.Surface:
        render_map = pygame.Surface(self._render_layers.render_shape)
        render_map.fill(semantic_color_tuple(SemanticClass.NON_DRIVABLE))
        render_map.blit(
            base_map_img,
            (self._render_layers.padding, self._render_layers.padding),
        )
        return render_map

    # =====================================================
    # --- Scene Control ---
    # =====================================================
    def reset(self, actors=None, hero_np_rng=None):
        """Reset scene and reload actors."""
        self.reset_scene(actors, hero_np_rng=hero_np_rng)
        self._theta = 0.0
        self._scene.blit(self._map_img, (0, 0))
        if getattr(self, "hero", None) is not None and getattr(self, "camera", None) is not None:
            self.hero.sync_rect(self.render_frame)
            self.camera.scroll()
            self.draw_fov()

    # =====================================================
    # --- FOV Handling ---
    # =====================================================
    def compute_crop_rect(self):
        """Compute the world-aligned crop rectangle centered on the ego pose."""
        crop_center = (
            self.camera.offset.x + self.fov_renderer.crop_size / 2.0,
            self.camera.offset.y + self.fov_renderer.crop_size / 2.0,
        )
        return self.fov_renderer.compute_crop_rect(crop_center, self._render_layers.render_shape)

    def extract_crop(self, crop_rect):
        """Extract the world-aligned crop patch from the scene surface."""
        return self.fov_renderer.extract_crop(self._scene, crop_rect)

    def rotate_crop(self, crop_surface):
        """Rotate the crop into the ego-aligned observation frame."""
        return self.fov_renderer.rotate_crop(crop_surface, self._theta)

    def compose_fov(self, rotated_surface, rotated_rect):
        """Compose the rotated crop into the final square observation surface."""
        self._fov_surface = self.fov_renderer.compose_output(rotated_surface, rotated_rect)

    def apply_fov_mask(self):
        """Apply static mask geometry after composing the observation image."""
        self.fov_renderer.apply_mask(self._fov_surface)

    # =====================================================
    # --- Simulation Step ---
    # =====================================================
    def step(self, action):
        """Update scene and render the cropped, rotated FOV."""
        self._scene_step(action)
        self.draw_fov()
    
    def draw_fov(self):
        if not getattr(self, "camera", None) or not getattr(self, "hero", None):
            self._fov_surface.fill((0, 0, 0))
            if self.mask_fov:
                self.apply_fov_mask()
            self._agent_tile = np.array([0, 0, 0], dtype=np.uint8)
            self._agent_tile_class = None
            return

        crop_rect = self.compute_crop_rect()
        crop_surface = self.extract_crop(crop_rect)
        rotated_fov, rect = self.rotate_crop(crop_surface)
        self.compose_fov(rotated_fov, rect)
        if self.mask_fov:
            self.apply_fov_mask()
        # Store semantic tile from authoritative map coordinates.
        self._agent_tile = self.semantic_tile_at(self.hero.position)
        self._agent_tile_class = self.semantic_class_at(self.hero.position)
        # Draw ego
        self.hero.set_fov_anchor(self.fov_renderer.anchor_px)
        self.hero.draw(self.canvas, self.map_surface)

    def semantic_tile_at(self, position):
        x = int(np.clip(round(float(position.x)), 0, self._query_X - 1))
        y = int(np.clip(round(float(position.y)), 0, self._query_Y - 1))
        return np.array(self._query_map_arr[y, x], dtype=np.uint8)

    def semantic_class_at(self, position):
        return semantic_class_from_rgb(self.semantic_tile_at(position))

    def is_obstacle_tile(self, tile):
        return semantic_class_from_rgb(tile) in BLOCKING_CLASSES

    # =====================================================
    # --- Properties ---
    # =====================================================
    @property
    def canvas(self):
        return self._fov_surface

    @property
    def map_surface(self):
        return self._scene

    @property
    def agent_tile(self):
        if getattr(self, "hero", None) is None:
            return np.array([0, 0, 0], dtype=np.uint8)
        return self.semantic_tile_at(self.hero.position)

    @property
    def agent_tile_class(self):
        if getattr(self, "hero", None) is None:
            return None
        return self.semantic_class_at(self.hero.position)
