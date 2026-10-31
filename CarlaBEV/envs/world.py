"""World: map assets + scene orchestration + rendering, composed.

``BaseMap`` owns

* the **static semantic raster** of the map (class ids, the authority for tile
  queries),
* the :class:`~CarlaBEV.src.scenes.scene.Scene` (actors, clock, collision,
  metrics) — composed, not inherited,
* the :class:`~CarlaBEV.envs.world_renderer.WorldRenderer` (static layer +
  dirty-rect dynamic layer on class ids) and the
  :class:`~CarlaBEV.envs.fov.FovRenderer` (ego-aligned observation).

Rendering is a pure observer of the scene: ``step()`` advances the scene, then
draws the actors' bodies and composes the FOV. The observation is the FOV
class-id raster; RGB is a palette lookup.

Attribute access that is not found on the map is delegated to the scene
(``map.hero``, ``map.actor_manager``, ``map.clock`` …) so existing callers keep
working.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

from .utils import load_map
from .fov import FovRenderSpec, FovRenderer
from .transforms import SurfaceFrame
from .world_renderer import WorldRenderer, surface_ids, surface_rgb
from CarlaBEV.src.scenes.scene import Scene
from CarlaBEV.src.units import UnitContract
from CarlaBEV.src.diagnostics import profiling
from CarlaBEV.semantics import (
    BLOCKING_CLASSES,
    MAP_LABEL_TO_CLASS,
    PALETTE_LUT,
    SemanticClass,
    semantic_class_from_id,
    semantic_class_from_rgb,
)


@dataclass(frozen=True)
class RenderMapLayers:
    padding: int
    query_shape: tuple[int, int]
    render_shape: tuple[int, int]
    render_frame: SurfaceFrame


def map_labels_to_ids(label_array: np.ndarray) -> np.ndarray:
    """Semantic PNG labels (0/127/255) -> class ids."""
    ids = np.full(label_array.shape, int(SemanticClass.NON_DRIVABLE), dtype=np.uint8)
    for label, semantic_class in MAP_LABEL_TO_CLASS.items():
        ids[label_array == label] = int(semantic_class)
    return ids


class BaseMap:
    """Map + scene + renderers for BEV environments."""

    def __init__(self, cfg, agent_class=None, units: UnitContract | None = None):
        self.cfg = cfg
        self._units = units if units is not None else UnitContract.from_env_config(cfg)
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

        # --- Static semantic raster (authoritative class ids of the map)
        _, _, _, label_array = load_map(cfg.map_name, cfg.size, metric=self._units.metric, return_labels=True)
        self._static_ids = map_labels_to_ids(label_array)
        self._query_map_arr = PALETTE_LUT[self._static_ids]  # RGB view kept for compatibility
        self._drivable_mask = self._static_ids == int(SemanticClass.DRIVABLE)
        self._query_Y, self._query_X = self._static_ids.shape
        self._render_layers = self._build_render_layers()
        self.surface_frame = SurfaceFrame()
        self.render_frame = self._render_layers.render_frame
        self.world_renderer = WorldRenderer(
            self._static_ids,
            padding=self._render_layers.padding,
            metric=self._units.metric,
            frame=self.render_frame,
        )

        # --- Scene (composition)
        action_space = getattr(cfg, "action_mode", getattr(cfg, "action_space", "discrete"))
        self.scene = Scene(
            size=cfg.size,
            action_space=action_space,
            units=self._units,
            render_frame=self.render_frame,
        )
        self.crop_resolution = self.fov_renderer.crop_resolution
        self.scene.crop_resolution = self.crop_resolution
        self.scene.drivable_mask = self._drivable_mask
        self.scene.tile_provider = self

        # --- Observation surface and state
        self._fov_surface = None
        self._theta = 0.0
        self._agent_tile_class = None
        self._agent_tile = np.array([0, 0, 0], dtype=np.uint8)
        self.draw_fov()

    # ------------------------------------------------------------------ delegation
    def __getattr__(self, name):
        scene = self.__dict__.get("scene")
        if scene is not None and hasattr(scene, name):
            return getattr(scene, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    @property
    def units(self) -> UnitContract:
        return self._units

    @property
    def traffic_lifecycle(self):
        return self.scene.traffic_lifecycle

    @traffic_lifecycle.setter
    def traffic_lifecycle(self, value):
        self.scene.traffic_lifecycle = value

    @property
    def hero(self):
        return self.scene.hero

    def _build_render_layers(self) -> RenderMapLayers:
        padding = int(self.fov_renderer.crop_size)
        return RenderMapLayers(
            padding=padding,
            query_shape=(self._query_X, self._query_Y),
            render_shape=(self._query_X + 2 * padding, self._query_Y + 2 * padding),
            render_frame=SurfaceFrame(origin=(padding, padding)),
        )

    # =====================================================
    # --- Scene Control ---
    # =====================================================
    def reset(self, actors=None, hero_np_rng=None):
        """Reset scene, redraw the world and the first observation."""
        self.scene.reset_scene(actors, hero_np_rng=hero_np_rng)
        self._theta = 0.0
        self.world_renderer.reset()
        if self.scene.hero is not None and getattr(self.scene, "camera", None) is not None:
            self.scene.camera.scroll()
            self.draw_world()
            self.draw_fov()

    # =====================================================
    # --- Simulation Step ---
    # =====================================================
    def step(self, action):
        """Advance the scene one physics step, then render (legacy 1:1 entry point)."""
        self.advance(action)
        self.render_world()

    def advance(self, action):
        """Advance the scene one physics step **without drawing** (multi-rate sub-step).

        Everything the outcome needs (semantic tile under the ego, heading) is
        updated here so that collision/reward evaluation never depends on a
        rendered frame.
        """
        self.scene._scene_step(action)
        hero = self.scene.hero
        self._theta = hero.yaw
        self._agent_tile_class = self.semantic_class_at(hero.position)
        self._agent_tile = self.semantic_tile_at(hero.position)

    def render_world(self):
        """Render the dynamic world layer and the FOV observation from the current state."""
        with profiling.phase("scene.draw_actors"):
            self.draw_world()
        with profiling.phase("map.draw_fov"):
            self.draw_fov()

    def draw_world(self):
        """Dynamic layer: restore last frame's dirty rects, then draw current bodies."""
        self.world_renderer.begin_frame()
        for drawable in self.scene.drawables():
            self.world_renderer.draw(drawable)

    # =====================================================
    # --- Snapshots / presenter (read-only views) ---
    # =====================================================
    def snapshot(self):
        """Immutable record of the drawable state at the current simulated time."""
        from CarlaBEV.envs.presenter import WorldSnapshot

        return WorldSnapshot.capture(self.scene)

    def render_view(self, snapshot, fov_renderer=None):
        """Render a FOV view of a (possibly interpolated) snapshot into ``fov_renderer``'s output.

        Used by the presenter; it redraws the dynamic world layer from the
        snapshot (the next ``render_world()`` restores it from the current
        state) and never touches the observation surface of this map.
        """
        fov = self.fov_renderer if fov_renderer is None else fov_renderer
        self.world_renderer.begin_frame()
        for drawable in snapshot.drawables():
            self.world_renderer.draw(drawable)
        hero = snapshot.hero
        if hero is None:
            from .world_renderer import new_id_surface

            return new_id_surface(fov.output_resolution)
        pos_px = self._units.metric.meters_to_surface((hero.x_m, hero.y_m))
        surface_pos = self.render_frame.world_to_surface((float(pos_px[0]), float(pos_px[1])))
        half = fov.crop_size / 2.0
        crop_center = (int(surface_pos.x - half) + half, int(surface_pos.y - half) + half)
        crop_rect = fov.compute_crop_rect(crop_center, self._render_layers.render_shape)
        crop_surface = fov.extract_crop(self.world_renderer.world, crop_rect)
        rotated, rect = fov.rotate_crop(crop_surface, hero.yaw)
        output = fov.compose_output(rotated, rect)
        if self.mask_fov:
            fov.apply_mask(output)
        ego = self.scene.hero
        fov.draw_ego_overlay(
            output,
            width_px=ego.w,
            length_px=ego.l,
            center_offset_px=self._units.metric.distance_meters_to_surface(ego.footprint.center_offset_m),
        )
        return output

    # =====================================================
    # --- FOV Handling ---
    # =====================================================
    def compute_crop_rect(self):
        """Compute the world-aligned crop rectangle centered on the ego pose."""
        camera = self.scene.camera
        crop_center = (
            camera.offset.x + self.fov_renderer.crop_size / 2.0,
            camera.offset.y + self.fov_renderer.crop_size / 2.0,
        )
        return self.fov_renderer.compute_crop_rect(crop_center, self._render_layers.render_shape)

    def extract_crop(self, crop_rect):
        return self.fov_renderer.extract_crop(self.world_renderer.world, crop_rect)

    def rotate_crop(self, crop_surface):
        return self.fov_renderer.rotate_crop(crop_surface, self._theta)

    def compose_fov(self, rotated_surface, rotated_rect):
        self._fov_surface = self.fov_renderer.compose_output(rotated_surface, rotated_rect)

    def apply_fov_mask(self):
        self.fov_renderer.apply_mask(self._fov_surface)

    def draw_fov(self):
        hero = self.scene.hero
        if not getattr(self.scene, "camera", None) or hero is None:
            from .world_renderer import new_id_surface

            self._fov_surface = new_id_surface(self.fov_renderer.output_resolution)
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
        # Semantic tile from the authoritative static raster at the ego position.
        self._agent_tile_class = self.semantic_class_at(hero.position)
        self._agent_tile = self.semantic_tile_at(hero.position)
        # Ego overlay: reference point on the anchor, body extending forward (up).
        self.fov_renderer.draw_ego_overlay(
            self._fov_surface,
            width_px=hero.w,
            length_px=hero.l,
            center_offset_px=self._units.metric.distance_meters_to_surface(hero.footprint.center_offset_m),
        )

    # =====================================================
    # --- Semantic queries ---
    # =====================================================
    def _clip_xy(self, position):
        x = int(np.clip(round(float(position[0])), 0, self._query_X - 1))
        y = int(np.clip(round(float(position[1])), 0, self._query_Y - 1))
        return x, y

    def semantic_id_at(self, position) -> int:
        x, y = self._clip_xy(position)
        return int(self._static_ids[y, x])

    def semantic_tile_at(self, position):
        return np.array(PALETTE_LUT[self.semantic_id_at(position)], dtype=np.uint8)

    def semantic_class_at(self, position):
        return semantic_class_from_id(self.semantic_id_at(position))

    def is_obstacle_tile(self, tile):
        return semantic_class_from_rgb(tile) in BLOCKING_CLASSES

    # =====================================================
    # --- Observation ---
    # =====================================================
    def observation_ids(self) -> np.ndarray:
        """(size, size) uint8 class ids of the current FOV."""
        return surface_ids(self._fov_surface)

    def observation_rgb(self) -> np.ndarray:
        """(size, size, 3) uint8 RGB of the current FOV (palette lookup)."""
        return surface_rgb(self._fov_surface)

    # =====================================================
    # --- Properties ---
    # =====================================================
    @property
    def canvas(self):
        """Current FOV surface (8-bit class ids with the semantic palette)."""
        return self._fov_surface

    @property
    def map_surface(self):
        """World surface (static map + dynamic actors), 8-bit class ids."""
        return self.world_renderer.world

    @property
    def meters_per_pixel(self) -> float:
        return self._units.metric.meters_per_pixel

    def map_extent_meters(self) -> tuple[float, float]:
        return self._units.metric.map_extent_meters(self._query_X, self._query_Y)

    @property
    def agent_tile(self):
        if self.scene.hero is None:
            return np.array([0, 0, 0], dtype=np.uint8)
        return self.semantic_tile_at(self.scene.hero.position)

    @property
    def agent_tile_class(self):
        if self.scene.hero is None:
            return None
        return self.semantic_class_at(self.scene.hero.position)

    @property
    def static_ids(self) -> np.ndarray:
        return self._static_ids

    def world_surface_rgb(self) -> pygame.Surface:
        """RGB copy of the world surface for tools that need a displayable surface."""
        return self.world_renderer.world.convert(24)
