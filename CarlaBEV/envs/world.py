import numpy as np
import math
import pygame

from .utils import load_map
from CarlaBEV.src.scenes.scene import Scene


class BaseMap(Scene):
    """
    A modular map interface for BEV-based environments.

    Handles map loading, cropping the FOV, rotation, and hero/world rendering.
    """

    def __init__(self, cfg, agent_class=None):
        self.cfg = cfg
        self.map_name = cfg.map_name
        self.AgentClass = agent_class
        self.size = cfg.size

        # --- Load map and create surfaces
        self._map_arr, self._map_img, _ = load_map(cfg.map_name, cfg.size)
        self._Y, self._X, _ = self._map_arr.shape
        self._scene = pygame.Surface((self._X, self._Y))

        # --- Rendering surfaces
        self.center = (self.size // 2, self.size // 2)
        self._fov_surface = pygame.Surface((self.size, self.size))
        self._pad = self.center[0]

        # --- Initialize base Scene
        super().__init__(size=cfg.size, screen=self._map_img)

        # --- Internal state
        self._theta = 0.0
        self._scene.blit(self._map_img, (0, 0))

    # =====================================================
    # --- Scene Control ---
    # =====================================================
    def reset(self, episode, actors=None):
        """Reset scene and reload actors."""
        super().reset_scene(episode, actors)
        self._theta = 0.0
        self._scene.blit(self._map_img, (0, 0))

    # =====================================================
    # --- FOV Handling ---
    # =====================================================
    def crop_fov(self, topleft):
        """Crop the field-of-view patch centered around hero."""
        self._xmin = np.clip(int(topleft.x), 0, self._X - self.size - self._pad - 1)
        self._ymin = np.clip(int(topleft.y), 0, self._Y - self.size - self._pad - 1)
        fov = self._scene.subsurface(
            (self._xmin, self._ymin, self.size + self._pad, self.size + self._pad)
        )
        return fov

    def rotate_fov(self, fov):
        """Rotate the cropped FOV according to hero yaw."""
        angle = math.degrees(self._theta) + 90
        rotated = pygame.transform.rotate(fov, angle)
        rect = rotated.get_rect(center=self.center)
        return rotated, rect

    # =====================================================
    # --- Simulation Step ---
    # =====================================================
    def step(self, camera_topleft):
        """Update scene and render the cropped, rotated FOV."""
        self._scene_step()

        # Crop around ego vehicle
        fov = self.crop_fov(camera_topleft)
        rotated_fov, rect = self.rotate_fov(fov)

        # Blit rotated FOV onto camera surface
        self._fov_surface.fill((0, 0, 0))
        self._fov_surface.blit(rotated_fov, rect)

        # Store agent tile (for reward)
        self._agent_tile = self._fov_surface.get_at(self.center)

        # Draw ego
        self.hero.draw(self.canvas, self.map_surface)

    def hero_step(self, action):
        """Advance hero agent one step and update heading."""
        self.hero.step(action)
        self._theta = self.hero.yaw

    # =====================================================
    # --- Metrics & Properties ---
    # =====================================================
    def dist2goal(self):
        """Euclidean distance to target."""
        return np.linalg.norm(self.hero.position - self.target_position)

    @property
    def canvas(self):
        return self._fov_surface

    @property
    def map_surface(self):
        return self._scene

    @property
    def agent_tile(self):
        return self._agent_tile
