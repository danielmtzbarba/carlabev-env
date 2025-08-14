import numpy as np
import math
import pygame

from .utils import load_map
from CarlaBEV.src.scenes.scene import Scene

#
class Town01(Scene):
    def __init__(self, size=1024) -> None:
        self.size = size  # The size of the square grid
        self._map_arr, self._map_img, _ = load_map(size)
        self._Y, self._X, _ = self._map_arr.shape
        self._scene = pygame.Surface((self._X, self._Y))
        Scene.__init__(self, size)
        #
        self.center = (int(self.size / 2), int(self.size / 2))
        self._fov_surface = pygame.Surface((self.size, self.size))
        self._pad_rotation = self.center[0]
        self._theta = 0
        self._scene.blit(self._map_img, (0, 0))

    def _crop_fov(self, topleft):
        self._xmin = np.clip(
            int(topleft.x), 0, self._X - self.size - self._pad_rotation - 1
        )
        self._ymin = np.clip(
            int(topleft.y), 0, self._Y - self.size - self._pad_rotation - 1
        )
        #
        self._fov = self._scene.subsurface(
            (
                self._xmin,
                self._ymin,
                self.size + self._pad_rotation,
                self.size + self._pad_rotation,
            )
        )
        return self._fov

    def rotate_fov(self):
        # get a rotated image
        if math.degrees(self._theta) > 90:
            pass

        rotated_image = pygame.transform.rotate(
            self._fov, math.degrees(self._theta) + 90
        )
        rotated_image_rect = rotated_image.get_rect(center=self.center)

        return rotated_image, rotated_image_rect

    def step(self, topleft, course):
        self._scene_step(course)
        self._crop_fov(topleft)
        rotated_image, rotated_image_rect = self.rotate_fov()
        self._fov_surface.blit(rotated_image, rotated_image_rect)
        self._agent_tile = self._fov_surface.get_at(self.center)

    def set_theta(self, theta):
        self._theta = theta

    def dist2target(self, hero_position):
        return np.linalg.norm(hero_position - self.target_position, ord=2)

    @property
    def canvas(self):
        return self._fov_surface
    
    @property
    def map_surface(self):
        return self._scene

    @property
    def scene_surface(self):
        return self._scene

    @property
    def agent_tile(self):
        return self._agent_tile
