import numpy as np
import math
import pygame

from .utils import load_map
from CarlaBEV.src.scenes.scene import Scene


class Town01(object):
    def __init__(self, size=1024) -> None:
        self._map_arr, self._map_img, _ = load_map(size)
        self._Y, self._X, _ = self._map_arr.shape
        self.size = size  # The size of the square grid
        self.center = (int(self.size / 2), int(self.size / 2))
        self._map_surface = pygame.Surface((self._X, self._Y))
        self._fov_surface = pygame.Surface((self.size, self.size))

        self._pad_rotation = self.center[0]
        self._theta = 0
        #
        self._map_surface.blit(self._map_img, (0, 0))
        self._scene = Scene(map_surface=self._map_surface, size=self.size)
        #
        self.reset()

    def reset(self):
        self._scene.reset()

    def crop_fov(self, topleft):
        self._xmin = np.clip(
            int(topleft.x), 0, self._X - self.size - self._pad_rotation - 1
        )
        self._ymin = np.clip(
            int(topleft.y), 0, self._Y - self.size - self._pad_rotation - 1
        )
        #
        self._fov = self._map_surface.subsurface(
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
        self._scene.step(course)
        self.crop_fov(topleft)
        rotated_image, rotated_image_rect = self.rotate_fov()
        self._fov_surface.blit(rotated_image, rotated_image_rect)
        self._agent_tile = self._fov_surface.get_at(self.center)

    def set_theta(self, theta):
        self._theta = theta

    def check_collision(self, hero):
        return self._scene.collision_check(hero)

    def dist2target(self, hero_position):
        return np.linalg.norm(hero_position - self.target_position, ord=2)

    @property
    def canvas(self):
        return self._fov_surface
    
    @property
    def map_surface(self):
        return self._map_surface

    @property
    def agent_tile(self):
        return self._agent_tile

    @property
    def agent_route(self):
        return self._scene.agent_route

    # Target
    @property
    def num_targets(self):
        return self._scene.num_targets

    @property
    def target_pose(self):
        return self._scene.target_pose

    @property
    def target_position(self):
        return self._scene.target_position
