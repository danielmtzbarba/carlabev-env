import numpy as np
import math
import pygame
import pygame.surfarray as surfarray


from .utils import load_map
from CarlaBEV.src.scenes.scene import Scene


class Town01(object):
    def __init__(self, target_id, size=1024, scale=1) -> None:
        self._map_arr, self._map_img = load_map(size)
        self._Y, self._X, _ = self._map_arr.shape
        self.size = size  # The size of the square grid
        self.center = (int(self.size / 2), int(self.size / 2))
        self._map_surface = pygame.Surface((self._X, self._Y))
        self._fov_surface = pygame.Surface((self.size, self.size))

        self._pad_rotation = self.center[0]
        self._theta = 0

        self.draw_map()

    def draw_map(self):
        self._map_surface.blit(self._map_img, (0, 0))
        self._scene = Scene(map_surface=self._map_surface, size=self.size)

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

    def step(self, topleft):
        self._scene.step()
        self.crop_fov(topleft)
        rotated_image, rotated_image_rect = self.rotate_fov()
        self._fov_surface.blit(rotated_image, rotated_image_rect)
        self._agent_tile = self._fov_surface.get_at(self.center)

    def has_collided(self, vehicle_rect, class_color):
        pixels = surfarray.pixels3d(
            self._fov_surface.subsurface(
                pygame.Rect(*self.center, vehicle_rect[2], vehicle_rect[3])
            )
        )
        if class_color in pixels:
            return True
        return False

    def set_theta(self, theta):
        self._theta = theta

    def got_target(self, hero):
        return self._scene.got_target(hero)

    def hit_pedestrian(self, hero):
        return self._scene.hit_pedestrian(hero)

    def next_target(self, target_id):
        self._scene.next_target(target_id)

    @property
    def canvas(self):
        return self._fov_surface

    @property
    def agent_tile(self):
        return self._agent_tile

    @property
    def target_pose(self):
        return self._scene.target_pose

    @property
    def target_position(self):
        return self._scene.target_position
