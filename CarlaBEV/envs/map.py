import numpy as np
import math
import pygame

from PIL import Image

# home
map_path = "/home/danielmtz/Data/datasets/CarlaBEV/maps/Town01/Town01-padded.jpg"

# msi
# map_path = "/home/dan/Data/datasets/CarlaBEV/Town01-1024-RGB.jpg"


class Target(pygame.sprite.Sprite):
    _target_img = pygame.image.load("rectangle-16.png")

    def __init__(self, target_location):
        pygame.sprite.Sprite.__init__(self)
        self.rect = self._target_img.get_rect()
        x, y = target_location[0], target_location[1]
        self.rect.center = (x, y)
        self.position = pygame.math.Vector2(x, y)

    def draw(self, map):
        print(self.position)
        map.blit(self._target_img, self.rect)


class Town01(object):
    _map_arr = np.array(Image.open(map_path))
    _map_img = pygame.image.load(map_path)
    _Y, _X, _ = _map_arr.shape

    def __init__(self, window_size, target_location) -> None:
        self._map_surface = pygame.Surface((self._X, self._Y))
        self._win_size = window_size[0]

        self.target_on_map = False
        self._target = Target(target_location)

        self.draw_map()

        self._pad_rotation = 500
        self._theta = 0

    def draw_map(self):
        self._map_surface.blit(self._map_img, (0, 0))
        self._target.draw(self._map_surface)
        pygame.image.save(self._map_surface, "map.png")

    def crop_fov(self, topleft):
        self._xmin = np.clip(
            int(topleft.x), 0, self._X - self._win_size - self._pad_rotation - 1
        )
        self._ymin = np.clip(
            int(topleft.y), 0, self._Y - self._win_size - self._pad_rotation - 1
        )
        #
        self._fov = self._map_surface.subsurface(
            (
                self._xmin,
                self._ymin,
                self._win_size + self._pad_rotation,
                self._win_size + self._pad_rotation,
            )
        )
        return self._fov

    def rotate_fov(self, pos, originPos=(512, 512)):
        # get a rotated image
        if np.math.degrees(self._theta) > 90:
            pass

        rotated_image = pygame.transform.rotate(self._fov, np.math.degrees(self._theta))
        rotated_image_rect = rotated_image.get_rect(center=(512, 512))

        return rotated_image, rotated_image_rect

    def blitRotate(self, display, topleft, pos=(0, 0), originPos=(0, 0)):
        self.crop_fov(topleft)
        rotated_image, rotated_image_rect = self.rotate_fov(pos, originPos)

        display.blit(rotated_image, rotated_image_rect)
        self._agent_tile = display.get_at((512, 512))

        return display

    def set_theta(self, theta):
        self._theta = theta

    @property
    def agent_tile(self):
        return self._agent_tile
