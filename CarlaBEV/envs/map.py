import numpy as np
from copy import deepcopy
import pygame

from PIL import Image
from scipy.ndimage import shift, rotate

from skimage import draw

# home
map_path = "/home/danielmtz/Data/datasets/CarlaBEV/maps/Town01/Town01-padded.jpg"

# msi
# map_path = "/home/dan/Data/datasets/CarlaBEV/Town01-1024-RGB.jpg"


class Town01(object):
    _img = pygame.image.load(map_path)

    def __init__(self, window_size) -> None:
        self._map_arr = np.array(Image.open(map_path))
        self._win_size = window_size[0]
        self._Y, self._X, _ = self._map_arr.shape
        self._origin = ((self._Y / 2), int(self._X / 2))
        self._pad_rotation = 500
        self._theta = 0

    def crop_fov(self, topleft):
        self._xmin = np.clip(
            int(topleft.x), 0, self._X - self._win_size - self._pad_rotation - 1
        )
        self._ymin = np.clip(
            int(topleft.y), 0, self._Y - self._win_size - self._pad_rotation - 1
        )
        #
        self._fov = self._img.subsurface(
            (
                self._xmin,
                self._ymin,
                self._win_size + self._pad_rotation,
                self._win_size + self._pad_rotation,
            )
        )
        return self._fov

    def rotate_fov(self, pos, originPos=(512, 512)):
        image_rect = self._fov.get_rect(
            topleft=(pos[0] - originPos[0], pos[1] - originPos[1])
        )
        offset_center_to_pivot = pygame.math.Vector2(pos) - image_rect.center

        # roatated offset from pivot to center
        rotated_offset = offset_center_to_pivot.rotate(-self._theta)

        # roatetd image center
        rotated_image_center = (
            pos[0] - rotated_offset.x - self._pad_rotation,
            pos[1] - rotated_offset.y - self._pad_rotation,
        )

        # get a rotated image
        rotated_image = pygame.transform.rotate(self._fov, self._theta)
        rotated_image_rect = rotated_image.get_rect(center=(512, 512))

        return rotated_image, rotated_image_rect

    def blit(self, display, agent_pos):
        display.blit(self._img, agent_pos)

    def blitRotate(self, display, topleft, pos=(0, 0), originPos=(0, 0)):
        self.crop_fov(topleft)
        rotated_image, rotated_image_rect = self.rotate_fov(pos, originPos)

        display.blit(rotated_image, rotated_image_rect)
        return display

    def set_theta(self, theta):
        self._theta = theta

    @property
    def origin(self):
        return self._origin
