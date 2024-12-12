import numpy as np
from copy import deepcopy
import pygame

from PIL import Image
from scipy.ndimage import shift, rotate

from skimage import draw

# home
# map_path = "/home/danielmtz/Data/datasets/CarlaBEV/maps/Town01/Town01-1024-RGB.jpg"

# msi
map_path = "/home/dan/Data/datasets/CarlaBEV/Town01-1024-RGB.jpg"


class Town01(object):
    _img = pygame.image.load(map_path)

    def __init__(self, window_size) -> None:
        self._map_arr = np.array(Image.open(map_path))
        self._win_size = window_size[0]
        self._Y, self._X, _ = self._map_arr.shape
        self._origin = ((self._Y / 2), int(self._X / 2))
        self._theta = 0

    def blit_fov(self, display, agent_pos, rotate_fov=False):
        padding = 500
        self._xmin = int(
            np.clip(
                abs(agent_pos[0]) - int(padding / 2), 0, self._X - self._win_size - 1
            )
        )
        self._ymin = int(
            np.clip(
                abs(agent_pos[1]) - int(padding / 2), 0, self._Y - self._win_size - 1
            )
        )
        print(self._xmin, self._ymin)

        fov = deepcopy(self._map_arr)
        fov = fov[
            self._xmin - int(padding / 2) : self._xmin
            + self._win_size
            + int(padding / 2),
            self._ymin - int(padding / 2) : self._ymin
            + self._win_size
            + int(padding / 2),
        ]
        self._fov = rotate(fov, self._theta)
        self._fov = pygame.surfarray.make_surface(np.swapaxes(fov, 1, 0))
        if rotate_fov:
            self.blitRotate(display)
        else:
            display.blit(self._fov, (0, 0))

    def blit(self, display, agent_pos):
        display.blit(pygame.surfarray.make_surface(self._img), agent_pos)

    def blitRotate(self, display, pos=(0, 0), originPos=(0, 0)):
        # offset from pivot to center
        image_rect = self._fov.get_rect(
            topleft=(pos[0] - originPos[0], pos[1] - originPos[1])
        )
        offset_center_to_pivot = pygame.math.Vector2(pos) - image_rect.center

        # roatated offset from pivot to center
        rotated_offset = offset_center_to_pivot.rotate(-self._theta)

        # roatetd image center
        rotated_image_center = (pos[0] - rotated_offset.x, pos[1] - rotated_offset.y)

        # get a rotated image
        rotated_image = pygame.transform.rotate(self._fov, self._theta)
        rotated_image_rect = rotated_image.get_rect(center=rotated_image_center)

        # rotate and blit the image
        display.blit(rotated_image, rotated_image_rect)

    def set_theta(self, theta):
        self._theta = theta

    @property
    def origin(self):
        return self._origin


class Map(object):
    def __init__(self) -> None:
        self.reset()
        self.set_target_location(np.array([7600, 1200]))

    def set_target_location(self, location):
        self._target_location = location
        rr, cc = draw.disk(location, radius=32, shape=self._map_arr.shape)
        self._map_arr[rr, cc] = (255, 0, 0)

    def reset(self):
        self._map_arr = np.array(Image.open(map_path))
        self._X, self._Y, _ = self._map_arr.shape
        self._win_size = 1024
        #
        self._ymin = 0
        self._xmin = self._X - self._win_size
        #
        self._realx, self._realy = deepcopy(self._xmin), deepcopy(self._ymin)
        self._outx, self._outy = 0, 0
        self.move_sliding_window([0, 0])

    def shift(self):
        fov_shifted = deepcopy(self._fov)

        if self._realx < 0:
            self._fov = shift(
                fov_shifted,
                shift=(-1 * (self._realx), 0, 0),
                mode="constant",
                cval=0,
            )
        elif self._realx + self._win_size >= self._X:
            self._fov = shift(
                fov_shifted,
                shift=(self._X - 1 - self._realx - self._win_size, 0, 0),
                mode="constant",
                cval=0,
            )

        if self._realy < 0:
            self._fov = shift(
                fov_shifted,
                shift=(0, -1 * (self._realy), 0),
                mode="constant",
                cval=0,
            )
        elif self._realy + self._win_size >= self._Y:
            self._fov = shift(
                fov_shifted,
                shift=(0, self._Y - 1 - self._realy - self._win_size, 0),
                mode="constant",
                cval=0,
            )

    def move_sliding_window(self, movement):
        self._realx += movement[0]
        self._realy += movement[1]

        self._xmin = np.clip(self._xmin + movement[0], 0, self._X - self._win_size - 1)
        self._ymin = np.clip(self._ymin + movement[1], 0, self._Y - self._win_size - 1)
        #
        self._fov = deepcopy(self._map_arr)
        self._fov = self._fov[
            self._xmin : self._xmin + self._win_size,
            self._ymin : self._ymin + self._win_size,
        ]
        if (
            self._realx < 0
            or self._realy < 0
            or self._realx + self._win_size >= self._X
            or self._realy + self._win_size >= self._Y
        ):
            self.shift()

    def _get_fov(self):
        return self._fov

    def _preprocess(self, arr):
        return pygame.surfarray.make_surface(np.swapaxes(arr, 1, 0))

    def get_map(self):
        return self._preprocess(self._get_fov())

    @property
    def map(self) -> np.array:
        return self._map_arr

    @property
    def target_location(self) -> np.array:
        return self._target_location
