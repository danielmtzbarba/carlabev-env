import numpy as np
from copy import deepcopy
import pygame

from PIL import Image
from scipy.ndimage import shift

from skimage import draw

# home
map_path = "/home/danielmtz/Data/datasets/CarlaBEV/maps/Town01/Town01-1024-RGB.jpg"

# msi
# map_path = "/home/dan/Data/datasets/CarlaBEV/Town01-1024-RGB.jpg"


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
