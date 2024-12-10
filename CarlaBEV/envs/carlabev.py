from enum import Enum
from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
from numpy._core.records import array
import pygame
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import shift

from skimage import draw

# home
map_path = "/home/danielmtz/Data/datasets/CarlaBEV/maps/Town01/Town01-1024-RGB.jpg"
# msi
map_path = "/home/dan/Data/datasets/CarlaBEV/Town01-1024-RGB.jpg"


class Actions(Enum):
    nothing = 0
    left = 1
    right = 2
    gas = 3
    brake = 4


class Vehicle(object):
    def __init__(self) -> None:
        self._gas = 0.0
        self._steer = 0.0
        self._theta = 0.0
        self._brake = 0.0

    def set_location(self, agent_spawn_loc):
        #        self._agent_location = agent_spawn_loc
        self._agent_location = np.array([512, 512])

    def step(self, action, map_size):
        self._agent_location = np.clip(self._agent_location + action, 0, map_size - 1)

    def gas(self, gas):
        """control: rear wheel drive

        Args:
            gas (float): How much gas gets applied. Gets clipped between 0 and 1.
        """
        gas = np.clip(gas, 0, 1)
        diff = gas - self._gas
        if diff > 0.1:
            diff = 0.1  # gradually increase, but stop immediately
        self._gas = diff

    def brake(self, b):
        """control: brake

        Args:
            b (0..1): Degree to which the brakes are applied. More than 0.9 blocks the wheels to zero rotation"""
        self._brake = np.clip(b, 0, 1)

    def steer(self, s):
        """control: steer

        Args:
            s (-1..1): target position, it takes time to rotate steering wheel from side-to-side"""
        self._steer = np.clip(s, -1, 1)

    @property
    def agent_location(self) -> np.array:
        return self._agent_location


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

    def move_sliding_window(self, direction):
        self._realx += direction[0]
        self._realy += direction[1]

        self._xmin = np.clip(self._xmin + direction[0], 0, self._X - self._win_size - 1)
        self._ymin = np.clip(self._ymin + direction[1], 0, self._Y - self._win_size - 1)
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


class CarlaBEV(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, size=1024):
        self.size = size  # The size of the square grid
        self.window_size = 1024  # The size of the PyGame window

        self.window_center = (self.window_size / 2, self.window_size / 2)

        self.map = Map()
        self.hero = Vehicle()

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(size, size, 3), dtype=np.uint8
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(5)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.nothing.value: np.array([0, 16]),
            Actions.left.value: np.array([0, 16]),
            Actions.right.value: np.array([0, 16]),
            Actions.gas.value: np.array([0, 16]),
            Actions.brake.value: np.array([0, 16]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._render_frame()

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self.hero.agent_location - self.map.target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        agent_spawn_loc = self.np_random.integers(0, self.size, size=2, dtype=int)
        self.hero.set_location(agent_spawn_loc)

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        target_location = self.hero.agent_location
        while np.array_equal(target_location, self.hero.agent_location):
            target_location = self.np_random.integers(0, 6000, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self.hero.step(direction, self.size)
        self.map.move_sliding_window(direction)

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self.hero.agent_location, self.map.target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.blit(self.map.get_map(), (0, 0))

        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            self.window_center,
            8,
        )

        # Finally, add some gridlines
        for x in np.arange(0, self.window_size + 16, 16):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

        else:  # rgb_array
            rgb_array = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
            return rgb_array

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
